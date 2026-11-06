"""A unified pool gives a request's pages back only after the last forward that
includes the request has been processed.

Under the overlap scheduler, the forward launched before a result is processed
still reads the pages of a request that result finishes, and writes its next
slot. A unified pool hands a freed page out again from the schedule stream
(zeroing it, or copying into it while compacting), which is not ordered after
that forward. So the request's frees and its prefix-tree unlock are queued and
run once no unprocessed forward includes it; the insert into the tree happens at
once. Admission counts the queued pages as soon-free, never as free now.
"""

import unittest
from array import array
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.managers.req_types import FINISH_LENGTH
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.mem_cache.base_prefix_cache import InsertParams, MatchPrefixParams
from sglang.srt.mem_cache.common import checkpoint_kv_cache, release_kv_cache
from sglang.srt.mem_cache.kv.alloc.band import Band
from sglang.srt.mem_cache.kv.alloc.release_queue import ReleaseQueue
from sglang.srt.mem_cache.kv.layout.mha import MHASubPoolSpec, MHAView
from sglang.srt.mem_cache.kv.layout.state import MambaSubPoolSpec
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.static_kv_storage import acquire_rows
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.test.ci.ci_register import register_cpu_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase
from sglang.test.unified_allocator_fixtures import (
    arena_of,
    build_swa_cache,
    build_swa_pool,
    setup_allocator_context,
)

register_cpu_ci(est_time=10, suite="base-a-test-cpu")
register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")


class _InFlight:
    """A request stand-in a forward still includes until ``retire``."""

    def __init__(self):
        self.inflight_steps = 1

    def retire(self):
        self.inflight_steps = 0


def _physical_pages(band, token_ids):
    pages = token_ids.to(torch.int64)[:: band.page_size] // band.page_size
    return band.ids.virtual_to_physical[pages].clone()


class TestHeldPagesStayBound(CustomTestCase):
    """A held free keeps its pages bound: no take, no move into them."""

    def setUp(self):
        super().setUp()
        setup_allocator_context()
        self.allocator = build_swa_pool(page_size=4, lazy=True)
        self.queue = self.allocator.release_queue
        self.owner = self.allocator.full_attn_allocator

    def test_only_a_lazy_chain_holds(self):
        self.assertIsInstance(self.queue, ReleaseQueue)
        eager = build_swa_pool(page_size=4, lazy=False)
        self.assertIsNone(eager.release_queue)

    def test_held_row_is_not_handed_out_before_its_forward_is_processed(self):
        allocator, owner, queue = self.allocator, self.owner, self.queue
        row = allocator.alloc(16)
        held_physical = set(_physical_pages(owner, row).tolist())
        free_before = allocator.available_size()
        req = _InFlight()

        with queue.holding(req):
            allocator.free_segment(row, start_pos=0)

        self.assertEqual(allocator.available_size(), free_before)
        self.assertEqual(queue.held_tokens(owner), 16)
        self.assertEqual(len(queue), 1)
        # Still bound: the forward's page table names these physical pages.
        self.assertEqual(set(_physical_pages(owner, row).tolist()), held_physical)

        # Fill the pool, compacting under pressure: no take reaches a held page.
        taken = allocator.alloc(free_before)
        self.assertIsNotNone(taken)
        self.assertFalse(held_physical & set(_physical_pages(owner, taken).tolist()))
        owner.flush_for_allocation()
        self.assertEqual(int((owner.ids.virtual_to_physical[row[::4] // 4] < 0).sum()), 0)

        # A step processed whose forward still includes the request changes nothing.
        queue.release_retired()
        self.assertEqual(len(queue), 1)

        req.retire()
        queue.release_retired()
        self.assertEqual(len(queue), 0)
        self.assertEqual(queue.held_tokens(owner), 0)
        self.assertEqual(allocator.available_size(), 16)
        self.assertTrue(bool((owner.ids.virtual_to_physical[row[::4] // 4] < 0).all()))

    def test_frees_skip_an_open_free_group(self):
        """A result's processing groups its frees; a held free must not be
        applied by the group's end before the forward is processed."""
        allocator, owner, queue = self.allocator, self.owner, self.queue
        row = allocator.alloc(8)
        free_before = allocator.available_size()
        req = _InFlight()

        allocator.free_group_begin()
        with queue.holding(req):
            allocator.free_segment(row, start_pos=0)
        allocator.free_group_end()

        self.assertEqual(allocator.available_size(), free_before)
        req.retire()
        queue.release_retired()
        self.assertEqual(allocator.available_size(), free_before + 8)

    def test_releases_run_in_queue_order_once_their_forwards_are_processed(self):
        allocator, owner, queue = self.allocator, self.owner, self.queue
        first, second = allocator.alloc(8), allocator.alloc(8)
        early, late = _InFlight(), _InFlight()
        order = []
        with queue.holding(late):
            allocator.free_segment(second, start_pos=0)
            queue.defer(lambda: order.append("late"))
        with queue.holding(early):
            allocator.free_segment(first, start_pos=0)
            queue.defer(lambda: order.append("early"))

        early.retire()
        queue.release_retired()
        self.assertEqual(order, ["early"])
        self.assertEqual(queue.held_tokens(owner), 8)
        late.retire()
        queue.release_retired()
        self.assertEqual(order, ["early", "late"])


def _finished_req(cache, tokens, *, committed, allocated):
    """A request whose row holds ``allocated`` token slots (the last ones the
    forward in flight writes), finished, still in one unprocessed forward."""
    pool = cache.req_to_token_pool
    req = Req(
        rid="finished",
        origin_input_text="",
        origin_input_ids=array("q", tokens[: committed - 1]),
        sampling_params=SamplingParams(temperature=0, max_new_tokens=1),
    )
    acquire_rows(pool, pool.state, [req])
    req.output_ids = array("q", tokens[committed - 1 : committed])
    req.full_untruncated_fill_ids = array("q", tokens[:committed])
    req.extend_end = committed
    slots = cache.token_to_kv_pool_allocator.alloc(allocated)
    pool.write((req.kv.req_pool_idx, slice(0, allocated)), slots)
    req.kv.kv_committed_len = committed
    req.kv.kv_allocated_len = allocated
    req.last_node = cache.root_node_handle()
    req.kv.cache_protected_len = 0
    req.lock = None
    req.extra_key = None
    req.finished_reason = FINISH_LENGTH(length=1)
    req.inflight_steps = 1
    return req, slots


class TestReleaseAfterTheForward(CustomTestCase):
    """`release_kv_cache` inserts at once and gives the rest back later."""

    def setUp(self):
        super().setUp()
        setup_allocator_context()
        self.allocator, self.cache = build_swa_cache(occupancy=4, lazy=True)
        self.queue = self.allocator.release_queue

    def _match(self, tokens):
        key = RadixKey(array("q", tokens))
        return self.cache.match_prefix(MatchPrefixParams(key=key)).device_prefix_len

    def test_insert_now_free_and_unlock_after_the_forward(self):
        tokens = list(range(500, 512))
        # The tree already holds the first four tokens: the insert frees the
        # request's own copies, which the running forward still reads.
        shared = self.allocator.alloc(4)
        self.cache.insert(
            InsertParams(key=RadixKey(array("q", tokens[:4])), value=shared)
        )
        req, _ = _finished_req(self.cache, tokens, committed=12, allocated=12)
        free_before = self.allocator.available_size()
        evictable_before = self.cache.evictable_size()

        release_kv_cache(req, self.cache, checkpoint=True)

        self.assertEqual(self._match(tokens), 12)  # inserted at once
        self.assertFalse(req.kv.holds_kv)
        self.assertEqual(self.allocator.available_size(), free_before)
        # Still locked, the shared prefix included.
        self.assertEqual(self.cache.evictable_size(), evictable_before - 4)
        self.assertEqual(len(self.queue), 1)
        self.assertEqual(self.queue.held_tokens(self.allocator.full_attn_allocator), 4)

        req.inflight_steps = 0
        self.queue.release_retired()
        self.assertEqual(self.allocator.available_size(), free_before + 4)
        self.assertEqual(self.cache.evictable_size(), evictable_before + 8)
        self.cache.sanity_check()

    def test_release_with_no_forward_in_flight_is_immediate(self):
        tokens = list(range(700, 708))
        req, _ = _finished_req(self.cache, tokens, committed=8, allocated=8)
        req.inflight_steps = 0
        evictable_before = self.cache.evictable_size()

        release_kv_cache(req, self.cache, checkpoint=True)

        self.assertEqual(len(self.queue), 0)
        self.assertEqual(self.cache.evictable_size(), evictable_before + 8)

    def test_checkpoint_of_a_running_request_holds_its_duplicate_pages(self):
        tokens = list(range(900, 908))
        shared = self.allocator.alloc(8)
        self.cache.insert(InsertParams(key=RadixKey(array("q", tokens)), value=shared))
        req, _ = _finished_req(self.cache, tokens, committed=8, allocated=8)
        req.finished_reason = None
        free_before = self.allocator.available_size()

        checkpoint_kv_cache(req, self.cache)

        self.assertEqual(self.allocator.available_size(), free_before)
        req.inflight_steps = 0
        self.queue.release_retired()
        self.assertEqual(self.allocator.available_size(), free_before + 8)


class TestAdmissionCountsHeldPagesAsSoonFree(CustomTestCase):
    def setUp(self):
        super().setUp()
        setup_allocator_context()
        self.allocator, self.cache = build_swa_cache(occupancy=4, lazy=True)
        self.queue = self.allocator.release_queue

    @staticmethod
    def _largest(fits, upper=1024):
        """The largest token count ``fits`` accepts (monotone)."""
        low, high = 0, upper
        while low < high:
            mid = (low + high + 1) // 2
            if fits(mid):
                low = mid
            else:
                high = mid - 1
        return low

    def _admitted(self):
        budget = self.allocator.create_prefill_budget(self.cache)
        return self._largest(
            lambda total: budget.check_prefill(
                extend_input_len=8,
                total_tokens=total,
                max_new_tokens=8,
                input_tokens=8,
                swa_host_hit_length=0,
                chunk_limit=None,
            )[0]
        )

    def _allocatable_now(self):
        budget = self.allocator.create_prefill_budget(self.cache)
        return self._largest(
            lambda tokens: budget.can_allocate_prefill(
                paged_input=tokens,
                extend_input_len=tokens,
                max_new_tokens=0,
                chunk_limit=None,
            )
        )

    def test_admission_counts_held_tokens_and_the_current_budget_does_not(self):
        row = self.allocator.alloc(16)
        admitted, allocatable = self._admitted(), self._allocatable_now()
        req = _InFlight()
        with self.queue.holding(req):
            self.allocator.free_segment(row, start_pos=0)

        # A request's total demand may count on the held pages...
        self.assertGreaterEqual(self._admitted(), admitted + 8)
        # ...but nothing allocated now may use them.
        self.assertEqual(self._allocatable_now(), allocatable)

        req.retire()
        self.queue.release_retired()
        self.assertGreater(self._allocatable_now(), allocatable)


class TestDecodeWaitsForHeldReleases(CustomTestCase):
    """A decode short of memory waits one step for KV a forward in flight
    gives back, instead of retracting requests whose pages would only be held."""

    def _scheduler(self, *, queue):
        s = Scheduler.__new__(Scheduler)
        s.enable_skip_finishing_decode = False
        s.uocr_runtime = None
        s.token_to_kv_pool_allocator = SimpleNamespace(release_queue=queue)
        s.decode_offload_manager = None
        return s

    def _batch(self, reqs, *, fits):
        return SimpleNamespace(
            reqs=reqs,
            batch_size=lambda: len(reqs),
            filter_batch=lambda **_: None,
            is_empty=lambda: not reqs,
            check_decode_mem=lambda: fits,
            batch_is_full=True,
        )

    def test_short_decode_waits_while_its_requests_are_in_flight(self):
        queue = ReleaseQueue(SimpleNamespace(), ())
        s = self._scheduler(queue=queue)
        batch = self._batch([SimpleNamespace(inflight_steps=1)], fits=False)
        self.assertIsNone(Scheduler.update_running_batch(s, batch))
        self.assertFalse(batch.batch_is_full)

    def test_short_decode_waits_while_a_release_is_queued(self):
        allocator = build_swa_pool(page_size=4, lazy=True)
        queue = allocator.release_queue
        with queue.holding(_InFlight()):
            allocator.free_segment(allocator.alloc(4), start_pos=0)
        s = self._scheduler(queue=queue)
        batch = self._batch([SimpleNamespace(inflight_steps=0)], fits=False)
        self.assertIsNone(Scheduler.update_running_batch(s, batch))

    def test_short_decode_retracts_when_nothing_comes_back(self):
        queue = ReleaseQueue(SimpleNamespace(), ())
        s = self._scheduler(queue=queue)
        batch = self._batch([SimpleNamespace(inflight_steps=0)], fits=False)

        class _Retracted(Exception):
            pass

        s.token_to_kv_pool_allocator.available_size = mock.Mock(
            side_effect=_Retracted
        )
        with self.assertRaises(_Retracted):
            Scheduler.update_running_batch(s, batch)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required (zeroing kernel)")
class TestHeldPageBytesSurviveHandOut(CustomTestCase):
    """Zeroing on hand-out never reaches a page a forward in flight reads."""

    def test_held_page_is_not_zeroed_by_a_take(self):
        page_size = 4
        spec = MHASubPoolSpec(
            name="full",
            layer_num=1,
            grow_direction="up",
            head_num=1,
            head_dim=8,
            store_dtype=torch.bfloat16,
        )
        state_spec = MambaSubPoolSpec(
            name="mamba",
            layer_num=1,
            grow_direction="down",
            conv_state_shapes=((8, 3),),
            conv_dtype=torch.bfloat16,
            temporal_state_shape=(2, 4, 4),
            temporal_dtype=torch.float32,
        )
        arena = arena_of(
            total_bytes=64 * page_size * spec.entry_bytes(),
            sub_pool_specs=[spec, state_spec],
            device="cuda",
            enable_memory_saver=False,
            page_size=page_size,
        )
        view = MHAView(
            arena=arena, spec=arena.spec("full"), page_size=page_size, model_layer_ids=(0,)
        )
        band = Band(
            arena=arena,
            sub_pool_name="full",
            device="cuda",
            is_id_owner=True,
            page_size=page_size,
            lazy_compaction=True,
        )
        queue = ReleaseQueue(band, (band,))
        pages = arena.raw[: view.num_pages * view.page_bytes].view(
            view.num_pages, view.page_bytes
        )
        row = band.alloc(4 * page_size)
        held = _physical_pages(band, row)
        pages[held] = 7
        req = _InFlight()
        with queue.holding(req):
            band.free(row, _pages=row[::page_size] // page_size)

        taken = band.alloc(band.available_size())
        torch.cuda.synchronize()
        self.assertFalse(set(held.tolist()) & set(_physical_pages(band, taken).tolist()))
        self.assertTrue(bool((pages[held] == 7).all()))

        req.retire()
        queue.release_retired()
        band.free(taken)
        again = band.alloc(band.available_size())
        torch.cuda.synchronize()
        reused = _physical_pages(band, again)
        self.assertTrue(set(held.tolist()) <= set(reused.tolist()))
        self.assertTrue(bool((pages[held] == 0).all()))


if __name__ == "__main__":
    unittest.main()
