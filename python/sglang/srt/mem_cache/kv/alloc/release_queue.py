"""Releases of unified KV pages that a forward still in flight may touch.

Under the overlap scheduler, the forward launched before a result is processed may
still read the pages of a request that result finishes (its prepared page table names
them) and write the request's next slot. A static pool hands a freed page out again
only to a later forward, which the forward stream orders after the running one. A
unified pool also writes a page from the schedule stream when it hands it out (zeroing)
or compacts into it, and the schedule stream is not ordered after the running forward.

So a request that ends (finishes, aborts, is retracted, or checkpoints into the prefix
tree) while a forward that includes it is unprocessed does not give its pages back at
once. Its band frees, and its prefix-tree unlock, are queued in their original order
with the request, and run once every forward that included it has been processed: the
host has synchronized that forward's completion while processing its result, so the
pages are quiet when they reach the band. The point is the same on every rank, so the
pools stay identical across ranks; no stream waits and no host syncs are added.

Until then the pages stay bound: no take, zeroing or compaction destination reaches
them. Admission counts them as soon-free (`held_tokens`), never as free now.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, Callable, Dict, Iterator, List, Optional

import torch

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.mem_cache.kv.alloc.band import Band


class _Release:
    """One request's queued release: its steps run in the order they were made."""

    __slots__ = ("req", "steps", "pages")

    def __init__(self, req: Req):
        self.req = req
        self.steps: List[Callable[[], None]] = []
        # Pages each band gets back, where the free names them exactly.
        self.pages: Dict[str, int] = {}

    def run(self) -> None:
        for step in self.steps:
            step()


class ReleaseQueue:
    """The releases a unified chain holds until the forwards that include their
    requests are processed. One per chain; every band of the chain consults it."""

    def __init__(self, allocator, bands) -> None:
        self._allocator = allocator
        self._bands = tuple(bands)
        self._queued: List[_Release] = []
        self._open: Optional[_Release] = None
        for band in self._bands:
            band.release_queue = self

    @staticmethod
    def in_flight(req: Optional[Req]) -> bool:
        """Whether a forward that includes ``req`` is launched and unprocessed."""
        return req is not None and getattr(req, "inflight_steps", 0) > 0

    @contextmanager
    def holding(self, req: Req) -> Iterator[_Release]:
        """Queue every band free made inside the block with ``req``'s release.
        Holds nest: an inner hold for the same release joins the open one."""
        if self._open is not None:
            yield self._open
            return
        release = _Release(req)
        self._open = release
        # A result's processing groups its frees (`free_group_begin`); a group
        # applies them at its end, after this hold closed. Frees made in the hold
        # skip the group so they reach the hold.
        groups = [
            (owner, owner.free_group, owner.free_page_reps_group)
            for owner in (self._allocator, *self._bands)
        ]
        for owner, _, _ in groups:
            owner.free_group = None
            owner.free_page_reps_group = None
        try:
            yield release
        finally:
            for owner, free_group, free_page_reps_group in groups:
                owner.free_group = free_group
                owner.free_page_reps_group = free_page_reps_group
            self._open = None
            if release.steps:
                self._queued.append(release)

    def hold_free(
        self, band: Band, free_index: torch.Tensor, pages: Optional[torch.Tensor]
    ) -> bool:
        """Queue a band free if a hold is open; return whether it was queued.
        The ids are copied: a caller may pass a view of a row that is reused."""
        release = self._open
        if release is None:
            return False
        index = free_index.clone()
        page_ids = None if pages is None else pages.clone()
        release.steps.append(lambda: band.free(index, _pages=page_ids))
        if page_ids is not None:
            count = int(page_ids.numel())
        elif band.page_size == 1:
            count = int(index.numel())
        else:
            # Token ids whose distinct pages are unknown without a device read:
            # not counted, so admission stays conservative.
            count = 0
        release.pages[band.sub_pool_name] = (
            release.pages.get(band.sub_pool_name, 0) + count
        )
        return True

    def defer(self, step: Callable[[], None]) -> bool:
        """Queue ``step`` with the open hold's release, after the frees made
        before it; return False (and do nothing) when no hold is open."""
        if self._open is None:
            return False
        self._open.steps.append(step)
        return True

    def release_retired(self) -> None:
        """Run, in queue order, the releases no unprocessed forward includes."""
        if not self._queued:
            return
        queued, self._queued = self._queued, []
        for release in queued:
            if self.in_flight(release.req):
                self._queued.append(release)
            else:
                release.run()

    def held_tokens(self, band: Band) -> int:
        """Tokens of ``band`` that queued releases give back (soon-free)."""
        pages = sum(r.pages.get(band.sub_pool_name, 0) for r in self._queued)
        return pages * band.page_size

    def held_requests(self) -> List[Req]:
        return [r.req for r in self._queued]

    def __len__(self) -> int:
        return len(self._queued)

    def clear(self) -> None:
        """Drop every queued release (the pool is being reset)."""
        assert self._open is None, "pool reset inside a release hold"
        self._queued = []
