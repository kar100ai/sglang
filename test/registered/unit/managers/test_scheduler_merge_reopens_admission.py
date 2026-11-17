"""Merging a prefill batch into a non-empty running batch reopens admission."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.disaggregation.mode import DisaggregationMode
from sglang.srt.managers.schedule_batch import NextBatchPlan
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.scheduler_components.request_list import (
    IndexedReqList,
    KeyedReqList,
    ListKind,
)

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _Req:
    def __init__(self, rid):
        self.rid = rid
        self.owner_list = None
        self.finished = False


def _reqs(prefix, n):
    return [_Req(f"{prefix}{i}") for i in range(n)]


class _Batch:
    def __init__(self, reqs, *, extend, batch_is_full=False):
        self.reqs = reqs
        self.forward_mode = SimpleNamespace(is_extend=lambda: extend)
        self.chunked_req = None
        self.is_prefill_only = False
        self.batch_is_full = batch_is_full

    def batch_size(self):
        return len(self.reqs)

    def is_empty(self):
        return not self.reqs

    def filter_batch(self, reqs_to_exclude=()):
        self.reqs = [
            r for r in self.reqs if not r.finished and r not in reqs_to_exclude
        ]

    def merge_batch(self, other):
        self.reqs = self.reqs + other.reqs


class TestMergeReopensAdmission(unittest.TestCase):
    def _scheduler(self, seen):
        s = Scheduler.__new__(Scheduler)
        s.scheduler_stage_metrics = None
        s.disaggregation_mode = DisaggregationMode.NULL
        s.process_pending_chunked_abort = Mock()
        s._process_hicache_events = Mock()
        s.enable_fpm = False
        s.dllm_config = None
        s.chunked_reqs = IndexedReqList(ListKind.CHUNKED)
        s.running_reqs = KeyedReqList(ListKind.DECODING)
        s.enable_hisparse = False
        s.invariant_checker = Mock()
        s.require_mlp_sync = False
        s._should_defer_prefill = Mock(return_value=False)
        s.dp_attn_adapter = Mock()
        s.dp_attn_adapter.maybe_prepare_mlp_sync_batch.side_effect = lambda b, **_: b
        s.dp_attn_adapter.maybe_convert_decode_to_extend.side_effect = lambda b: b
        s.update_running_batch = Mock(side_effect=lambda b: b)
        s._arm_prefill_decode_interval = Mock()
        s.ngram_embedding_manager = Mock()
        s.ngram_embedding_manager.prepare_for_forward.side_effect = lambda b, **_: b

        def get_new_batch_prefill(running_batch):
            seen.append((running_batch.batch_size(), running_batch.batch_is_full))
            return NextBatchPlan(batch_to_run=None, running_batch=running_batch)

        s.get_new_batch_prefill = get_new_batch_prefill
        return s

    def test_merge_into_running_batch_clears_full_flag(self):
        # One request ran; the next pass admitted 28 and refused one for KV
        # budget, which set the flag on the running batch.
        seen = []
        s = self._scheduler(seen)
        running = _Batch(_reqs("r", 1), extend=False, batch_is_full=True)
        prefill = _Batch(_reqs("p", 28), extend=True)

        s.get_next_batch_to_run(running_batch=running, last_batch=prefill)

        # The 28 now count as running, so admission is tried again this step.
        self.assertEqual(seen[-1], (29, False))

    def test_merge_into_empty_running_batch_clears_full_flag(self):
        # Unchanged: the prefill batch replaces an empty running batch.
        seen = []
        s = self._scheduler(seen)
        running = _Batch([], extend=False, batch_is_full=True)
        prefill = _Batch(_reqs("p", 29), extend=True)

        s.get_next_batch_to_run(running_batch=running, last_batch=prefill)

        self.assertEqual(seen[-1], (29, False))

    def test_decode_step_keeps_full_flag(self):
        # Unchanged: with nothing merged and nothing finished, the flag stays.
        seen = []
        s = self._scheduler(seen)
        running = _Batch(_reqs("r", 29), extend=False, batch_is_full=True)

        s.get_next_batch_to_run(running_batch=running, last_batch=running)

        self.assertEqual(seen[-1], (29, True))


if __name__ == "__main__":
    unittest.main()
