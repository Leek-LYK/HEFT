import unittest
from types import SimpleNamespace
import torch
from active_adaptation.utils.multimotion import ProgressiveMultiMotionDataset, _build_motion_sample_probs


def sampler(enabled):
    ds = object.__new__(ProgressiveMultiMotionDataset)
    ds.device = torch.device('cpu')
    ds.resample_on_reset = enabled
    ds.motion_sample_probs = torch.full((100,), .01)
    ds.motion_to_dataset_id = torch.zeros(100, dtype=torch.long)
    ds.global_lengths = torch.arange(100) + 150
    ds._motion_ids_A = torch.zeros(4, dtype=torch.long)
    ds._dataset_ids_A = torch.zeros(4, dtype=torch.long)
    ds._len_A = torch.full((4,), 150, dtype=torch.int32)
    ds._cache_valid = torch.ones(4, dtype=torch.bool)
    ds._cache_motion_id = torch.zeros(4, dtype=torch.long)
    ds._cache_start = torch.zeros(4, dtype=torch.long)
    ds._cache_end = torch.full((4,), 100, dtype=torch.long)
    ds._student_cache_has_override = torch.zeros(4, dtype=torch.bool)
    return ds


class FullSamplingTest(unittest.TestCase):
    def test_reset_explores_corpus_and_preserves_rewind_rows(self):
        torch.manual_seed(20260904)
        ds = sampler(True)
        seen = set()
        for _ in range(300):
            lengths = ds.reset(torch.arange(4), resample_mask=torch.tensor([True, False, True, False]))
            seen.update(ds._motion_ids_A.tolist())
            torch.testing.assert_close(lengths.long(), ds.global_lengths[ds._motion_ids_A])
        self.assertGreater(len(seen), 95)
        self.assertEqual(ds._motion_ids_A[[1, 3]].tolist(), [0, 0])
        self.assertEqual(ds._cache_valid.tolist(), [False, True, False, True])

    def test_legacy_default_does_not_resample(self):
        ds = sampler(False)
        for _ in range(20):
            ds.reset(torch.arange(4))
        self.assertEqual(ds._motion_ids_A.tolist(), [0, 0, 0, 0])
        self.assertTrue(ds._cache_valid.all())

    def test_shard_count_weights_give_uniform_global_motion_probability(self):
        teacher = SimpleNamespace(motion_to_dataset_id=torch.tensor([0] + [1] * 9), lengths=torch.ones(10))
        groups = [{'weight': n, 'sampling_mode': 'motion_uniform'} for n in (1, 9)]
        probs = _build_motion_sample_probs(groups=groups, teacher_dataset=teacher, device=torch.device('cpu'))
        torch.testing.assert_close(probs, torch.full((10,), .1))


if __name__ == '__main__':
    unittest.main()
