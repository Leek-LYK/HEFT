import json
import os
import tempfile
import unittest
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from tensordict import TensorDict

from active_adaptation.learning.modules.valuenorm import ValueNorm1
from active_adaptation.learning.ppo.ppo import synchronize_vecnorm
from scripts.utils.helpers import MaskedVecNorm


def _ddp_worker(rank: int, world_size: int, rendezvous: str, output: str) -> None:
    os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")
    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=world_size,
    )
    try:
        torch.manual_seed(7)
        model = torch.nn.parallel.DistributedDataParallel(
            torch.nn.Linear(3, 1, bias=False)
        )
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        optimizer.zero_grad()
        model(torch.full((4, 3), float(rank + 1))).sum().backward()
        gradient = model.module.weight.grad.detach().clone()
        optimizer.step()

        vecnorm = MaskedVecNorm(["obs"], decay=1.0)
        vecnorm(
            TensorDict(
                {
                    "obs": torch.tensor(
                        [[1.0 + 2.0 * rank], [float("nan")]], dtype=torch.float32
                    ),
                    "_vecnorm_valid": torch.tensor([[True], [False]]),
                },
                batch_size=[2],
            )
        )
        synchronize_vecnorm(vecnorm)

        value_norm = ValueNorm1(1, beta=0.0)
        value_norm.update(torch.tensor([[1.0 + 2.0 * rank]]))

        payload = {
            "gradient": gradient.tolist(),
            "weight": model.module.weight.detach().tolist(),
            "vecnorm_loc": float(vecnorm.loc["obs"].item()),
            "vecnorm_scale": float(vecnorm.scale["obs"].item()),
            "value_mean": float(value_norm.running_mean_var()[0].item()),
        }
        Path(output, f"rank_{rank}.json").write_text(json.dumps(payload))
    finally:
        dist.destroy_process_group()


class DDPTrainingSyncTest(unittest.TestCase):
    def test_gradients_vecnorm_and_valuenorm_are_globally_synchronized(self):
        with tempfile.TemporaryDirectory() as directory:
            rendezvous = str(Path(directory, "rendezvous"))
            mp.spawn(
                _ddp_worker,
                args=(2, rendezvous, directory),
                nprocs=2,
                join=True,
            )
            rows = [
                json.loads(Path(directory, f"rank_{rank}.json").read_text())
                for rank in range(2)
            ]
        self.assertEqual(rows[0], rows[1])
        self.assertEqual(rows[0]["gradient"], [[6.0, 6.0, 6.0]])
        self.assertAlmostEqual(rows[0]["vecnorm_loc"], 2.0, places=6)
        self.assertAlmostEqual(rows[0]["value_mean"], 2.0, places=6)


if __name__ == "__main__":
    unittest.main()
