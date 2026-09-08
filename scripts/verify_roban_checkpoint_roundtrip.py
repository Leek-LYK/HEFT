"""Reload a Roban checkpoint twice and verify deterministic action equality."""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torchrl.envs.utils import ExplorationType, set_exploration_type

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _load_once(cfg, checkpoint: Path):
    from scripts.utils.helpers import make_env_policy

    local_cfg = OmegaConf.create(copy.deepcopy(OmegaConf.to_container(cfg, resolve=False)))
    local_cfg.checkpoint_path = str(checkpoint)
    env, policy, vecnorm, _ = make_env_policy(local_cfg)
    carry = env.reset()
    fixed = carry.select("policy", "priv", "priv_critic", "is_init", strict=False)[
        :4
    ].clone().cpu()
    td = fixed.clone().to(policy.device)
    with torch.inference_mode(), set_exploration_type(ExplorationType.MODE):
        policy.get_rollout_policy("checkpoint_roundtrip")(td)
    action = td["action"].detach().cpu()
    if not torch.isfinite(action).all():
        raise FloatingPointError("deterministic action is not finite")
    dataset_device = str(env.base_env.command_manager.dataset.teacher_dataset.ds_device)
    env.close(raise_if_closed=False)
    del env, policy, vecnorm, carry, td
    gc.collect()
    torch.cuda.empty_cache()
    return fixed, action, dataset_device


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=8)
    args = parser.parse_args()
    checkpoint = args.checkpoint.resolve()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = OmegaConf.create(
        copy.deepcopy(OmegaConf.to_container(payload["cfg"], resolve=False))
    )
    OmegaConf.set_struct(cfg, False)
    cfg.task.num_envs = int(args.num_envs)
    cfg.headless = True
    cfg.app.headless = True
    cfg.app.enable_cameras = False
    cfg.eval_render = False
    if cfg.task.get("nonfinite_diagnostics", None) is not None:
        cfg.task.nonfinite_diagnostics.enabled = False

    fixed, expected, first_device = _load_once(cfg, checkpoint)
    from scripts.utils.helpers import make_env_policy

    second_cfg = OmegaConf.create(
        copy.deepcopy(OmegaConf.to_container(cfg, resolve=False))
    )
    second_cfg.checkpoint_path = str(checkpoint)
    second_env, second_policy, _, _ = make_env_policy(second_cfg)
    td = fixed.clone().to(second_policy.device)
    with torch.inference_mode(), set_exploration_type(ExplorationType.MODE):
        second_policy.get_rollout_policy("checkpoint_roundtrip")(td)
    actual = td["action"].detach().cpu()
    torch.testing.assert_close(actual, expected, rtol=1.0e-5, atol=1.0e-6)
    delta = float((actual - expected).abs().max().item())
    second_device = str(
        second_env.base_env.command_manager.dataset.teacher_dataset.ds_device
    )
    second_env.close(raise_if_closed=False)

    digest = hashlib.sha256(expected.numpy().tobytes()).hexdigest()
    report = {
        "status": "PASS",
        "checkpoint_path": str(checkpoint),
        "checkpoint_size_bytes": checkpoint.stat().st_size,
        "saved_phase": payload["policy"].get("last_phase"),
        "saved_num_updates": payload["policy"].get("_meta", {}).get("num_updates"),
        "reload_count": 2,
        "num_envs": int(args.num_envs),
        "dataset_devices": [first_device, second_device],
        "fixed_input_shape": list(fixed["policy"].shape),
        "action_shape": list(expected.shape),
        "action_sha256": digest,
        "action_rtol": 1.0e-5,
        "action_atol": 1.0e-6,
        "action_max_abs_delta": delta,
        "action_matches": True,
        "all_finite": bool(torch.isfinite(actual).all().item()),
    }
    args.output.resolve().write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
