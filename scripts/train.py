import torch
# import warp
import hydra
import numpy as np

import einops
import logging
import os
import sys
import time
import datetime
import resource

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from omegaconf import OmegaConf, DictConfig
from collections import OrderedDict
from tqdm import tqdm
from setproctitle import setproctitle
import torch.distributed as dist

import active_adaptation as aa
from active_adaptation.utils.torchrl import SyncDataCollector, TDTimeBuffer

# local import
from scripts.utils.helpers import make_env_policy, EpisodeStats, evaluate
from scripts.utils.train_record import TrainStateRecorder
from torchrl.envs.utils import set_exploration_type, ExplorationType
from active_adaptation.utils.wandb import finish_wandb_run, init_wandb_run

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False
torch.set_float32_matmul_precision('high')

import os

FILE_PATH = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(FILE_PATH, "..", "cfg")


def _tensorboard_log_dir(cfg: DictConfig, diagnostics_enabled: bool) -> str:
    tensorboard_cfg = cfg.get("tensorboard", {})
    configured = tensorboard_cfg.get("log_dir", None)
    if configured:
        return os.path.abspath(str(configured))
    if diagnostics_enabled:
        audit_kind = "pilot" if bool(cfg.get("pilot", {}).get("enabled", False)) else "smoke"
        audit_output = cfg.get(audit_kind, {}).get("output_dir", None)
        if audit_output:
            return os.path.join(os.path.abspath(str(audit_output)), "tensorboard")
    return os.path.join(os.getcwd(), "tensorboard")


def _write_tensorboard_scalars(writer, metrics: dict, step: int) -> None:
    for name, value in sorted(metrics.items()):
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                continue
            value = value.detach().item()
        if isinstance(value, (int, float, np.integer, np.floating)):
            writer.add_scalar(str(name), float(value), global_step=int(step))
    writer.flush()

@hydra.main(config_path=CONFIG_PATH, config_name="train", version_base=None)
def main(cfg: DictConfig):
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)
    smoke_enabled = bool(cfg.get("smoke", {}).get("enabled", False))
    pilot_enabled = bool(cfg.get("pilot", {}).get("enabled", False))
    if smoke_enabled and pilot_enabled:
        raise ValueError("smoke and pilot diagnostics cannot be enabled together")
    diagnostics_enabled = smoke_enabled or pilot_enabled
    pilot_cfg = cfg.get("pilot", {})
    nonfinite_cfg = pilot_cfg.get("nonfinite_diagnostics", {})
    if bool(nonfinite_cfg.get("enabled", False)):
        if not pilot_enabled:
            raise ValueError("event-level nonfinite diagnostics are pilot-only")
        if not bool(cfg.task.get("recover_nonfinite_state", False)):
            raise ValueError(
                "event-level nonfinite diagnostics require recovery to be enabled"
            )
        if not pilot_cfg.get("output_dir", None):
            raise ValueError("pilot.output_dir is required")
        cfg.task.nonfinite_diagnostics = OmegaConf.create(
            OmegaConf.to_container(nonfinite_cfg, resolve=True)
        )
        diagnostic_output = str(pilot_cfg.output_dir)
        if aa.is_distributed() and not aa.is_main_process():
            diagnostic_output = os.path.join(
                diagnostic_output, f"rank_{aa.get_local_rank()}"
            )
        cfg.task.nonfinite_diagnostics.output_dir = diagnostic_output
    if diagnostics_enabled and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    
    print(f"is_distributed: {aa.is_distributed()}, local_rank: {aa.get_local_rank()}/{aa.get_world_size()}")
    if aa.is_distributed():
        torch.cuda.set_device(aa.get_local_rank())
        dist.init_process_group(
            backend="nccl",
        )
        cfg.seed = cfg.seed + aa.get_local_rank() * 10000

    from scripts.refinetrack_v01.runtime import check_config
    check_config(cfg)
    env, policy, vecnorm, primer = make_env_policy(cfg)
    initial_schedule_progress = float(getattr(policy, "progress", 0.0))

    frames_per_batch = env.num_envs * cfg.algo.train_every
    total_frames = cfg.get("total_frames", -1) // aa.get_world_size()
    total_frames = total_frames // frames_per_batch * frames_per_batch
    total_iters = total_frames // frames_per_batch
    save_interval = cfg.get("save_interval", -1)
    start_iter = cfg.get("start_iter", 0)
    diagnostics = None
    if diagnostics_enabled:
        from active_adaptation.utils.ppo_smoke import PPOSmokeDiagnostics
        diagnostics = PPOSmokeDiagnostics(
            cfg,
            env,
            policy,
            vecnorm,
            total_iters=total_iters,
            audit_kind="pilot" if pilot_enabled else "smoke",
        )
    tensorboard_enabled = bool(cfg.get("tensorboard", {}).get("enabled", False))
    tensorboard_writer = None
    if tensorboard_enabled and aa.is_main_process():
        from torch.utils.tensorboard import SummaryWriter

        tensorboard_dir = _tensorboard_log_dir(cfg, diagnostics_enabled)
        tensorboard_writer = SummaryWriter(
            log_dir=tensorboard_dir,
            flush_secs=int(cfg.tensorboard.get("flush_secs", 10)),
        )
        aa.print(f"TensorBoard logging enabled: {tensorboard_dir}")
    need_logging = aa.is_main_process()
    run = None
    if need_logging:
        default_run_name = f"{cfg.exp_name}-{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')}"
        output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
        if cfg.wandb.get("mode", "disabled") != "disabled":
            run = init_wandb_run(cfg.wandb, config=cfg, name=default_run_name)
            run.name = f"{run.id}-{default_run_name}" if run.id else default_run_name
            output_dir = run.dir
        setproctitle(run.name if run is not None else default_run_name)
        os.makedirs(output_dir, exist_ok=True)
        cfg_save_path = os.path.join(output_dir, "cfg.yaml")
        OmegaConf.save(cfg, cfg_save_path)

        import inspect
        import shutil
        source_path = inspect.getfile(policy.__class__)
        target_path = os.path.join(output_dir, os.path.basename(source_path))
        shutil.copy(source_path, target_path)
        if run is not None:
            run.save(cfg_save_path, policy="now")
            run.save(target_path, policy="now")

        log_interval = (env.max_episode_length // cfg.algo.train_every) + 1
        logging.info(f"Log interval: {log_interval} steps")

        stats_keys = [
            k for k in env.reward_spec.keys(True, True) 
            if isinstance(k, tuple) and k[0] == "stats"
        ]
        episode_stats = EpisodeStats(stats_keys, device=env.device)

        def save(policy, checkpoint_name: str):
            ckpt_path = os.path.join(output_dir, f"{checkpoint_name}.pt")
            state_dict = OrderedDict()
            if run is not None:
                state_dict["wandb"] = {"name": run.name, "id": run.id}
            state_dict["policy"] = policy.state_dict()
            state_dict["env"] = env.state_dict()
            state_dict["cfg"] = cfg
            state_dict["vecnorm"] = vecnorm.state_dict()
            torch.save(state_dict, ckpt_path + ".tmp")
            os.replace(ckpt_path + ".tmp", ckpt_path)
            if run is not None:
                run.save(ckpt_path, policy="now", base_path=run.dir)
            logging.info(f"Saved checkpoint to {str(ckpt_path)}")

        def should_save(i):
            if not aa.is_main_process():
                return False
            return i > 0 and i % save_interval == 0
    
    rollout_policy = policy.get_rollout_policy("train")
    preserve_pilot_progress = bool(
        pilot_enabled
        and cfg.checkpoint_path is not None
        and pilot_cfg.get("preserve_checkpoint_schedule_progress", True)
    )
    if preserve_pilot_progress:
        if hasattr(policy, "step_schedule"):
            policy.step_schedule(initial_schedule_progress, -1)
        if hasattr(env, "step_schedule"):
            env.step_schedule(initial_schedule_progress, -1)
    env_frames = 0
    carry = env.reset()
    if diagnostics is not None:
        diagnostics.on_reset(carry)

    assert env.training
    stop_after = cfg.get("stop_after_rollouts", None)
    if stop_after is not None and (int(stop_after) <= 0 or diagnostics_enabled):
        raise ValueError("stop_after_rollouts must be positive and is for ordinary training only")
    end_iter = total_iters if stop_after is None else min(total_iters, start_iter + int(stop_after))
    iter_range = range(start_iter, end_iter)
    if aa.is_main_process():
        progress = tqdm(iter_range)
    else:
        progress = iter_range

    N = env.num_envs
    T = cfg.algo.train_every
    data_buf = TDTimeBuffer(N, T, device=policy.device)

    train_recorder = None
    if aa.is_main_process():
        train_record_cfg = cfg.get("train_record", {})
        record_output_dir = train_record_cfg.get("output_dir", "train_records")
        if not os.path.isabs(record_output_dir):
            record_output_dir = os.path.join(os.getcwd(), record_output_dir)
        train_recorder = TrainStateRecorder(
            env,
            interval=int(train_record_cfg.get("interval", 0)),
            num_envs=int(train_record_cfg.get("envs", 16)),
            num_steps=int(train_record_cfg.get("steps", 256)),
            output_dir=record_output_dir,
            seed=int(cfg.seed),
            start_iter=int(start_iter),
            enabled=bool(train_record_cfg.get("enabled", True)),
        )

    for i in progress:
        try:
            start = time.perf_counter()
            if train_recorder is not None:
                train_recorder.maybe_start(i)

            with torch.inference_mode(), set_exploration_type(ExplorationType.RANDOM):
                torch.compiler.cudagraph_mark_step_begin() # for compiled policy
                for t in range(cfg.algo.train_every):
                    if diagnostics is not None:
                        diagnostics.before_transition(iteration=i, rollout_step=t)
                    carry = rollout_policy(carry)

                    td, carry = env.step_and_maybe_reset(carry)
                    if train_recorder is not None:
                        rollout_step = i * cfg.algo.train_every + t
                        train_recorder.on_step(i, rollout_step)

                    # deal with value
                    policy.critic(td)
                    policy.critic(td["next"])
                    td.get(("next", "state_value"))[:] = torch.where(
                        td["next", "done"],
                        td["state_value"],
                        td["next", "state_value"]
                    )

                    if diagnostics is not None:
                        diagnostics.observe_transition(td, iteration=i, rollout_step=t)

                    # clean up tensordict
                    td["next"] = td["next"].select("reward", "done", "terminated", "state_value", "stats", strict=False)
                    td = td.select(
                        "policy",
                        "priv",
                        "priv_critic",
                        "action",
                        "action_log_prob",
                        "loc",
                        "scale",
                        "is_init",
                        "state_value",
                        "next",
                        strict=False,
                    )

                    data_buf.write_step(t, td)

                data = data_buf.td

            if i == start_iter and cfg.get('refinetrack', {}).get('enabled', False):
                from scripts.refinetrack_v01.runtime_probe import probe
                probe(env, policy, carry, output_dir, cfg.refinetrack.condition)

            rollout_time = time.perf_counter() - start
            training_start = time.perf_counter()

            progress_ratio = i / total_iters
            if preserve_pilot_progress:
                progress_ratio = max(initial_schedule_progress, progress_ratio)
            if hasattr(policy, "step_schedule"):
                policy.step_schedule(progress_ratio, i)
            if hasattr(env, "step_schedule"):
                env.step_schedule(progress_ratio, i)

            if diagnostics is not None:
                diagnostics.before_update(data, iteration=i)
            train_carry = policy.train_op(data, vecnorm)
            training_time = time.perf_counter() - training_start
            if cfg.get('refinetrack', {}).get('enabled', False):
                from scripts.refinetrack_v01.runtime import log_iteration
                log_iteration(output_dir, i, rollout_time, training_time, data.numel())
            if diagnostics is not None:
                diagnostics.after_update(
                    data,
                    train_carry,
                    iteration=i,
                    rollout_time=rollout_time,
                    training_time=training_time,
                )
        except BaseException as error:
            if diagnostics is not None:
                diagnostics.abort(error)
            if tensorboard_writer is not None:
                tensorboard_writer.close()
            env.close(raise_if_closed=False)
            raise

        if tensorboard_enabled:
            recorder = getattr(env.base_env, "nonfinite_diagnostics", None)
            local_recovery_count = float(
                env.base_env.nonfinite_state_recovery_count
            )
            # Match PPO's scalar reward semantics by summing reward groups for
            # each environment transition before aggregating across ranks.
            rollout_reward = data["next", "reward"].detach().sum(dim=-1)
            reward_sum = rollout_reward.double().sum()
            reward_count = torch.tensor(
                float(rollout_reward.numel()),
                device=env.device,
                dtype=torch.float64,
            )
            reward_min = rollout_reward.min().double()
            reward_max = rollout_reward.max().double()
            terminal_count = data["next", "done"].double().sum()
            cumulative_recoveries = torch.tensor(
                local_recovery_count,
                device=env.device,
                dtype=torch.float64,
            )
            cumulative_events = torch.tensor(
                float(
                    local_recovery_count
                    if recorder is None
                    else recorder.total_events
                ),
                device=env.device,
                dtype=torch.float64,
            )
            if aa.is_distributed():
                dist.all_reduce(reward_sum, op=dist.ReduceOp.SUM)
                dist.all_reduce(reward_count, op=dist.ReduceOp.SUM)
                dist.all_reduce(reward_min, op=dist.ReduceOp.MIN)
                dist.all_reduce(reward_max, op=dist.ReduceOp.MAX)
                dist.all_reduce(terminal_count, op=dist.ReduceOp.SUM)
                dist.all_reduce(cumulative_recoveries, op=dist.ReduceOp.SUM)
                dist.all_reduce(cumulative_events, op=dist.ReduceOp.SUM)
            if tensorboard_writer is not None:
                rollout_offset = int(
                    cfg.get("pilot", {}).get("rollout_offset", 0)
                    if pilot_enabled
                    else 0
                )
                tensorboard_step = rollout_offset + i + 1
                tensorboard_metrics = {
                    **{f"train/{name}": value for name, value in train_carry.items()},
                    "throughput/rollout_transitions_per_s": (
                        data.numel() * aa.get_world_size() / rollout_time
                    ),
                    "throughput/end_to_end_transitions_per_s": (
                        data.numel()
                        * aa.get_world_size()
                        / (rollout_time + training_time)
                    ),
                    "timing/rollout_s": rollout_time,
                    "timing/training_s": training_time,
                    "rollout/reward_mean_global": reward_sum
                    / reward_count.clamp_min(1.0),
                    "rollout/reward_min_global": reward_min,
                    "rollout/reward_max_global": reward_max,
                    "rollout/terminal_count_global": terminal_count,
                    "memory/gpu_allocated_mib": torch.cuda.memory_allocated(env.device)
                    / (1024**2),
                    "memory/gpu_reserved_mib": torch.cuda.memory_reserved(env.device)
                    / (1024**2),
                    "memory/gpu_peak_allocated_mib": torch.cuda.max_memory_allocated(env.device)
                    / (1024**2),
                    "memory/gpu_peak_reserved_mib": torch.cuda.max_memory_reserved(env.device)
                    / (1024**2),
                    "memory/cpu_peak_rss_mib": resource.getrusage(
                        resource.RUSAGE_SELF
                    ).ru_maxrss
                    / 1024.0,
                    "nonfinite/cumulative_simulator_events_global": cumulative_events,
                    "nonfinite/cumulative_recoveries_global": cumulative_recoveries,
                    "frames/per_rank_cumulative": (i + 1) * frames_per_batch,
                    "frames/global_cumulative": (
                        (i + 1) * frames_per_batch * aa.get_world_size()
                    ),
                }
                _write_tensorboard_scalars(
                    tensorboard_writer, tensorboard_metrics, tensorboard_step
                )

        if need_logging:
            info = {}
            env_frames += data.numel()
            episode_stats.add(data)

            if i % log_interval == 0 and len(episode_stats):
                for k, v in sorted(episode_stats.pop().items(True, True)):
                    key = "train/" + ("/".join(k) if isinstance(k, tuple) else k)
                    info[key] = torch.mean(v.float()).item()
            
            info.update(train_carry)
            info.update(env.extra)
            info.update(env.stats_ema)

            info["env_frames"] = env_frames * aa.get_world_size()
            info["rollout_fps"] = data.numel() / rollout_time * aa.get_world_size()
            info["training_time"] = training_time
        
            if save_interval and save_interval > 0 and should_save(i):
                save(policy, f"checkpoint_{i}")

            if tensorboard_writer is not None:
                _write_tensorboard_scalars(
                    tensorboard_writer,
                    {k: v for k, v in info.items() if k.startswith("train/stats/")},
                    i + 1,
                )
            if run is not None:
                run.log(info, step=i)
            print(OmegaConf.to_yaml({k: v for k, v in info.items() if isinstance(v, (float, int))}))
    
    if diagnostics is not None:
        try:
            diagnostics.finalize()
        except BaseException as error:
            diagnostics.abort(error)
            if tensorboard_writer is not None:
                tensorboard_writer.close()
            env.close(raise_if_closed=False)
            raise
        env.close(raise_if_closed=False)

    if aa.is_main_process():
        if train_recorder is not None:
            train_recorder.flush()

        if need_logging:
            save(policy, "checkpoint_final")

            # A rank-0-only DDP forward may broadcast buffers while other ranks
            # wait at the shutdown barrier. Evaluate multi-GPU checkpoints in a
            # separate single-process job instead.
            if run is not None and not aa.is_distributed():
                policy_eval = policy.get_rollout_policy("eval")
                info, trajs, stats = evaluate(env, policy_eval, render=cfg.eval_render, seed=cfg.seed)
                run.log(info, step=end_iter)

            if run is not None:
                finish_wandb_run(run)
    if tensorboard_writer is not None:
        tensorboard_writer.close()
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
    exit(0)


if __name__ == "__main__":
    main()
