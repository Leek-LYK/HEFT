"""HEFT tracking environments with explicit cross-embodiment policy boundaries."""
import torch
from tensordict import TensorDict
from omegaconf import OmegaConf

from active_adaptation.envs.mdp.commands.motion_tracking.core import MotionTrackingCommand
from active_adaptation.utils.math import quat_apply_inverse, quat_in_frame, quat_to_rot6d
from active_adaptation.utils.helpers import TensorRingBuffer
from active_adaptation.utils.xwbc_data import BODIES, ROOT


class XWBCMotionCommand(MotionTrackingCommand):
    """Uniform training starts and deterministic complete-clip evaluation resets."""
    fixed_motion_ids = None

    def sample_init(self, env_ids):
        ds = self.dataset
        if self.fixed_motion_ids is None:
            lengths = ds.reset(env_ids)
            frame = (torch.rand(len(env_ids), device=self.device) * (lengths-2).clamp_min(1)).long()
        else:
            ids = self.fixed_motion_ids[env_ids]
            ds._motion_ids_A[env_ids] = ids
            ds._dataset_ids_A[env_ids] = ds.motion_to_dataset_id[ids]
            ds._len_A[env_ids] = ds.global_lengths[ids].to(ds._len_A.dtype)
            ds._invalidate_cache(env_ids)
            lengths = ds._len_A[env_ids]
            frame = torch.zeros_like(lengths)
        self.lengths[env_ids] = lengths
        self.t[env_ids] = frame.to(self.t.dtype)
        self._reinit_requested[env_ids] = False
        teacher, student = ds.get_teacher_student_slice(env_ids, self.t[env_ids], teacher_steps=1, student_steps=1)
        self._set_motion_origin_offset(env_ids, teacher, student)
        motion, _ = self._apply_motion_origin_offset(teacher, student, env_ids)
        self.sample_init_robot(env_ids, motion, lift_height=.02)


def environment_config(robot, bank, num_envs, evaluation=False):
    import active_adaptation.learning  # register baseline Hydra configs
    from hydra import compose, initialize_config_dir
    profile = {'g1': 'G1_tracking', 'roban': 'Roban_tracking'}[robot]
    with initialize_config_dir(version_base=None, config_dir=str(ROOT / 'cfg')):
        cfg = compose(config_name='train', overrides=[f'task/profile={profile}'])
    OmegaConf.resolve(cfg)
    task = cfg.task
    OmegaConf.set_struct(task, False)
    task.num_envs = num_envs
    task.viewer.headless = True
    task.student_train = False  # use absolute tracking reward, not HEFT stage selection
    task.recover_nonfinite_state = True
    task.max_episode_length = int(bank.lengths.max().item()) + 2 if evaluation else 1000
    task.command._target_ = 'active_adaptation.envs.xwbc.XWBCMotionCommand'
    task.command.dataset = dict(groups=bank.groups(robot), keep_on_cpu=True, skip_initial_frames=0,
                                resample_on_reset=True, offset_pos_z=0.,
                                cache_request_range=[-4, 8], cache_fill_range=[-8, 40])
    task.command.future_steps = [0]
    task.command.student_future_steps = [0]
    task.command.reinit_prob = 0.
    task.command.boot_indicator_max = 0
    task.command.student_motion_randomization.enable = False
    task.command.keypoint_patterns = ['^' + name + '$' for name in BODIES[robot][1:]]
    task.command.required_motion_body_patterns = list(task.command.required_motion_body_patterns) + task.command.keypoint_patterns
    task.command.init_noise = {k: 0. for k in task.command.init_noise}
    task.command.shared_joint_pos = {'std': {'.*': 0.}, 'bias': {'.*': 0.}}
    task.command.shared_joint_vel = {'std': {'.*': 0.}}
    # No payload, mass augmentation, WPC, applied wrench or perturbation callbacks.
    task.randomization = {}
    # Actor/proprio and privileged critic are built separately by XWBCPool.
    task.observation = {}
    task.action.boot_protect = False
    task.action.max_delay = 0
    task.action.alpha = [1., 1.]
    task.action.prev_action_obs = 'applied'
    task.action.torque_limit_scale_range = [1., 1.]
    task.termination.pop('motion_xy_range_termination', None)
    task.termination.nonfinite_state = {}
    # Same control task for both robot-specific action interfaces.
    task.reward.loco.feet_air_time_ref_dense.enabled = False
    for group in task.reward.values():
        if OmegaConf.is_dict(group):
            group.pop('_scale_factor_', None)
    return task


class XWBCPool:
    def __init__(self, robot, bank, num_envs, history=8, evaluation=False, history_steps=None, action_history=8):
        from active_adaptation.envs import SimpleEnv
        self.robot_name, self.bank, self.evaluation = robot, bank, evaluation
        self.cfg = environment_config(robot, bank, num_envs, evaluation)
        self.resolved_config = OmegaConf.to_container(self.cfg, resolve=True)
        self.env = SimpleEnv(self.cfg)
        self.env.eval() if evaluation else self.env.train()
        self.command = self.env.command_manager
        self.robot = self.env.robot
        resolved = {self.robot.body_names[i] for i in self.command.keypoint_idx_asset.tolist()}
        if resolved != set(BODIES[robot][1:]):
            raise ValueError(f'Incomplete tracking keypoints: {resolved}')
        if self.env.randomizations:
            raise ValueError('The XWBC protocol requires unloaded, unperturbed environments')
        self.device = self.env.device
        self.num_envs, self.action_dim = num_envs, self.env.action_dim
        self.joint_ids = self.env.action_manager.joint_ids
        self.history_steps = None if history_steps is None else list(history_steps)
        self.action_history = int(action_history)
        if self.history_steps is None:
            # Keep v0.1 run01's interleaved, oldest-first history exactly intact.
            self.history = torch.zeros(num_envs, history, 3*self.action_dim+6, device=self.device)
            self.proprio_dim = history * (3*self.action_dim+6)
        else:
            if (not self.history_steps or self.history_steps[0] != 0
                    or len(set(self.history_steps)) != len(self.history_steps)
                    or any(not isinstance(x, int) or isinstance(x, bool) or x < 0 for x in self.history_steps)):
                raise ValueError('History steps must be unique nonnegative integers starting with 0')
            if not 1 <= self.action_history <= self.env.action_manager._action_applied_buf.capacity:
                raise ValueError('Action history exceeds the HEFT action buffer capacity')
            self._state_history = TensorRingBuffer(num_envs, max(self.history_steps)+1,
                2*self.action_dim+6, device=self.device, dtype=torch.float32)
            self._history_indices = torch.tensor(self.history_steps, device=self.device)
            self.proprio_dim = len(self.history_steps)*(2*self.action_dim+6)+self.action_history*self.action_dim
        self.source = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.fixed_source = None
        self.all_ids = torch.arange(num_envs, device=self.device)
        self.reset(self.all_ids)

    @torch.no_grad()
    def reset(self, ids):
        if ids.numel() == 0:
            return
        mask = torch.zeros(self.num_envs, 1, device=self.device, dtype=torch.bool)
        mask[ids] = True
        self.env.reset(TensorDict({'_reset': mask}, [self.num_envs], device=self.device))
        self.env.sim.forward()
        self.env.sim.sense()
        self.env.scene.update(0.)
        # Only reset rows need new reference buffers; active episodes keep their cache.
        command = self.command
        teacher, student = command.dataset.get_teacher_student_slice(
            ids, command.t[ids], teacher_steps=command.future_steps,
            student_steps=command.student_future_steps)
        teacher, student = command._apply_motion_origin_offset(teacher, student, ids)
        command._motion[ids] = teacher
        command._motion_student[ids] = student
        self.source[ids] = (torch.randint(3, (len(ids),), device=self.device)
                            if self.fixed_source is None else self.fixed_source)
        self._reset_history(ids)

    def _reset_history(self, ids):
        if self.history_steps is None:
            self.history[ids] = 0.
            self.history[ids, -1] = self.proprio()[ids]
        else:
            self._state_history.reset(ids)
            self._state_history.buffer[ids, self._state_history.head] = self.proprio()[ids, :-self.action_dim]

    def proprio(self):
        d = self.robot.data
        gravity = torch.zeros(self.num_envs, 3, device=self.device)
        gravity[:, 2] = -1
        return torch.cat((d.joint_pos[:, self.joint_ids], d.joint_vel[:, self.joint_ids] / 10,
                          d.root_link_ang_vel_b / 10, quat_apply_inverse(d.root_link_quat_w, gravity),
                          self.env.action_manager.applied_action), -1).clamp(-20, 20)

    def privileged(self):
        c, d = self.command, self.robot.data
        ref = c._motion
        root_diff = ref.root_pos_w[:, 0] + self.env.scene.env_origins - d.root_link_pos_w
        root_diff = quat_apply_inverse(d.root_link_quat_w, root_diff)
        rot = quat_to_rot6d(quat_in_frame(d.root_link_quat_w, ref.root_quat_w[:, 0])).flatten(1)
        return torch.cat((ref.joint_pos[:, 0, c.joint_idx_motion]-d.joint_pos[:, c.joint_idx_asset],
                          (ref.joint_vel[:, 0, c.joint_idx_motion]-d.joint_vel[:, c.joint_idx_asset])/10,
                          root_diff, rot), -1).clamp(-20, 20)

    def observe(self):
        return dict(proprio=self.history_observation(), privileged=self.privileged(),
                    motion=self.command.dataset._motion_ids_A.clone(), frame=self.command.t.long().clone(),
                    source=self.source.clone())

    def history_observation(self):
        if self.history_steps is None:
            return self.history.flatten(1).clone()
        states = self._state_history.take(self._history_indices).flatten(1)
        actions = self.env.action_manager.get_recent_action_obs(self.action_history).flatten(1)
        return torch.cat((states, actions), -1)

    @torch.no_grad()
    def step(self, action, mean):
        td = TensorDict(dict(action=action, loc=mean), [self.num_envs], device=self.device)
        result = self.env.step(td)['next']
        if self.history_steps is None:
            self.history = self.history.roll(-1, 1)
            self.history[:, -1] = self.proprio()
        else:
            self._state_history.push(self.proprio()[:, :-self.action_dim])
        return result, self.observe()

    @torch.no_grad()
    def metrics(self):
        c, d = self.command, self.robot.data
        idx = c.keypoint_idx_asset
        ref = c._motion
        reference = ref.body_pos_w[:, 0, c.keypoint_idx_motion] + self.env.scene.env_origins[:, None]
        actual = d.body_link_pos_w[:, idx]
        world = (actual-reference).norm(dim=-1).mean(-1)
        # Root-relative metric in world axes; global heading errors remain visible.
        rel = ((actual-d.root_link_pos_w[:, None]) -
               (reference-(ref.root_pos_w[:, 0]+self.env.scene.env_origins)[:, None])).norm(dim=-1).mean(-1)
        vel = (d.body_link_lin_vel_w[:, idx]-ref.body_vel_w[:, 0, c.keypoint_idx_motion]).norm(dim=-1).mean(-1)
        dot = (d.root_link_quat_w * ref.root_quat_w[:, 0]).sum(-1).abs().clamp(0, 1)
        return dict(mpkpe_world_m=world, mpkpe_root_relative_m=rel, velocity_mps=vel,
                    root_rotation_rad=2*dot.acos(),
                    root_position_m=(d.root_link_pos_w-ref.root_pos_w[:, 0]-self.env.scene.env_origins).norm(dim=-1))

    def close(self):
        self.env.close(raise_if_closed=False)
