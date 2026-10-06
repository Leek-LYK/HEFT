"""Privileged teacher view on the SAME live HEFT pool used by the student."""
import torch
from active_adaptation.envs.xwbc import XWBCPool
from active_adaptation.utils.math import quat_apply_inverse,quat_in_frame,quat_to_rot6d

class StagedPool(XWBCPool):
    def teacher_observation(self,observation):
        d=self.robot.data;idx=self.command.keypoint_idx_asset
        q=d.root_link_quat_w[:,None]
        pos=quat_apply_inverse(q,d.body_link_pos_w[:,idx]-d.root_link_pos_w[:,None])
        rot=quat_to_rot6d(quat_in_frame(q,d.body_link_quat_w[:,idx])).flatten(1)
        vel=quat_apply_inverse(q,d.body_link_lin_vel_w[:,idx]-d.root_link_lin_vel_w[:,None])/10
        # All quantities come from the current actual simulator state. No second rollout.
        return torch.cat((observation['privileged'],d.root_link_lin_vel_b/10,
                          (d.root_link_pos_w[:,2]-self.env.scene.env_origins[:,2])[:,None],
                          pos.flatten(1),rot,vel.flatten(1)),-1).clamp(-20,20)
