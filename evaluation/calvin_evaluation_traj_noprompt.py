# Copyright (2024) Bytedance Ltd. and/or its affiliates

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Class for evaluating GR-1 on Calvin Benchmark."""
import torch
import torchvision.transforms as T
import torch.nn.functional as F
import numpy as np
from einops import rearrange

import clip

import models.vision_transformer as vits
from models.gr1 import GR1

from calvin_agent.models.calvin_base_model import CalvinBaseModel
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
def resample_sequence(sequence, target_length=30):

    # 确保 sequence 是 float 类型
    sequence = sequence.float()
    sequence = sequence.unsqueeze(0).permute(0, 2, 1)  # 调整形状为 (1, 2, N)
    resampled_sequence = F.interpolate(sequence, size=target_length, mode='linear', align_corners=True)
    resampled_sequence = resampled_sequence.permute(0, 2, 1).squeeze(0)  # 调整回原始形状 (target_length, 2)

    return resampled_sequence

def unnormalize_data(ndata, stats={'min': 0,'max': 224}):
    ndata = (ndata + 1) / 2 # [-1, 1] -> [0, 1] 域
    data = ndata * (stats['max'] - stats['min']) + stats['min']
    # 防越界保护
    data = torch.clamp(data, min=stats['min'], max=stats['max'])
    return data
def contains_words(inst, include_words=[], exclude_words=[]):
    for word in include_words:
        if word not in inst:
            return False
    for word in exclude_words:
        if word in inst:
            return False
    return True

class OneTimeBool:
    def __init__(self):
        self._value = True

    def __bool__(self):
        if self._value:
            self._value = False
            return True
        return False
class GR1CalvinEvaluation(CalvinBaseModel):
    def __init__(self,
                 policy,
                 policy_traj,
                 variant,
                 preprocessor,
                 device
    ):
        """Constructor."""
        self.tokenizer = clip.tokenize
        self.seq_len = variant['seq_len']
        self.chunk_size = variant['chunk_size']
        self.test_chunk_size = variant['test_chunk_size']
        self.use_hand_rgb = variant['use_hand_rgb']
        self.act_dim = variant['act_dim']
        self.state_dim = variant['state_dim']
        self.device = device
        # init for reset 
        self.rgb_list = []
        self.hand_rgb_list = []
        self.state_list = []
        self.state_total_list = []
        self.traj_2d_list = []
        self.rollout_step_counter = 0
        self.strategy_flag = False
        self.once_flag = True
        self.one_time = OneTimeBool()
        # Preprocess
        self.preprocessor = preprocessor 
        self.policy = policy
        self.policy_traj = policy_traj
        # policy config
        self.num_diffusion_iters = 100
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=self.num_diffusion_iters,
            # the choise of beta schedule has big impact on performance
            # we found squared cosine works the best
            beta_schedule='squaredcos_cap_v2',
            # clip output to [-1,1] to improve stability
            clip_sample=True,
            # our network predicts noise (instead of denoised action)
            prediction_type='epsilon'
        )
    def expanded_mask_patch(self, masked_patch, batch_size, sequence_length):
        masked_patch_expand = masked_patch.reshape(batch_size, sequence_length,14,14)  
        kernel = torch.ones((sequence_length, 1, 3, 3)).to(self.device)
        masked_patch_expand = F.conv2d(masked_patch_expand.float(), kernel.float(), padding=1, groups=sequence_length)
        masked_patch_expand = torch.where(masked_patch_expand > 0, torch.tensor(1.0), torch.tensor(0.0))
        masked_patch_expand = masked_patch_expand.reshape(batch_size, sequence_length,14*14)
        return masked_patch_expand
    def reset(self):
        """Reset function."""
        self.traj_2d_list_last = []
        self.rgb_list = []
        self.hand_rgb_list = []
        self.state_list = []
        self.state_total_list = []
        self.traj_2d_list = []
        self.rollout_step_counter = 0
        self.once_flag = True
        self.reset_point = False
        self.leave_reset_num = 0
        self.one_time = OneTimeBool()

    def step(self, obs, goal,step = 0,diff_flag = False, debug=False,draw_flag=False,procs_id=0):
        """Step function."""
        step = step
        # Language
        # goal = 'grasp and lift the block'
        text = goal
        tokenized_text = self.tokenizer(text)

        # RGB
        rgb = rearrange(torch.from_numpy(obs['rgb_obs']['rgb_static']), 'h w c -> c h w')
        hand_rgb = rearrange(torch.from_numpy(obs['rgb_obs']['rgb_gripper']), 'h w c -> c h w')
        self.rgb_list.append(rgb)
        self.hand_rgb_list.append(hand_rgb)

        # State
        state = obs['robot_obs']
        arm_state = state[:6]
        gripper_state = state[-1]
        state = torch.from_numpy(np.hstack([arm_state, gripper_state]))
        self.state_list.append(state)
        self.state_total_list.append(state)
        # Buffer
        buffer_len = len(self.rgb_list)
        if buffer_len > self.seq_len:
            self.rgb_list.pop(0)
            self.hand_rgb_list.pop(0)
            self.state_list.pop(0)
            assert len(self.rgb_list) == self.seq_len
            assert len(self.hand_rgb_list) == self.seq_len
            assert len(self.state_list) == self.seq_len
            buffer_len = len(self.rgb_list)
        
        # Static RGB
        c, h, w = rgb.shape
        rgb_data = torch.zeros((1, self.seq_len, c, h, w))
        rgb_tensor = torch.stack(self.rgb_list, dim=0)  # (t, c, h, w)
        rgb_data[0, :buffer_len] = rgb_tensor

        # Hand RGB
        c, h, w = hand_rgb.shape
        hand_rgb_data = torch.zeros((1, self.seq_len, c, h, w))
        hand_rgb_tensor = torch.stack(self.hand_rgb_list, dim=0)  # (t, c, h, w)
        hand_rgb_data[0, :buffer_len] = hand_rgb_tensor

        # State
        state_tensor = torch.stack(self.state_list, dim=0)  # (l, act_dim)
        gripper_state_data = - torch.ones((1, self.seq_len)).float()
        gripper_state_data[0, :buffer_len] = state_tensor[:, 6]
        gripper_state_data = (gripper_state_data + 1.0) / 2
        gripper_state_data = gripper_state_data.long()
        gripper_state_data = F.one_hot(gripper_state_data, num_classes=2).float()  # (1, t, 2)
        arm_state_data = torch.zeros((1, self.seq_len, self.act_dim - 1)).float()  # (1, t, act_dim - 1)
        arm_state_data[0, :buffer_len] = state_tensor[:, :6]

        # Attention mask
        attention_mask = torch.zeros(1, self.seq_len).long()
        attention_mask[0, :buffer_len] = 1

        # Forward pass
        tokenized_text = tokenized_text.to(self.device)
        rgb_data = rgb_data.to(self.device)
        hand_rgb_data = hand_rgb_data.to(self.device)
        arm_state_data = arm_state_data.to(self.device)
        gripper_state_data = gripper_state_data.to(self.device)
        state_data = {'arm': arm_state_data, 'gripper': gripper_state_data}
        attention_mask = attention_mask.to(self.device)

        rgb_data, hand_rgb_data = self.preprocessor.rgb_process(rgb_data, hand_rgb_data, train=False)

        re_diff_set = 200
        # 大策略
        if step == 0:
            with torch.no_grad():
                self.noise_scheduler.set_timesteps(self.num_diffusion_iters)
                rgb_static_norm = rgb_data[:,len(self.rgb_list)-1].unsqueeze(0)
                # rgb_static_norm = rgb_data[:,-1].unsqueeze(0)
                noisy_action = torch.randn([1,30,2], device=self.device)
                out_action = noisy_action
                language_embedding, obs_embeddings, patch_embeddings = None, None, None

                for k in self.noise_scheduler.timesteps:
                        # predict noise
                        noise_pred, language_embedding, obs_embeddings, patch_embeddings = self.policy_traj(rgb_static_norm, tokenized_text, timesteps=k, noisy_actions=out_action,
                                            language_embedding=language_embedding, obs_embeddings=obs_embeddings, patch_embeddings=patch_embeddings)
                        # inverse diffusion step (remove noise)
                        out_action = self.noise_scheduler.step(
                            model_output=noise_pred,
                            timestep=k,
                            sample=out_action
                        ).prev_sample
                re_out_action = unnormalize_data(out_action)

                self.traj_2d_list = []
                for _ in range(10):
                    self.traj_2d_list.append(re_out_action.squeeze(0)) 


        # 送入网络中            
        re_out_action = self.traj_2d_list[-1]
        traj_2d = torch.zeros((1, self.seq_len, 30, 2))
        traj_2d_tensor = torch.stack(self.traj_2d_list, dim=0)  # (t, c, h, w)
        traj_2d[0, :buffer_len] = traj_2d_tensor[:buffer_len]
        traj_2d = traj_2d.to(self.device)

        with torch.no_grad():
            prediction = self.policy(
                rgb=rgb_data, 
                hand_rgb=hand_rgb_data,
                state=state_data,
                language=tokenized_text,
                action_2d = traj_2d,
                attention_mask=attention_mask
        )

       
        # Arm action
        arm_action_preds = prediction['arm_action_preds']  # (1, t, chunk_size, act_dim - 1)
        arm_action_preds = arm_action_preds.view(-1, self.chunk_size, self.act_dim - 1)  # (t, chunk_size, act_dim - 1)
        arm_action_preds = arm_action_preds[attention_mask.flatten() > 0]

        # Gripper action
        gripper_action_preds = prediction['gripper_action_preds']  # (1, t, chunk_size, 1)
        gripper_action_preds = gripper_action_preds.view(-1, self.chunk_size, 1)  # (t, chunk_size, 1)
        gripper_action_preds = gripper_action_preds[attention_mask.flatten() > 0]

        # Use the last action
        arm_action_pred = arm_action_preds[-1, :self.test_chunk_size]  # (test_chunk_size, act_dim - 1)
        gripper_action_pred = gripper_action_preds[-1, :self.test_chunk_size]  # (test_chunk_size, 1)
        gripper_action_pred = torch.nn.Sigmoid()(gripper_action_pred)
        gripper_action_pred = gripper_action_pred > 0.5
        gripper_action_pred = gripper_action_pred.int().float()
        gripper_action_pred = gripper_action_pred * 2.0 - 1.0
        action_pred = torch.cat((arm_action_pred, gripper_action_pred), dim=-1)  # (act_dim,)
        action_pred = action_pred.detach().cpu()

        self.rollout_step_counter += 1

        traj_2d_pred = prediction['traj_2d_preds'][0][-1]

        return action_pred,re_out_action,traj_2d_pred
