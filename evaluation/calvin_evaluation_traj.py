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
from traj_predict.model.TrajPredictPolicy import Traj_PreProcess
def unnormalize_data(ndata, stats={'min': 0,'max': 224}):
    ndata = (ndata + 1) / 2 # [-1, 1] -> [0, 1] 域
    data = ndata * (stats['max'] - stats['min']) + stats['min']
    return data
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

    def reset(self):
        """Reset function."""
        self.rgb_list = []
        self.hand_rgb_list = []
        self.state_list = []
        self.rollout_step_counter = 0

    def step(self, obs, goal):
        """Step function."""
        # Language
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

        with torch.no_grad():
            prediction = self.policy(
                rgb=rgb_data, 
                hand_rgb=hand_rgb_data,
                state=state_data,
                language=tokenized_text,
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

        # import time
        # start_time = time.time()
        # traj_pred
        self.noise_scheduler.set_timesteps(self.num_diffusion_iters)
        rgb_static_norm = rgb_data[:,-1].unsqueeze(0)
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
        # end_time = time.time()
        # execution_time = end_time - start_time

        # print(f"Code block executed in {execution_time} seconds.")
        return action_pred,re_out_action