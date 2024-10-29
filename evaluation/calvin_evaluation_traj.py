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
        if step == 0 or step%re_diff_set == 0 or diff_flag == True or self.strategy_flag == True:
            if step == 200:
                print("fsc tag twice")
            if self.strategy_flag:
                self.strategy_flag = False
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

                # 后处理 移动sliding door时候
                if all(keyword in text for keyword in ["push the sliding door"]):
                    max_index = re_out_action[..., 0].argmax()
                    min_index = re_out_action[..., 0].argmin()
                    re_out_action[:,max_index,0] = min(re_out_action[..., 0].max() + 5,224) # 防越界
                    re_out_action[:,min_index,0] = max(0,re_out_action[..., 0].min() - 5) # 防越界

                self.traj_2d_list = []
                for _ in range(10):
                    self.traj_2d_list.append(re_out_action.squeeze(0)) 


        # 小策略
        state_total_tensor = torch.stack(self.state_total_list, dim=0)  
        if step > 30 and "the sliding door" in text and "left" in text: # 暂时只给sliding door用
            # 计算top 30帧的方差
            variance = torch.var(state_total_tensor[-30:], dim=0, unbiased=False)
            if sum(variance)<0.001: # 方差过小，则采用小策略
                # re_out_action = torch.zeros((30, 2))
                # re_out_action[:,0] = re_out_action[:,0] + 130# x
                # re_out_action[:,1] = re_out_action[:,1] + 125# y
                re_out_action = self.traj_2d_list[-1] # 终点坐标点
                re_out_action = re_out_action[-1].expand(30,-1)
                self.traj_2d_list = []
                for _ in range(10):
                    self.traj_2d_list.append(re_out_action.squeeze(0))
                self.reset_point = True
                # print(sum(variance))
            else:
                # 挣脱计算 
                if self.reset_point:
                    self.leave_reset_num = self.leave_reset_num + 1 
                    if self.leave_reset_num > 20:
                        self.reset_point = False
                        self.strategy_flag = True # 再次尝试2d_diff
                        self.leave_reset_num = 0
        # 小策略 防止生成2d_diff的时候没有看到目标物体：柜里的目标和move slider right/left
        # if all(keyword in text for keyword in ["lift", "block", "sliding cabinet"]) or all(keyword in text for keyword in ["push the sliding door"]): 
        # 小策略 防止生成2d_diff的时候没有看到目标物体：柜里的目标
        if all(keyword in text for keyword in ["lift", "block", "sliding cabinet"]): 
            if step < 50:
                re_out_action = torch.zeros((30, 2))
                re_out_action[:,0] = re_out_action[:,0] + 40# x 40
                re_out_action[:,1] = re_out_action[:,1] + 112# y 112
                
                self.traj_2d_list = []
                for _ in range(10):
                    self.traj_2d_list.append(re_out_action.squeeze(0)) 
            else:
                if self.one_time:
                    self.strategy_flag = True

        # 手动画轨迹图
        if debug:
            if draw_flag:
                import cv2
                # 鼠标回调函数
                def create_click_event(points,img):
                    def click_event(event, x, y, flags, params):
                        if event == cv2.EVENT_LBUTTONDOWN:
                            cv2.circle(img, (x, y), 3, (0, 255, 0), -1)
                            points.append((x, y))
                            if len(points) > 1:
                                cv2.line(img, points[-2], points[-1], (0, 0, 255), 2)
                            cv2.imshow('image_draw', img)
                    return click_event
                image_draw = self.preprocessor.rgb_recovery(rgb_data[:,len(self.rgb_list)-1])
                image_draw = cv2.cvtColor(image_draw[0][0].permute(1, 2, 0).cpu().numpy(), cv2.COLOR_BGR2RGB)

                img_sampled = image_draw.copy()
                cv2.imshow('image_draw', image_draw)
                points = []
                cv2.setMouseCallback('image_draw', create_click_event(points,image_draw))
                cv2.waitKey(0)
                # 转为tensor
                points_tensor = torch.tensor(points, dtype=torch.float32)
                if len(points_tensor) == 0:
                    points_tensor_resampled = torch.zeros((30, 2), dtype=torch.float32)
                else:
                    points_tensor_resampled = resample_sequence(points_tensor)
                for point_2d in points_tensor_resampled :
                    cv2.circle(img_sampled, tuple(point_2d.int().tolist()), radius=3, color=(0, 0, 255), thickness=-1)
                cv2.imshow('image_sampled', img_sampled)
                cv2.waitKey(0)
                re_out_action = points_tensor_resampled
                self.traj_2d_list = []
                for _ in range(10):
                    self.traj_2d_list.append(re_out_action.squeeze(0)) 

        # if step == 0:
        #     re_out_action = torch.zeros((30, 2))
        #     # re_out_action[:,0] = re_out_action[:,0] + 30# x 40
        #     # re_out_action[:,1] = re_out_action[:,1] + 82# y 112
        #     self.traj_2d_list = []
        #     for _ in range(10):
        #         self.traj_2d_list.append(re_out_action.squeeze(0))
        # if step == 30: 
        #     self.strategy_flag = True
                
          
        
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

        if False:
            if debug:
                # visualization image
                p = 16
                h_p = 14
                w_p = 14
                rgb_vis= rgb_data.reshape(shape=(rgb_data.shape[0], rgb_data.shape[1], 3, h_p, p, w_p, p)) 
                rgb_vis = rgb_vis.permute(0, 1, 3, 5, 4, 6, 2)
                rgb_vis = rgb_vis.reshape(shape=(rgb_vis.shape[0], rgb_vis.shape[1], h_p * w_p, (p**2) * 3))  # (b, t, n_patches, p*p*3)
                mean = rgb_vis.mean(dim=-1, keepdim=True)
                std = rgb_vis.var(dim=-1, unbiased=True, keepdim=True).sqrt() + 1e-6
                
                masked_patch = prediction["masked_2d_patch"]
                masked_patch = masked_patch.unsqueeze(-1)
                

                patch_image = masked_patch*prediction["obs_targets"]
                patch_image = patch_image * std + mean
                #reshape 196 to 224*224
                patch_image = patch_image.reshape(rgb_vis.shape[0], rgb_vis.shape[1], h_p, w_p, p, p, 3)
                patch_image = patch_image.permute(0, 1, 6, 2, 4, 3, 5)
                patch_image = patch_image.reshape(rgb_vis.shape[0], rgb_vis.shape[1], 3, h_p * p, w_p * p)
                patch_image = self.preprocessor.rgb_recovery(patch_image)
                
                patch_image_preds = masked_patch*prediction["obs_preds"]
                patch_image_preds = patch_image_preds * std + mean
                #reshape 196 to 224*224
                patch_image_preds = patch_image_preds.reshape(rgb_vis.shape[0], rgb_vis.shape[1], h_p, w_p, p, p, 3)
                patch_image_preds = patch_image_preds.permute(0, 1, 6, 2, 4, 3, 5)
                patch_image_preds = patch_image_preds.reshape(rgb_vis.shape[0], rgb_vis.shape[1], 3, h_p * p, w_p * p)
                patch_image_preds = self.preprocessor.rgb_recovery(patch_image_preds)
                
                obs_targets = prediction["obs_targets"]
                obs_targets = obs_targets * std + mean
                obs_targets = obs_targets.reshape(rgb_vis.shape[0], rgb_vis.shape[1], h_p, w_p, p, p, 3)
                obs_targets = obs_targets.permute(0, 1, 6, 2, 4, 3, 5)
                obs_targets = obs_targets.reshape(rgb_vis.shape[0], rgb_vis.shape[1], 3, h_p * p, w_p * p)
                obs_targets = self.preprocessor.rgb_recovery(obs_targets)
                
                
                obs_preds = prediction["obs_preds"]
                obs_preds = obs_preds * std + mean
                obs_preds = obs_preds.reshape(rgb_vis.shape[0], rgb_vis.shape[1], h_p, w_p, p, p, 3)
                obs_preds = obs_preds.permute(0, 1, 6, 2, 4, 3, 5)
                obs_preds = obs_preds.reshape(rgb_vis.shape[0], rgb_vis.shape[1], 3, h_p * p, w_p * p)
                obs_preds = self.preprocessor.rgb_recovery(obs_preds)

                import cv2
                for batch_idx in range(obs_targets.shape[0]):
                    # for seq_idx in range(obs_targets.shape[1]):
                    seq_idx = len(self.rgb_list)-1
                    rgb_static_ori = cv2.cvtColor(obs_targets[batch_idx][seq_idx].permute(1, 2, 0).cpu().numpy(), cv2.COLOR_BGR2RGB)

                    point_2d_resized = prediction["traj_2d_preds"][batch_idx][seq_idx] * 224
                    for point_2d in point_2d_resized :
                        cv2.circle(rgb_static_ori, tuple(point_2d.int().tolist()), radius=3, color=(0, 0, 255), thickness=-1)

                    rgb_static_pred = cv2.cvtColor(obs_preds[batch_idx][seq_idx].permute(1, 2, 0).cpu().numpy(), cv2.COLOR_BGR2RGB)
                    rgb_static_patch_image = cv2.cvtColor(patch_image[batch_idx][seq_idx].permute(1, 2, 0).cpu().numpy(), cv2.COLOR_BGR2RGB)
                    rgb_static_patch_image_preds = cv2.cvtColor(patch_image_preds[batch_idx][seq_idx].permute(1, 2, 0).cpu().numpy(), cv2.COLOR_BGR2RGB)
                    cv2.imshow('Original with pred 5 points RGB Static Image', rgb_static_ori)  # 注意这里需要调整回 HWC 格式
                    cv2.imshow('Predicted RGB Static Image', rgb_static_pred)  # 注意这里需要调整回 HWC 格式
                    cv2.imshow('Patched RGB Static Image', rgb_static_patch_image)  # 注意这里需要调整回 HWC 格式
                    cv2.imshow('Patched_Pred RGB Static Image', rgb_static_patch_image_preds)  # 注意这里需要调整回 HWC 格式
                    # cv2.waitKey(0)
            if debug:
                # visualization image
                p = 16
                h_p = 14
                w_p = 14
                rgb_vis= hand_rgb_data.reshape(shape=(hand_rgb_data.shape[0], hand_rgb_data.shape[1], 3, h_p, p, w_p, p)) 
                rgb_vis = rgb_vis.permute(0, 1, 3, 5, 4, 6, 2)
                rgb_vis = rgb_vis.reshape(shape=(rgb_vis.shape[0], rgb_vis.shape[1], h_p * w_p, (p**2) * 3))  # (b, t, n_patches, p*p*3)
                mean = rgb_vis.mean(dim=-1, keepdim=True)
                std = rgb_vis.var(dim=-1, unbiased=True, keepdim=True).sqrt() + 1e-6
                
                obs_targets = prediction["obs_hand_targets"]
                obs_targets = obs_targets * std + mean
                obs_targets = obs_targets.reshape(rgb_vis.shape[0], rgb_vis.shape[1], h_p, w_p, p, p, 3)
                obs_targets = obs_targets.permute(0, 1, 6, 2, 4, 3, 5)
                obs_targets = obs_targets.reshape(rgb_vis.shape[0], rgb_vis.shape[1], 3, h_p * p, w_p * p)
                obs_targets = self.preprocessor.rgb_recovery(obs_targets)
                
                
                obs_preds = prediction["obs_hand_preds"]
                obs_preds = obs_preds * std + mean
                obs_preds = obs_preds.reshape(rgb_vis.shape[0], rgb_vis.shape[1], h_p, w_p, p, p, 3)
                obs_preds = obs_preds.permute(0, 1, 6, 2, 4, 3, 5)
                obs_preds = obs_preds.reshape(rgb_vis.shape[0], rgb_vis.shape[1], 3, h_p * p, w_p * p)
                obs_preds = self.preprocessor.rgb_recovery(obs_preds)
                import cv2
                for batch_idx in range(obs_targets.shape[0]):
                    # for seq_idx in range(obs_targets.shape[1]):
                    seq_idx = len(self.rgb_list)-1
                    rgb_static_ori = cv2.cvtColor(obs_targets[batch_idx][seq_idx].permute(1, 2, 0).cpu().numpy(), cv2.COLOR_BGR2RGB)
                    rgb_static_pred = cv2.cvtColor(obs_preds[batch_idx][seq_idx].permute(1, 2, 0).cpu().numpy(), cv2.COLOR_BGR2RGB)

                    cv2.imshow('Hand Image', rgb_static_ori)  # 注意这里需要调整回 HWC 格式
                    cv2.imshow('Predicted Hand Image', rgb_static_pred)  # 注意这里需要调整回 HWC 格式

            
        
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
        # 策略调优 先考虑x方向的问题 不考虑slider
        # if self.once_flag and ("the sliding door" not in text):
        # # if self.once_flag:
        #     dis_x = re_out_action[...,0].max() - re_out_action[...,0].min()
        #     mid_x = (re_out_action[..., 0].max() + re_out_action[..., 0].min()) / 2
        #     max_index = re_out_action[..., 0].argmax()
        #     min_index = re_out_action[..., 0].argmin()
        #     dis_y = re_out_action[...,1].max() - re_out_action[...,1].min()
        #     if dis_y < 35 or dis_x > 100: # 扁平形状的才考虑 或者x非常长也可以
        #         if max_index > min_index and dis_x > 60 and traj_2d_pred[0][0] * 224 >  mid_x:                    
        #             self.strategy_flag = True
        #             self.once_flag = False
        #         elif max_index < min_index and dis_x > 60 and traj_2d_pred[0][0] * 224 < mid_x:
        #             self.strategy_flag = True
        #             self.once_flag = False

        return action_pred,re_out_action,traj_2d_pred
