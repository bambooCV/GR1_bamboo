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

"""GR-1 model."""
import torch
import torch.nn as nn

import transformers
from flamingo_pytorch import PerceiverResampler

from transformers import GPT2Model
from models.vision_transformer import Block
from models.transformer_utils import get_2d_sincos_pos_embed
import torch.nn.functional as F
class TokenLearnerAttention(nn.Module):
    def __init__(self, input_dim, num_points, num_tokens, output_dim, num_heads=2):
        super(TokenLearnerAttention, self).__init__()
        self.num_tokens = num_tokens
        self.input_dim = input_dim
        # 位置编码
        self.positional_encoding = nn.Parameter(torch.randn(num_points, input_dim))
        # 可学习的token表示
        self.token_embed = nn.Parameter(torch.randn(num_tokens, input_dim))
        
        # 多头注意力机制
        self.self_attention = nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads)
        
        # 最后的线性层，将输入维度转换为输出维度
        self.linear = nn.Linear(input_dim, output_dim)
        
    
    def forward(self, x):
        batch_size = x.size(0)
        seq_length = x.size(1)
        num_points = x.size(2)
        # 位置编码添加到点特征中
        pos_encoding = self.positional_encoding.unsqueeze(1).repeat(1, batch_size * seq_length, 1)  # (seq_length, batch_size * seq_length, input_dim)

        # 将x的维度调整为 (batch_size * seq_length, num_points, input_dim)
        x_reshaped = x.view(batch_size * seq_length, num_points, self.input_dim).permute(1, 0, 2)  # (num_points, batch_size * seq_length, input_dim)
        x_reshaped += pos_encoding  # 添加位置编码
        # 重复token表示以适应批量大小和点的数量
        tokens = self.token_embed.unsqueeze(1).repeat(1, batch_size * seq_length, 1)  # (num_tokens, batch_size * seq_length, input_dim)
        
        # 使用token表示进行注意力计算
        attn_output, _ = self.self_attention(tokens, x_reshaped, x_reshaped)
        
        # 结果变换回(batch_size, num_points, num_tokens, input_dim)
        attn_output = attn_output.permute(1, 0, 2).view(batch_size, seq_length, self.num_tokens, self.input_dim)
        
        # 将输出通过线性层转换为期望的输出维度
        output = self.linear(attn_output)
        
        return output

class GR1(nn.Module):
    def __init__(
            self,
            model_clip,
            model_mae,
            state_dim,
            act_dim,
            hidden_size,
            sequence_length,
            chunk_size,
            training_target,
            img_feat_dim,
            patch_feat_dim,
            lang_feat_dim,
            resampler_params,
            without_norm_pixel_loss=False,
            use_hand_rgb=True,
            use_2d_traj=True,
            **kwargs
    ):
        super().__init__()
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.sequence_length = sequence_length
        self.chunk_size = chunk_size

        # GPT
        self.hidden_size = hidden_size
        config = transformers.GPT2Config(
            vocab_size=1,
            n_embd=hidden_size,
            **kwargs
        )
        self.transformer = GPT2Model(config)

        # Perciever resampler
        self.n_patch_latents = resampler_params['num_latents']
        self.perceiver_resampler = PerceiverResampler(
            dim=patch_feat_dim,
            depth=resampler_params['depth'],
            dim_head=resampler_params['dim_head'],
            heads=resampler_params['heads'],
            num_latents=self.n_patch_latents,
            num_media_embeds=resampler_params['num_media_embeds']) 
    

        # CLIP for language encoding
        self.model_clip = model_clip
        for _, param in self.model_clip.named_parameters():
            param.requires_grad = False

        # MAE for image encoding
        self.model_mae = model_mae
        for _, param in self.model_mae.named_parameters():
            param.requires_grad = False
        
        self.n_patches = 49
        self.patch_size = 16
        self.image_size = 224
        self.img_feat_dim = img_feat_dim
        self.lang_feat_dim = lang_feat_dim
        self.patch_feat_dim = patch_feat_dim
        self.use_hand_rgb = use_hand_rgb
        self.use_2d_traj = use_2d_traj

        self.act_pred = False
        self.fwd_pred = False
        self.fwd_pred_hand = False
        if 'act_pred' in training_target:
            self.act_pred = True
        if 'fwd_pred' in training_target:
            self.fwd_pred = True
        if 'fwd_pred_hand' in training_target:
            self.fwd_pred_hand = True
        self.act_2d_pred = True
        
        
        self.without_norm_pixel_loss = without_norm_pixel_loss

        # Embedding functions for states
        self.embed_arm_state = torch.nn.Linear(self.state_dim - 1, hidden_size)
        self.embed_gripper_state = torch.nn.Linear(2, hidden_size) # one-hot gripper state
        self.embed_state = torch.nn.Linear(2*hidden_size, hidden_size)
        if self.use_2d_traj:
            self.traj_2d_tokens = 5
            self.embed_traj_2d = nn.Embedding(224, hidden_size) # 224 是索引范围
            self.traj_2d_resampler = TokenLearnerAttention(hidden_size*2, 30, self.traj_2d_tokens, hidden_size) # 3个learnable token 代表2的坐标
            
            self.n_patch_latents_2d = resampler_params['num_latents']
            self.perceiver_resampler_2d = PerceiverResampler(
                dim=patch_feat_dim,
                depth=resampler_params['depth'],
                dim_head=resampler_params['dim_head'],
                heads=resampler_params['heads'],
                num_latents=self.n_patch_latents_2d,
                num_media_embeds=resampler_params['num_media_embeds']) 
            
            if self.act_2d_pred:
                # sub prediction 2d traj
                self.pred_2d_points_num = 5
                self.action_2d_queries = nn.Embedding(self.pred_2d_points_num, hidden_size) 
                self.pred_act_2d_mlps = nn.ModuleList([
                    nn.Linear(hidden_size, hidden_size//2),
                    nn.Linear(hidden_size//2, hidden_size//2)])
                self.pred_2d_points = nn.Linear(hidden_size//2, 2) # 2d points action 
        
        # Relative timestep embedding
        self.embed_timestep = nn.Embedding(self.sequence_length, hidden_size)

        # Embedding function for languages
        self.embed_lang = torch.nn.Linear(self.lang_feat_dim, hidden_size)

        # Embedding functions for images
        self.embed_hand_img = torch.nn.Linear(self.img_feat_dim, hidden_size)
        self.embed_img = torch.nn.Linear(self.img_feat_dim, hidden_size)
        self.embed_hand_patch = torch.nn.Linear(self.patch_feat_dim, hidden_size) 
        self.embed_patch = torch.nn.Linear(self.patch_feat_dim, hidden_size)

        # Layer norm
        self.embed_ln = nn.LayerNorm(hidden_size)

        # Action query token
        self.action_queries = nn.Embedding(1, hidden_size) # arm + gripper, weight from bytedance
        self.action_chunk_queries = nn.Embedding(chunk_size, hidden_size) 
        self.action_chunk_queries.weight.data.fill_(0) # finetune it from zero weight

        
        # Observation query token
        self.obs_queries = nn.Embedding(self.n_patch_latents + 1, self.hidden_size)
        self.obs_hand_queries = nn.Embedding(self.n_patch_latents + 1, self.hidden_size)

        # Action prediction
        self.pred_act_mlps = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size//2),
            nn.Linear(hidden_size//2, hidden_size//2)])
        self.pred_arm_act = nn.Linear(hidden_size//2, self.act_dim-1) # arm action
        self.pred_gripper_act = nn.Linear(hidden_size//2, 1) # gripper action (binary)


        
        # Forward prediction
        self.decoder_embed = nn.Linear(hidden_size, hidden_size, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, hidden_size))
        decoder_depth = 2
        self.decoder_blocks = nn.ModuleList([
            Block(hidden_size, 16, 4, qkv_bias=True, qk_scale=None, norm_layer=nn.LayerNorm)
            for i in range(decoder_depth)])
        self.decoder_norm = nn.LayerNorm(hidden_size)
        self.decoder_pred = nn.Linear(hidden_size, self.patch_size**2 * 3, bias=True) # decoder to patch
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, (self.image_size//self.patch_size)**2,
            hidden_size), requires_grad=False)  # (1, n_patch, h)
        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], (self.image_size//self.patch_size))
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))
    
    
    def expanded_mask_patch(self, masked_patch, batch_size, sequence_length):
        masked_patch_expand = masked_patch.reshape(batch_size, sequence_length,14,14)  
        traj_2d_kernel = torch.ones((sequence_length, 1, 3, 3)).to(device=masked_patch.device)
        masked_patch_expand = F.conv2d(masked_patch_expand.float(), traj_2d_kernel.float(), padding=1, groups=sequence_length)
        masked_patch_expand = torch.where(masked_patch_expand > 0, torch.tensor(1.0), torch.tensor(0.0))
        masked_patch_expand = masked_patch_expand.reshape(batch_size, sequence_length,14*14)
        return masked_patch_expand
    
    
    def forward(self, 
                rgb, 
                hand_rgb, 
                state, 
                language, 
                action_2d,
                attention_mask
    ):
        obs_preds = None
        obs_hand_preds = None
        obs_targets = None
        obs_hand_targets = None
        arm_action_preds = None
        gripper_action_preds = None

        batch_size, sequence_length, c, h, w = rgb.shape

        # Embed state
        arm_state = state['arm']
        gripper_state = state['gripper']
        arm_state_embeddings = self.embed_arm_state(arm_state.view(batch_size, sequence_length, self.state_dim-1))
        gripper_state_embeddings = self.embed_gripper_state(gripper_state)
        state_embeddings = torch.cat((arm_state_embeddings, gripper_state_embeddings), dim=2)
        state_embeddings = self.embed_state(state_embeddings)  # (b, t, h)

        # Embed language
        lang_embeddings = self.model_clip.encode_text(language)
        lang_embeddings = lang_embeddings / (lang_embeddings.norm(dim=1, keepdim=True) + 1e-6) # normalization 
        lang_embeddings = self.embed_lang(lang_embeddings.float())  # (b, h)
    
        # Get obs and patch feature from MAE
        obs_embeddings, patch_embeddings = self.model_mae(
            rgb.view(batch_size*sequence_length, c, h, w))  # (b * t, img_feat_dim), (b * t, n_patches, patch_feat_dim)
        obs_embeddings = obs_embeddings.view(batch_size, sequence_length, -1)  # (b, t, img_feat_dim)
        if self.use_hand_rgb:
            hand_obs_embeddings, hand_patch_embeddings = self.model_mae(
                hand_rgb.view(batch_size*sequence_length, c, h, w))
            hand_obs_embeddings = hand_obs_embeddings.view(batch_size, sequence_length, -1)  # (b, t, img_feat_dim)
        if self.fwd_pred:
            p = self.patch_size
            h_p = h // p
            w_p = w // p
            rgb = rgb.reshape(shape=(batch_size, sequence_length, 3, h_p, p, w_p, p)) 
            obs_targets = rgb.permute(0, 1, 3, 5, 4, 6, 2)
            obs_targets = obs_targets.reshape(shape=(batch_size, sequence_length, h_p * w_p, (p**2) * 3))  # (b, t, n_patches, p*p*3)
            if not self.without_norm_pixel_loss:
                # norm the target 
                obs_targets = (obs_targets - obs_targets.mean(dim=-1, keepdim=True)
                    ) / (obs_targets.var(dim=-1, unbiased=True, keepdim=True).sqrt() + 1e-6)
            if self.fwd_pred_hand:
                hand_rgb = hand_rgb.reshape(shape=(batch_size, sequence_length, 3, h_p, p, w_p, p))
                obs_hand_targets = hand_rgb.permute(0, 1, 3, 5, 4, 6, 2)
                obs_hand_targets = obs_hand_targets.reshape(shape=(batch_size, sequence_length, h_p * w_p, (p**2)*3))  # (b, t, n_patches, p*p*3)
                if not self.without_norm_pixel_loss:
                    # norm the target 
                    obs_hand_targets = (obs_hand_targets - obs_hand_targets.mean(dim=-1, keepdim=True)
                        ) / (obs_hand_targets.var(dim=-1, unbiased=True, keepdim=True).sqrt() + 1e-6)            
        if self.use_2d_traj:
            action_2d = torch.round(action_2d).int()
            action_2d = torch.clamp(action_2d, min=0, max=223)
            num_patches_per_row = 224 // 16
            patch_row = action_2d[...,1] // 16
            patch_col = action_2d[...,0] // 16
            patch_index = patch_row * num_patches_per_row + patch_col
            masked_patch = torch.zeros((batch_size,sequence_length,patch_embeddings.shape[-2])).to(device=patch_embeddings.device)
            for patch_index_i in range(patch_index.size(0)):
                for patch_index_j in range(patch_index.size(1)):
                    # 获取当前序列中的索引
                    indices = patch_index[patch_index_i, patch_index_j]
                    # 将 masked_patch 中对应位置设为 1
                    masked_patch[patch_index_i, patch_index_j, indices] = 1
            masked_patch = self.expanded_mask_patch(masked_patch, batch_size, sequence_length) # 是否扩展mask
            selected_patch_embeddings = patch_embeddings * masked_patch.view(batch_size*sequence_length, -1).unsqueeze(-1)
            selected_patch_embeddings = selected_patch_embeddings.unsqueeze(1)  # (b * t, 1, n_patches, patch_feat_dim)
            selected_patch_embeddings = self.perceiver_resampler_2d(selected_patch_embeddings)  # (b * t, 1, n_patch_latents, patch_feat_dim)
            selected_patch_embeddings = selected_patch_embeddings.squeeze(1)  # (b * t, n_patch_latents, patch_feat_dim)
            selected_patch_embeddings = selected_patch_embeddings.view(batch_size, sequence_length, self.n_patch_latents_2d, self.patch_feat_dim) 
            # 把坐标点embedding 并resample到3个点
            x_embeddings = self.embed_traj_2d(action_2d[:, :, :, 0])
            y_embeddings = self.embed_traj_2d(action_2d[:, :, :, 1])
            embed_traj_2d = torch.cat((x_embeddings, y_embeddings), dim=-1)
            embed_traj_2d = self.traj_2d_resampler(embed_traj_2d)
            # embed_traj_2d = embed_traj_2d.view(batch_size*sequence_length, -1,embed_traj_2d.shape[-1])  # (b*t,points_num, h)
            # # action_2d_traj = action_2d_traj + time_embeddings.view(sequence_length, 1, self.hidden_size)
            # # stacked_inputs = torch.cat((stacked_inputs, action_2d_traj), dim=2) 
            # # 把轨迹点和patch特征拼接起来作为perceiver的输入
            # patch_embeddings = torch.concat((patch_embeddings, embed_traj_2d), dim=1) 
        # Use resampler to process patch embeddings
        patch_embeddings = patch_embeddings.unsqueeze(1)  # (b * t, 1, n_patches, patch_feat_dim)
        patch_embeddings = self.perceiver_resampler(patch_embeddings)  # (b * t, 1, n_patch_latents, patch_feat_dim)
        patch_embeddings = patch_embeddings.squeeze(1)  # (b * t, n_patch_latents, patch_feat_dim)
        patch_embeddings = patch_embeddings.view(batch_size, sequence_length, self.n_patch_latents, self.patch_feat_dim)  # (b, t, n_patch_latents, patch_feat_dim)
        if self.use_2d_traj:
            patch_embeddings = torch.cat((patch_embeddings, selected_patch_embeddings), dim=2) 
        if self.use_hand_rgb:
            hand_patch_embeddings = hand_patch_embeddings.unsqueeze(1)
            hand_patch_embeddings = self.perceiver_resampler(hand_patch_embeddings)
            hand_patch_embeddings = hand_patch_embeddings.squeeze(1)
            hand_patch_embeddings = hand_patch_embeddings.view(batch_size, sequence_length, self.n_patch_latents, self.patch_feat_dim)  # (b, t, n_patch_latents, patch_feat_dim)
        
        # Embed images and patches
        obs_embeddings = self.embed_img(obs_embeddings.float())  # (b, t, h)
        patch_embeddings = self.embed_patch(patch_embeddings.float())  # (b, t, n_patch_latents, h)
        if self.use_hand_rgb:
            hand_obs_embeddings = self.embed_hand_img(hand_obs_embeddings.float())  # (b, t, h)
            hand_patch_embeddings = self.embed_hand_patch(hand_patch_embeddings.float())  # (b, t, n_patch_latents, h)
        
        # Add timestep embeddings
        time_embeddings = self.embed_timestep.weight  # (sqe, h)
        lang_embeddings = lang_embeddings.view(batch_size, 1, -1) + time_embeddings
        state_embeddings = state_embeddings + time_embeddings
        if self.use_2d_traj:
            embed_traj_2d = embed_traj_2d + time_embeddings.view(sequence_length, 1, self.hidden_size)
        patch_embeddings = patch_embeddings + time_embeddings.view(sequence_length, 1, self.hidden_size)
        obs_embeddings = obs_embeddings + time_embeddings
        if self.use_hand_rgb:
            hand_obs_embeddings = hand_obs_embeddings + time_embeddings
            hand_patch_embeddings = hand_patch_embeddings + time_embeddings.view(sequence_length, 1, self.hidden_size)

        # Format sequence: lang, state, patch, obs,hand_patch, hand_obs, action_2d, [ACT], [OBS], [OBS_HAND]
        lang_embeddings = lang_embeddings.view(batch_size, sequence_length, 1, self.hidden_size)
        state_embeddings = state_embeddings.view(batch_size, sequence_length, 1, self.hidden_size)
        obs_embeddings = obs_embeddings.view(batch_size, sequence_length, 1, self.hidden_size)
        if self.use_2d_traj:
            stacked_inputs = torch.cat(
                (lang_embeddings, 
                state_embeddings, 
                embed_traj_2d,
                patch_embeddings, 
                obs_embeddings), dim=2)  # (b, t, n_tokens, h)
        else:
            stacked_inputs = torch.cat(
                    (lang_embeddings, 
                    state_embeddings, 
                    patch_embeddings, 
                    obs_embeddings), dim=2)  # (b, t, n_tokens, h)
        if self.use_hand_rgb:
            hand_obs_embeddings = hand_obs_embeddings.view(batch_size, sequence_length, 1, self.hidden_size)
            stacked_inputs = torch.cat(
                (stacked_inputs,
                 hand_patch_embeddings, 
                 hand_obs_embeddings), dim=2)  # (b, t, n_tokens, h)
        if self.act_2d_pred:
            action_2d_queries = self.action_2d_queries.weight  # (1, h)
            action_2d_queries = action_2d_queries.view(1, 1, self.pred_2d_points_num, self.hidden_size).repeat(batch_size, sequence_length, 1, 1)  # (b, t, pred_2d_points_num, h)
            stacked_inputs = torch.cat((stacked_inputs, action_2d_queries), dim=2)  # (b, t, n_tokens, h)
        if self.act_pred:
            action_queries = self.action_queries.weight  # (1, h)
            action_chunk_queries = self.action_chunk_queries.weight + action_queries # (chunk_size, h)
            action_chunk_queries = action_chunk_queries.view(1, 1, self.chunk_size, self.hidden_size).repeat(batch_size, sequence_length, 1, 1)  # (b, t, chunk_size, h)
            stacked_inputs = torch.cat((stacked_inputs, action_chunk_queries), dim=2)  # (b, t, n_tokens, h)
        if self.fwd_pred:
            obs_queries = self.obs_queries.weight  # (n_patch_latents + 1, h)
            obs_queries = obs_queries.view(1, 1, self.n_patch_latents + 1, self.hidden_size).repeat(batch_size, sequence_length, 1, 1)  # (b, t, n_patch_latents + 1, h)
            stacked_inputs = torch.cat((stacked_inputs, obs_queries), dim=2)
            if self.fwd_pred_hand:
                obs_hand_queries = self.obs_hand_queries.weight # 10, h
                obs_hand_queries = obs_hand_queries.view(1, 1, self.n_patch_latents+1, self.hidden_size).repeat(batch_size, sequence_length, 1, 1)
                stacked_inputs = torch.cat((stacked_inputs, obs_hand_queries), dim=2)

        # Number of tokens
        n_lang_tokens = 1
        n_state_tokens = 1
        if self.use_2d_traj:
            n_traj_2d_tokens = self.traj_2d_tokens
            n_patch_2d_tokens = self.n_patch_latents_2d
        n_patch_tokens = self.n_patch_latents
        n_obs_tokens = 1
        n_hand_patch_tokens = self.n_patch_latents
        n_hand_obs_tokens = 1
        
        # 汇总所有可见token INPUT
        n_tokens = n_lang_tokens + n_state_tokens + n_patch_tokens + n_obs_tokens
        if self.use_2d_traj:
            n_tokens += n_traj_2d_tokens
            n_tokens += n_patch_2d_tokens
        if self.use_hand_rgb:
            n_tokens += n_hand_obs_tokens
            n_tokens += n_hand_patch_tokens
            
        # 汇总所有生成的token OUTPUT 并标记每个输出token的起始位置
        if self.act_2d_pred:
            n_act_2d_pred_tokens = self.pred_2d_points_num
            act_2d_query_token_start_i = n_tokens
            n_tokens += n_act_2d_pred_tokens
        n_act_pred_tokens = self.chunk_size
        if self.act_pred:
            act_query_token_start_i = n_tokens
            n_tokens += n_act_pred_tokens
        if self.fwd_pred:
            obs_query_token_start_i = n_tokens
            n_tokens += (n_patch_tokens + n_obs_tokens)
            if self.fwd_pred_hand:
                obs_hand_query_token_start_i = n_tokens
                n_tokens += (n_patch_tokens + n_obs_tokens) 

        # Layer norm
        stacked_inputs = stacked_inputs.reshape(batch_size, n_tokens * sequence_length, self.hidden_size)
        stacked_inputs = self.embed_ln(stacked_inputs)

        # Attention mask 把output屏蔽 input unmask
        stacked_attention_mask = attention_mask.view(batch_size, sequence_length, 1)
        unmasked_tokens = n_lang_tokens + n_state_tokens + n_patch_tokens + n_obs_tokens
        if self.use_2d_traj:
            unmasked_tokens += n_traj_2d_tokens
            unmasked_tokens += n_patch_2d_tokens
        if self.use_hand_rgb:
            unmasked_tokens += n_hand_obs_tokens
            unmasked_tokens += n_hand_patch_tokens
        stacked_attention_mask = stacked_attention_mask.repeat(
                1, 1, unmasked_tokens)
        
        # 需要mask的
        if self.act_2d_pred:
            act_2d_query_attention_mask = torch.zeros((batch_size, sequence_length, n_act_2d_pred_tokens), dtype=torch.long, device=stacked_inputs.device)
            stacked_attention_mask = torch.cat((stacked_attention_mask, act_2d_query_attention_mask), dim=2)
        if self.act_pred:
            act_query_attention_mask = torch.zeros((batch_size, sequence_length, n_act_pred_tokens), dtype=torch.long, device=stacked_inputs.device)
            stacked_attention_mask = torch.cat((stacked_attention_mask, act_query_attention_mask), dim=2)
        if self.fwd_pred:
            obs_query_attention_mask = torch.zeros((batch_size, sequence_length, n_patch_tokens + n_obs_tokens), dtype=torch.long, device=stacked_inputs.device)
            stacked_attention_mask = torch.cat((stacked_attention_mask, obs_query_attention_mask), dim=2)
            if self.fwd_pred_hand:
                obs_hand_query_attention_mask = torch.zeros((batch_size, sequence_length, n_patch_tokens + n_obs_tokens), dtype=torch.long, device=stacked_inputs.device)
                stacked_attention_mask = torch.cat((stacked_attention_mask, obs_hand_query_attention_mask), dim=2)
        stacked_attention_mask = stacked_attention_mask.reshape(batch_size, n_tokens * sequence_length)

        # GPT forward pass
        transformer_outputs = self.transformer(
            inputs_embeds=stacked_inputs,
            attention_mask=stacked_attention_mask,
        )
        # GR1 output
        x = transformer_outputs['last_hidden_state']
        x = x.reshape(batch_size, sequence_length, n_tokens, self.hidden_size)

        # 2d action prediction
        if self.act_2d_pred:
            action_2d_embedding = x[:, :, act_2d_query_token_start_i:(act_2d_query_token_start_i+self.pred_2d_points_num)]
            action_2d_embedding = action_2d_embedding + embed_traj_2d
            for pred_act_2d_mlp in self.pred_act_2d_mlps:
                action_2d_embedding = pred_act_2d_mlp(action_2d_embedding)
            traj_2d_preds = self.pred_2d_points(action_2d_embedding)

        # Action prediction
        if self.act_pred:
            action_embedding = x[:, :, act_query_token_start_i:(act_query_token_start_i+self.chunk_size)]
            for pred_act_mlp in self.pred_act_mlps:
                action_embedding = pred_act_mlp(action_embedding)
            arm_action_preds = self.pred_arm_act(action_embedding)  # (b, t, chunk_size, act_dim - 1)
            gripper_action_preds = self.pred_gripper_act(action_embedding)  # (b, t, chunk_size, 1)
        # Forward prediction vision decoder
        if self.fwd_pred:
            mask_token = self.mask_token  # (1, 1, 1, h)
            mask_tokens = mask_token.repeat(batch_size, sequence_length, (self.image_size//self.patch_size)**2, 1)  # (b, l, n_patches, h)
            mask_tokens = mask_tokens + self.decoder_pos_embed.unsqueeze(0).repeat(batch_size, sequence_length, 1, 1)  # (b, l, n_patches, h)

            obs_pred = self.decoder_embed(x[:, :, obs_query_token_start_i:(obs_query_token_start_i + n_patch_tokens + n_obs_tokens)])  # (b, l, n_patch_latents + 1, h)
            obs_pred_ = torch.cat([obs_pred, mask_tokens], dim=2)  # (b, l, n_patches + n_patch_latens + 1, h)
            obs_pred_ = obs_pred_.reshape(-1, obs_pred_.shape[-2], obs_pred_.shape[-1])  # (b * l, n_patches + n_patch_latens + 1, h)
            for blk in self.decoder_blocks:
                obs_pred_ = blk(obs_pred_)
            obs_pred_ = self.decoder_norm(obs_pred_)
            obs_preds = self.decoder_pred(obs_pred_)  # (b * l, n_patches + n_patch_latens + 1, h)
            obs_preds = obs_preds.reshape(batch_size, sequence_length, -1, obs_preds.shape[-1])  # (b, l, n_patches + n_patch_latens + 1, h)
            obs_preds = obs_preds[:, :, (n_patch_tokens+n_obs_tokens):]  # (b, l, n_patches, h)

            if self.fwd_pred_hand:
                obs_pred_hand = self.decoder_embed(x[:, :, obs_hand_query_token_start_i:(obs_hand_query_token_start_i + n_patch_tokens + n_obs_tokens)])
                obs_pred_hand_ = torch.cat([obs_pred_hand, mask_tokens], dim=2)
                obs_pred_hand_ = obs_pred_hand_.reshape(-1, obs_pred_hand_.shape[-2], obs_pred_hand_.shape[-1])
                for blk in self.decoder_blocks:
                    obs_pred_hand_ = blk(obs_pred_hand_)
                obs_pred_hand_ = self.decoder_norm(obs_pred_hand_)
                obs_hand_preds = self.decoder_pred(obs_pred_hand_)
                obs_hand_preds = obs_hand_preds.reshape(batch_size, sequence_length, -1, obs_hand_preds.shape[-1])
                obs_hand_preds = obs_hand_preds[:, :, (n_patch_tokens+n_obs_tokens):]
        
        prediction = {
            'obs_preds': obs_preds,
            'obs_targets': obs_targets,
            'masked_2d_patch': masked_patch, # 基于2d点的patch索引
            'obs_hand_preds': obs_hand_preds,
            'obs_hand_targets': obs_hand_targets,
            'arm_action_preds': arm_action_preds,
            'gripper_action_preds': gripper_action_preds,
            'traj_2d_preds': traj_2d_preds
        }
        return prediction
