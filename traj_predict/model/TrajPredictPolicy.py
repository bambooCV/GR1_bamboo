

import torch,clip
import torch.nn as nn
from traj_predict.model.transformer_for_diffusion import TransformerForDiffusion 
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
import models.vision_transformer as vits
from torchvision.transforms.v2 import Resize, RandomResizedCrop 
from PIL import Image


class Traj_PreProcess(): 
    def __init__(
            self,
            rgb_shape, 
            rgb_mean, 
            rgb_std, 
            crop_area_scale,
            crop_aspect_ratio,
            device,
        ):
        self.train_transforms = RandomResizedCrop(rgb_shape, crop_area_scale, crop_aspect_ratio, interpolation=Image.BICUBIC, antialias=True).to(device)
        self.test_transforms = Resize(rgb_shape, interpolation=Image.BICUBIC, antialias=True).to(device)
        self.rgb_mean = torch.tensor(rgb_mean, device=device).view(1, 1, -1, 1, 1)
        self.rgb_std = torch.tensor(rgb_std, device=device).view(1, 1, -1, 1, 1)
    
    def rgb_process(self, rgb_static, train=False):
        batch_size = rgb_static.shape[0]
        original_shape = rgb_static.shape[-2:]  # (height, width) 
        rgb_static = rgb_static.float()*(1/255.)
        crop_boxes = []
        for idx in range(batch_size):
            if train:
                i, j, h, w = self.train_transforms.get_params(rgb_static[idx], self.train_transforms.scale, self.train_transforms.ratio)
                crop_boxes.append([j, i, w, h])
                rgb_static_reshape = self.train_transforms(rgb_static)  
            else:
                rgb_static_reshape = self.test_transforms(rgb_static)
                crop_boxes.append([0, 0, original_shape[1], original_shape[0]])
        crop_boxes = torch.tensor(crop_boxes).unsqueeze(1)
        # torchvision Normalize forces sync between CPU and GPU, so we use our own
        rgb_static_norm = (rgb_static_reshape - self.rgb_mean) / (self.rgb_std + 1e-6)
        return rgb_static_norm,rgb_static_reshape,crop_boxes
class TrajPredictPolicy(nn.Module):
    def __init__(
        self,
        
    ):
        super().__init__()

        # vision encoders model
        model_mae = vits.__dict__['vit_base'](patch_size=16, num_classes=0)
        checkpoint_vit = torch.load("/gpfsdata/home/shichao/EmbodiedAI/manipulation/PretrainModel_Download/vit/mae_pretrain_vit_base.pth")
        model_mae.load_state_dict(checkpoint_vit['model'], strict=False)
        # language encoders model
        model_clip, _ = clip.load("ViT-B/32",device="cpu") 
        # CLIP for language encoding
        self.model_clip = model_clip
        for _, param in self.model_clip.named_parameters():
            param.requires_grad = False
        # MAE for image encoding
        self.model_mae = model_mae
        for _, param in self.model_mae.named_parameters():
            param.requires_grad = False
            
            
        self.hidden_size = 512
        self.img_feat_dim = 768
        # project functions for images
        self.proj_static_img = torch.nn.Linear(self.img_feat_dim, self.hidden_size)
        
        # predict noise model 
        self.action_dim = 2 # x,y
        self.action_horizon = 30
        self.patch_size = 14
        self.noise_pred_net =  TransformerForDiffusion(
                input_dim=self.action_dim ,
                output_dim=self.action_dim ,
                horizon=self.action_horizon,
                n_obs_steps=1*self.patch_size**2,
                cond_dim=512,
                causal_attn=True,
                # time_as_cond=False,
                # n_cond_layers=4
            )
              
    def forward(self, 
        rgb_static_norm,
        language,
        timesteps,
        noisy_actions,
        language_embedding = None,
        obs_embeddings = None,
        patch_embeddings = None
        ):
        # model input prepare: noisy_actions, timesteps, obs_cond
        # image batch*seq, channel, height, width
        batch_size, sequence, channel, height, width = rgb_static_norm.shape
        rgb_static_norm = rgb_static_norm.view(batch_size*sequence, channel, height, width)

        if language_embedding is None and obs_embeddings is None and patch_embeddings is None:
            with torch.no_grad():
                language_embedding = self.model_clip.encode_text(language).unsqueeze(1)
                obs_embeddings, patch_embeddings = self.model_mae(rgb_static_norm)
            patch_embeddings = self.proj_static_img(patch_embeddings)

        
        # concatenate vision feature and language obs
        obs_features = torch.cat([patch_embeddings, language_embedding], dim=1)
        obs_cond = obs_features
        
        noise_pred = self.noise_pred_net(noisy_actions, timesteps, obs_cond)
        
        return noise_pred,language_embedding,obs_embeddings,patch_embeddings