import os
os.environ['CUDA_VISIBLE_DEVICES'] = '4'
import sys
import torch,clip
import torch.nn as nn
from traj_predict.model.transformer_for_diffusion import TransformerForDiffusion 
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
import models.vision_transformer as vits
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TrajPredictPolicy(nn.Module):
    def __init__(
        self,
        model_clip,
        model_mae,
        
    ):
        super().__init__()
        
        # CLIP for language encoding
        self.model_clip = model_clip
        for _, param in self.model_clip.named_parameters():
            param.requires_grad = False
                # MAE for image encoding
        self.model_mae = model_mae
        for _, param in self.model_mae.named_parameters():
            param.requires_grad = False
            
# vision encoders model
model_mae = vits.__dict__['vit_base'](patch_size=16, num_classes=0).to(device)
checkpoint_vit = torch.load("/gpfsdata/home/shichao/EmbodiedAI/manipulation/PretrainModel_Download/vit/mae_pretrain_vit_base.pth")
model_mae.load_state_dict(checkpoint_vit['model'], strict=False)
# language encoders model
model_clip, _ = clip.load("ViT-B/32", device=device) 
# predict noise model 
action_dim = 2 # x,y
action_horizon = 10
patch_size = 16
noise_pred_net =  TransformerForDiffusion(
        input_dim=action_dim,
        output_dim=action_dim,
        horizon=action_horizon,
        n_obs_steps=1*patch_size,
        cond_dim=512,
        causal_attn=True,
        # time_as_cond=False,
        # n_cond_layers=4
    )
noise_pred_net = noise_pred_net.to(device)

# policy network
num_diffusion_iters = 100
noise_scheduler = DDPMScheduler(
    num_train_timesteps=num_diffusion_iters,
    # the choise of beta schedule has big impact on performance
    # we found squared cosine works the best
    beta_schedule='squaredcos_cap_v2',
    # clip output to [-1,1] to improve stability
    clip_sample=True,
    # our network predicts noise (instead of denoised action)
    prediction_type='epsilon'
)
batchsize = 4


image_features = torch.randn(batchsize, patch_size, 512).to(device) # batch,channel,height,width
language_features = torch.randn(batchsize, 1,512).to(device) # batch,feature_dim
naction = torch.randn(batchsize, action_horizon, 2).to(device) # batch,action_seq,action_dim(x,y)

timesteps = torch.randint(
    0, noise_scheduler.config.num_train_timesteps,
    (batchsize,), device=device
).long()
# sample noise to add to actions
noise = torch.randn(naction.shape, device=device)
noisy_actions = noise_scheduler.add_noise(
    naction, noise, timesteps) # sample
# concatenate vision feature and language obs
obs_features = torch.cat([image_features, language_features], dim=1)
obs_cond = obs_features
# predict the noise residual
noise_pred = noise_pred_net(
    noisy_actions, timesteps, obs_cond)

print("fsc test")

