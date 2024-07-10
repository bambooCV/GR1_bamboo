import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import sys
import torch,clip
import torch.nn as nn
from traj_predict.model.transformer_for_diffusion import TransformerForDiffusion 
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
import models.vision_transformer as vits
from traj_predict.traj_diffusion_data_loader import LMDBDataset,DataPrefetcher
from torch.utils.data import DataLoader
from torchvision.transforms.v2 import Resize, RandomResizedCrop 
from PIL import Image
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, InitProcessGroupKwargs
from datetime import timedelta
from transformers import get_scheduler
from traj_predict.model.TrajPredictPolicy import TrajPredictPolicy
from PreProcess import PreProcess
os.environ["WANDB_API_KEY"] = 'KEY'
os.environ["WANDB_MODE"] = "offline"


if __name__ == '__main__':
    import json
    from models.gr1 import GR1 
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # Preparation
    cfg = json.load(open('configs_eval_test.json'))
    preprocessor = PreProcess(
        cfg['rgb_static_pad'],
        cfg['rgb_gripper_pad'],
        cfg['rgb_shape'],
        cfg['rgb_mean'],
        cfg['rgb_std'],
        device,
    )
    model_clip, _ = clip.load(cfg['clip_backbone'], device=device) 
    model_mae = vits.__dict__['vit_base'](patch_size=16, num_classes=0).to(device)
    checkpoint = torch.load(cfg['mae_ckpt'])
    model_mae.load_state_dict(checkpoint['model'], strict=False)
    # 预训练模型读入
    model_traj = TrajPredictPolicy()
    model_path_traj = "Save/ddp_task_ABC_D_best_checkpoint_epoch72.pth"
    state_dict_traj = torch.load(model_path_traj,map_location=device)['model_state_dict']
    new_state_dict = {}
    for key, value in state_dict_traj.items():
        new_key = key.replace('module.', '')
        new_state_dict[new_key] = value
        multi_gpu = True
    if multi_gpu:
        model_traj.load_state_dict(new_state_dict,strict=False)
    else:
        model_traj.load_state_dict(state_dict_traj,strict=False)

    model_traj.to(device)
    model = GR1(
        model_clip,
        model_mae,
        state_dim=cfg['state_dim'],
        act_dim=cfg['act_dim'],
        hidden_size=cfg['embed_dim'],
        sequence_length=cfg['seq_len'],
        chunk_size=cfg['chunk_size'],
        training_target=['act_pred', 'fwd_pred', 'fwd_pred_hand'],
        img_feat_dim=cfg['img_feat_dim'],
        patch_feat_dim=cfg['patch_feat_dim'],
        lang_feat_dim=cfg['lang_feat_dim'],
        resampler_params={
            'depth': cfg['resampler_depth'],
            'dim_head': cfg['resampler_dim_head'],
            'heads': cfg['resampler_heads'],
            'num_latents': cfg['resampler_num_latents'],
            'num_media_embeds': cfg['resampler_num_media_embeds'],
        },
        without_norm_pixel_loss=False,
        use_hand_rgb=True,
        n_layer=cfg['n_layer'],
        n_head=cfg['n_head'],
        n_inner=4*cfg['embed_dim'],
        activation_function=cfg['activation_function'],
        n_positions=cfg['n_positions'],
        resid_pdrop=cfg['dropout'],
        attn_pdrop=cfg['dropout'],
    ).to(device)  # for fused optimizer
    if cfg['load_bytedance_ckpt']:
        model.load_state_dict(torch.load(cfg['bytedance_ckpt_path'])['state_dict'], strict=False)
   
    if cfg['compile_model']:
        model = torch.compile(model)




    
    print("fsc test")
        
       




