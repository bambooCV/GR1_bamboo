import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import sys
import torch,clip
import torch.nn as nn
from traj_predict.model.transformer_for_diffusion import TransformerForDiffusion 
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
import models.vision_transformer as vits
from traj_predict.traj_diffusion_data_loader import LMDBDataset,DataPrefetcher,contains_words
from torch.utils.data import DataLoader
from torchvision.transforms.v2 import Resize, RandomResizedCrop 
from PIL import Image
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from transformers import get_scheduler
import wandb
from time import time
import matplotlib.pyplot as plt
os.environ["WANDB_API_KEY"] = 'KEY'
os.environ["WANDB_MODE"] = "offline"

def normalize_data(data, stats={'min': 0,'max': 224}):
    # nomalize to [0,1]
    ndata = (data - stats['min']) / (stats['max'] - stats['min'])
    # normalize to [-1, 1]
    ndata = ndata * 2 - 1
    return ndata
def unnormalize_data(ndata, stats={'min': 0,'max': 224}):
    ndata = (ndata + 1) / 2 # [-1, 1] -> [0, 1] 域
    data = ndata * (stats['max'] - stats['min']) + stats['min']
    return data
class PreProcess(): 
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
    
    
def transform_points(points, crop_box, transformed_image):
    transformed_points = []
    for batch_idx in range(points.shape[0]):
        for seq_idx in range(points.shape[1]):
            crop_x, crop_y, crop_w, crop_h = crop_box[batch_idx, seq_idx]
            scale_x = transformed_image.shape[-1] / crop_w
            scale_y = transformed_image.shape[-2] / crop_h
            transformed_points.append([(int((x - crop_x) * scale_x), int((y - crop_y) * scale_y)) for x, y in points[batch_idx, seq_idx]])
    transformed_points = torch.tensor(transformed_points).unsqueeze(1)
    return transformed_points

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


if __name__ == '__main__':
    # wandb输出
    wandb_model = False
    if wandb_model:
        wandb.init(project='robotic traj diffusion task_D_D', group='robotic traj diffusion', name='traj diffusion_0626normaction')
    # config prepare
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 1
    num_workers = 4
    # lmdb_dir = "/home/DATASET_PUBLIC/calvin/calvin_debug_dataset/calvin_lmdb"
    # lmdb_dir = "/home/DATASET_PUBLIC/calvin/task_D_D/calvin_lmdb"
    lmdb_dir = "/home/DATASET_PUBLIC/calvin/task_ABC_D/calvin_lmdb"
    #image preprocess
    preprocessor = PreProcess(
        rgb_shape = [224,224],
        rgb_mean = [0.485, 0.456, 0.406],
        rgb_std =  [0.229, 0.224, 0.225],
        crop_area_scale = [0.9, 1.0],
        crop_aspect_ratio =[1.0, 1.0],
        device = device
    )
    # data loader
    train_dataset = LMDBDataset(
        lmdb_dir = lmdb_dir, 
        sequence_length = 1, 
        chunk_size = 30,# 最长不超过30
        action_dim = 2, # x,y,gripper_state
        start_ratio = 0,
        end_ratio = 0.9, 
    )
    val_dataset = LMDBDataset(
        lmdb_dir = lmdb_dir, 
        sequence_length = 1, 
        chunk_size = 30,# 最长不超过30
        action_dim = 2,
        start_ratio = 0.95,
        end_ratio = 1, 
    )
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, # to be flattened in prefetcher  
        num_workers=num_workers,
        pin_memory=True, # Accelerate data reading
        shuffle=True,
        prefetch_factor=2,
        persistent_workers=True,
    ) 
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, # to be flattened in prefetcher  
        num_workers=num_workers,
        pin_memory=True, # Accelerate data reading
        shuffle=False,
        prefetch_factor=2,
        persistent_workers=True,
    ) 
    train_prefetcher = DataPrefetcher(train_loader, device)
    val_prefetcher = DataPrefetcher(val_loader, device)
             
    model = TrajPredictPolicy()
    # 预训练模型读入
    # model_path = "Save/diffusion_2D_trajectory/ddp_task_ABC_D_best_checkpoint_epoch72.pth"
    model_path = "Save/ddp_task_ABC_D_best_checkpoint_24.pth"
    state_dict = torch.load(model_path,map_location=device)['model_state_dict']
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace('module.', '')
        new_state_dict[new_key] = value
        multi_gpu = True
    if multi_gpu:
        model.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(state_dict)
    model = model.to(device)
    
    # policy config
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
    val_total_loss = 0
    val_index = 0
    colors = ["pink", "blue", "red"]
    directions = ["right", "left"]
    exclude_words = ["rotate"]
    include_conditions = [(color, direction) for color in colors for direction in directions]
    with tqdm(total=len(val_loader), desc=f"Val Epoch {72}", leave=False) as pbar:
            with torch.no_grad():
                batch, load_time = val_prefetcher.next()
                val_index = 0
                # 算 light bulb
                while batch is not None: # and val_index < 20:
                    eval_flag = False
                    for inst in batch['inst']:
                        if "lightbulb" in inst or "light bulb" in inst:
                        # if any(contains_words(inst, include_words=cond, exclude_words=exclude_words) for cond in include_conditions):
                            eval_flag = True

                    if eval_flag:
                        model.eval()
                        language = batch['inst_token']
                        language_embedding = batch['inst_emb']
                        image = batch['rgb_static']
                        naction = batch['actions']
                        # example inputs
                        rgb_static_norm,rgb_static_reshape,crop_boxes = preprocessor.rgb_process(batch['rgb_static'], train=False)  
                        naction_transformed = transform_points(naction, crop_boxes, rgb_static_reshape).to(device).to(torch.float32)   # diffusion label    
                        naction_trans_norm = normalize_data(naction_transformed)
                        noisy_action = torch.randn(naction.shape, device=device)
                        batch_val_size,sequence,chunk_size,dim = noisy_action.shape
                        noisy_action = noisy_action.reshape(batch_val_size*sequence,chunk_size,dim)
                        out_action = noisy_action
                        # init scheduler
                        noise_scheduler.set_timesteps(num_diffusion_iters)
                        language_embedding, obs_embeddings, patch_embeddings = None, None, None
                        # val_sample_loss_values = []
                        start_time = time()

                        for k in noise_scheduler.timesteps:
                            # predict noise
                            noise_pred, language_embedding, obs_embeddings, patch_embeddings = model(rgb_static_norm, language, timesteps=k, noisy_actions=out_action,
                                                language_embedding=language_embedding, obs_embeddings=obs_embeddings, patch_embeddings=patch_embeddings)
                            # inverse diffusion step (remove noise)
                            out_action = noise_scheduler.step(
                                model_output=noise_pred,
                                timestep=k,
                                sample=out_action
                            ).prev_sample
                            # val_sample_loss =nn.functional.mse_loss(out_action, naction_trans_norm.squeeze(1))
                            # val_sample_loss_values.append(val_sample_loss.cpu())
                        end_time = time()
                        execution_time = end_time - start_time

                        # print(f"Code block executed in {execution_time} seconds.")
                        # plt.figure(figsize=(10, 6))
                        # plt.plot(noise_scheduler.timesteps, val_sample_loss_values, marker='o', linestyle='-', color='b')
                        # plt.xlabel('Timestep (k)')
                        # plt.ylabel('Validation Sample Loss (MSE)')
                        # plt.title('Validation Sample Loss vs Timestep')
                        # plt.grid(True)
                        # plt.gca().invert_xaxis()  # 将横坐标从 100 到 0 反转显示
                        # plt.show()
                        
                        re_out_action = unnormalize_data(out_action)
                        val_sample_loss =nn.functional.mse_loss(out_action, naction_trans_norm.squeeze(1))
                        val_total_loss += val_sample_loss.item()
                        # visualization croped image
                        # ################Convert tensor to NumPy array for visualization
                        re_out_action = re_out_action.unsqueeze(1)
                        import cv2
                        for batch_idx in range(image.shape[0]):
                            for seq_idx in range(image.shape[1]):

                                rgb_static_np = cv2.cvtColor(rgb_static_reshape[batch_idx][seq_idx].squeeze().permute(1, 2, 0).cpu().numpy(), cv2.COLOR_BGR2RGB)
                                # for point_2d in naction_transformed[batch_idx,seq_idx,:,:]:
                                #     cv2.circle(rgb_static_np, tuple(point_2d.int().tolist()), radius=3, color=(0, 0, 255), thickness=-1)
                                rgb_static_np = (rgb_static_np * 255).astype(np.uint8)
                                # cv2.putText(rgb_static_np, batch['inst'][batch_idx], (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (0, 0, 0), 1)
                                cv2.imshow("Ori Cropped I mage", rgb_static_np)
                                
                                rgb_static_np2 = cv2.cvtColor(rgb_static_reshape[batch_idx][seq_idx].squeeze().permute(1, 2, 0).cpu().numpy(), cv2.COLOR_BGR2RGB)
                                for point_2d in re_out_action[batch_idx,seq_idx,:,:]:
                                    cv2.circle(rgb_static_np2, tuple(point_2d.int().tolist()), radius=3, color=(0, 0, 255), thickness=-1)
                                rgb_static_np2 = (rgb_static_np2 * 255).astype(np.uint8)
                                cv2.putText(rgb_static_np2, batch['inst'][batch_idx], (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (0, 0, 0), 1)
                                cv2.imshow("Pred Cropped I mage", rgb_static_np2)
                                
                                
                                cv2.waitKey(0)
                    batch, load_time = val_prefetcher.next()
                    val_index = val_index + 1
                    pbar.update(1)
            avg_val_loss = val_total_loss/val_index
            print(f"avg_val_loss: {avg_val_loss}")
       




