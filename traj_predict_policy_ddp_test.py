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
import wandb
from traj_predict.traj_func import PreProcess
os.environ["WANDB_API_KEY"] = 'KEY'
os.environ["WANDB_MODE"] = "offline"
def contains_words(inst, include_words=[], exclude_words=[]):
    for word in include_words:
        if word not in inst:
            return False
    for word in exclude_words:
        if word in inst:
            return False
    return True
def save_checkpoint(epoch, model, optimizer,  loss,save_dir="./Save"):
    save_path = os.path.join(save_dir, f'test_checkpoint.pth')
    # 获取未冻结的参数
    # model_state_dict = {k: v for k, v in model.state_dict().items() if v.requires_grad}
    torch.save({
        'epoch': epoch,
        'model_state_dict':model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }, save_path)
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
def pre_processing(rgb_static_norm, language, model_clip, model_mae,train=False):
        
    batch_size, sequence, channel, height, width = rgb_static_norm.shape
    rgb_static_norm = rgb_static_norm.view(batch_size*sequence, channel, height, width)
    language_embedding = model_clip.encode_text(language).unsqueeze(1)
    obs_embeddings, patch_embeddings = model_mae(rgb_static_norm)
    return language_embedding, obs_embeddings, patch_embeddings

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
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    acc = Accelerator(
        kwargs_handlers=[ddp_kwargs]
    )
    wandb_model = False
    if wandb_model and acc.is_main_process:
        wandb.init(project='robotic traj diffusion task_ABC_D arguement', group='robotic traj diffusion', name='DDP traj diffusion_ABC_D_0806')
    device = acc.device
    # config prepare
    batch_size = 64
    num_workers = 4
    # lmdb_dir = "/home/DATASET_PUBLIC/calvin/calvin_debug_dataset/calvin_lmdb"
    # lmdb_dir = "/home/DATASET_PUBLIC/calvin/task_D_D/calvin_lmdb"
    lmdb_dir = "/home/DATASET_PUBLIC/calvin/task_ABC_D/calvin_lmdb"
    #image preprocess
    preprocessor = PreProcess(
        rgb_static_pad = 10,
        rgb_gripper_pad = 4,
        rgb_shape = [224,224], 
        rgb_mean = [0.485, 0.456, 0.406],
        rgb_std =  [0.229, 0.224, 0.225],
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

 
    model = TrajPredictPolicy()
    # 预训练模型读入
    # model_path = "Save/diffusion_2D_trajectory/ddp_task_ABC_D_best_checkpoint_epoch72.pth"
    # model_path = "Save/ddp_task_ABC_D_best_checkpoint_0.pth"
    # model.load_state_dict(torch.load(model_path)['model_state_dict'],strict=False)
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
    # training config
    epoch_num = 100

    optimizer = torch.optim.AdamW(
            params=model.parameters(),
            lr=1e-4, weight_decay=1e-6)
    model, optimizer, train_loader, val_loader = acc.prepare(
        model, 
        optimizer, 
        train_loader, 
        val_loader, 
        device_placement=[True, True, False, False], # 前两个参数为模型参数放入GPU，最后两个参数为数据参数，放入CPU
    )
    train_prefetcher = DataPrefetcher(train_loader, device)
    val_prefetcher = DataPrefetcher(val_loader, device)
    lr_scheduler = get_scheduler(
            name='cosine',
            optimizer=optimizer,
            num_warmup_steps=5,
            num_training_steps=len(train_loader) * epoch_num
        )


    best_loss = float('inf')
    for epoch in tqdm(range(epoch_num), desc="Epochs"):
        total_loss = 0
        val_total_loss = 0
        # training
        # Only show progress bar on main process
        with tqdm(total=len(train_loader), desc=f"Train Epoch {epoch+1}", leave=False,disable= not acc.is_main_process) as pbar:
            batch, load_time = train_prefetcher.next()
            while batch is not None:
                model.train()
                # batch data for training  image_features.language_features.naction
                language = batch['inst_token']
                image = batch['rgb_static']
                naction = batch['actions']
                
                rgb_static_norm,rgb_gripper_norm,naction_transformed = preprocessor.rgb_process(batch['rgb_static'], batch["rgb_gripper"],batch['actions'],train=True)    
                naction_trans_norm = normalize_data(naction_transformed)
                # visualization croped image
                # Convert tensor to NumPy array for visualization
                import cv2
                for batch_idx in range(image.shape[0]):
                    for seq_idx in range(image.shape[1]):
                        rgb_static_rgb = cv2.cvtColor(image[batch_idx][seq_idx].permute(1, 2, 0).cpu().numpy(), cv2.COLOR_BGR2RGB)
                        for point_2d in naction[batch_idx,seq_idx,:,:]:
                            cv2.circle(rgb_static_rgb, tuple(point_2d.int().tolist()), radius=3, color=(0, 0, 255), thickness=-1)
                        cv2.imshow('Ori RGB Static Image', rgb_static_rgb)  # 注意这里需要调整回 HWC 格式
                        
                        rgb_static_reshape = preprocessor.rgb_recovery(rgb_static_norm)
                        rgb_static_np = cv2.cvtColor(rgb_static_reshape[batch_idx][seq_idx].permute(1, 2, 0).cpu().numpy(), cv2.COLOR_BGR2RGB)

                        for point_2d in naction_transformed[batch_idx,seq_idx,:,:]:
                            cv2.circle(rgb_static_np, tuple(point_2d.int().tolist()), radius=3, color=(0, 0, 255), thickness=-1)
                        cv2.imshow("Cropped Image", rgb_static_np)
                        cv2.waitKey(0)
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps,
                    (naction_trans_norm.shape[0],), device=device).long()
                noise = torch.randn(naction_trans_norm.shape, device=device)
                noisy_actions = noise_scheduler.add_noise(naction_trans_norm, noise, timesteps) 
                b,s,chunk_size,dim = noisy_actions.shape
                noisy_actions = noisy_actions.reshape(b*s,chunk_size,dim)
                
                noise_pred,_,_,_ = model(rgb_static_norm, language, timesteps, noisy_actions)
                # loss = nn.functional.mse_loss(noise_pred, noise.squeeze(1)) 原始loss
                mask = batch['mask'].unsqueeze(-1).expand_as(noise_pred) # 先广播到同样大小
                masked_loss = nn.functional.mse_loss(noise_pred * mask, noise.squeeze(1) * mask, reduction='none')
                # 难样本loss增强
                hard_weights = torch.ones_like(noise_pred,device=device)
                condition = torch.tensor([("light bulb" in inst) for inst in batch['inst']], device=noise_pred.device).view(-1, 1, 1).expand_as(noise_pred)
                hard_weights[condition] = 2.0
                masked_loss = masked_loss * hard_weights
                # 计算有效掩码的总和
                mask_sum = mask.sum()
                if mask_sum > 0:
                    loss = masked_loss.sum() / mask_sum
                else:
                    loss = masked_loss.sum()  # 如果掩码全为0，则直接求和
                # optimize
                if wandb_model and acc.is_main_process:
                    wandb.log({'loss': loss})
                # loss.backward()
                acc.backward(loss)
            
                optimizer.step()
                optimizer.zero_grad()
                # step lr scheduler every batch
                # this is different from standard pytorch behavior
                lr_scheduler.step()
                # logging
                total_loss += loss.item()
                pbar.set_postfix(
                    ordered_dict={
                        "epoch": epoch,
                        "loss": loss.item(),
                        "lr": optimizer.state_dict()["param_groups"][0]["lr"],
                    }
                )
                pbar.update(1)
                batch, load_time = train_prefetcher.next()
            avg_train_loss = total_loss/len(train_loader)
            if wandb_model and acc.is_main_process:
                wandb.log({'avg_train_loss': avg_train_loss})
            if avg_train_loss < best_loss:
                best_loss = avg_train_loss
                save_checkpoint(epoch, model, optimizer, best_loss)
            if acc.is_main_process:

                print(f'Epoch {epoch+1}/{epoch_num}, Train Average Loss: {avg_train_loss:.4f})')
            
        # evaluation
        with tqdm(total=len(val_loader), desc=f"Val Epoch {epoch+1}", leave=False,disable= not acc.is_main_process) as pbar:
            with torch.no_grad():
                batch, load_time = val_prefetcher.next()
                val_index = 0
                # 算 light bulb
                while batch is not None and val_index < 20:
                    eval_flag = False
                    colors = ["pink", "blue", "red"]
                    directions = ["right", "left"]
                    exclude_words = ["rotate","turn"]
                    include_conditions = [(color, direction) for color in colors for direction in directions]
                    for inst in batch['inst']:
                        # if "lightbulb" in inst or "light bulb" in inst:
                        if any(contains_words(inst, include_words=cond, exclude_words=exclude_words) for cond in include_conditions) or \
                           "lightbulb" in inst or "light bulb" in inst :
                            eval_flag = True

                    if eval_flag:
                        model.eval()
                        language = batch['inst_token']
                        image = batch['rgb_static']
                        naction = batch['actions']
                        # example inputs
                        rgb_static_norm,rgb_gripper_norm,naction_transformed = preprocessor.rgb_process(batch['rgb_static'], batch["rgb_gripper"],batch['actions'],train=False)    
                        naction_trans_norm = normalize_data(naction_transformed)
                        noisy_action = torch.randn(naction.shape, device=device)
                        batch_val_size,sequence,chunk_size,dim = noisy_action.shape
                        noisy_action = noisy_action.reshape(batch_val_size*sequence,chunk_size,dim)
                        out_action = noisy_action
                        # init scheduler
                        noise_scheduler.set_timesteps(num_diffusion_iters)
                        language_embedding, obs_embeddings, patch_embeddings = None, None, None
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
                        re_out_action = unnormalize_data(out_action)
                        val_sample_loss =nn.functional.mse_loss(out_action, naction_trans_norm.squeeze(1))
                        if wandb_model and acc.is_main_process:
                            wandb.log({'val_sample_loss': val_sample_loss})
                                        # logging
                        val_total_loss += val_sample_loss.item()
                        pbar.set_postfix(
                            ordered_dict={
                                "epoch": epoch,
                                "val_loss": val_sample_loss.item(),
                            }
                        )
                        pbar.update(1)
                        batch, load_time = val_prefetcher.next()
                        val_index = val_index + 1
                    else:
                        pbar.set_postfix(
                            ordered_dict={
                                "epoch": epoch,
                                "val_loss": 0,
                            }
                        )
                        pbar.update(1)
                        batch, load_time = val_prefetcher.next()                       
                if val_index == 0:
                    avg_val_loss = 9999
                else:
                    avg_val_loss = val_total_loss/val_index
                if wandb_model and acc.is_main_process:
                    wandb.log({'avg_val_loss': avg_val_loss})
            if acc.is_main_process:
                print(f'Epoch {epoch+1}/{epoch_num}, Val Average Loss: {avg_val_loss:.4f})')
        
       




