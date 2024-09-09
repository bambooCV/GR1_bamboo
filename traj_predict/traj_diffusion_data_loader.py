import io
import gc
from time import time
import lmdb
from pickle import loads
import numpy as np
import torch
from torchvision import transforms
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.io import decode_jpeg
from torch.utils.data import DataLoader
import random
import cv2 
ORIGINAL_STATIC_RES = 200
ORIGINAL_GRIPPER_RES = 84

def contains_words(inst, include_words=[], exclude_words=[]):
    for word in include_words:
        if word not in inst:
            return False
    for word in exclude_words:
        if word in inst:
            return False
    return True
def add_noise(sequence, noise_level=2.0):
    noise = torch.randn(sequence.shape) * noise_level
    noisy_sequence = sequence + noise
    return noisy_sequence
def resample_sequence(sequence, target_length):
    """
    使用插值将 sequence 重新采样到 target_length，并将结果四舍五入为整数
    :param sequence: 原始序列，形状为 [N, 2]
    :param target_length: 目标序列长度
    :return: 重新采样后的序列，形状为 [target_length, 2]
    """
    # 确保 sequence 是 float 类型
    sequence = sequence.float()
    sequence = sequence.unsqueeze(0).permute(0, 2, 1)  # 调整形状为 (1, 2, N)
    resampled_sequence = F.interpolate(sequence, size=target_length, mode='linear', align_corners=True)
    resampled_sequence = resampled_sequence.permute(0, 2, 1).squeeze(0)  # 调整回原始形状 (target_length, 2)
    resampled_sequence = add_noise(resampled_sequence, noise_level=0.75)
    # 将结果四舍五入为整数
    resampled_sequence = torch.round(resampled_sequence).int()
    
    return resampled_sequence
    
def visulization_image(rgb_static,actions,inst):
  
    rgb_static_rgb = cv2.cvtColor(rgb_static.permute(1, 2, 0).numpy(), cv2.COLOR_BGR2RGB)
    # for point_2d in future_2d_actions[:,:2]:
    #     cv2.circle(rgb_static_rgb, tuple(point_2d.int().tolist()), radius=3, color=(0, 255, 255), thickness=-1)
    for point_2d in actions:
        cv2.circle(rgb_static_rgb, tuple(point_2d.int().tolist()), radius=3, color=(0, 0, 255), thickness=-1)
    # 把inst的文字放在图片左下角 放在左下角！

    cv2.putText(rgb_static_rgb, inst, (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (0, 0, 0), 1)
    cv2.imshow('Processed RGB Static Image', rgb_static_rgb)  # 注意这里需要调整回 HWC 格式


class DataPrefetcher():
    def __init__(self, loader, device):
        self.device = device
        self.loader = loader
        self.iter = iter(self.loader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            # Dataloader will prefetch data to cpu so this step is very quick
            self.batch = next(self.iter)
        except StopIteration:
            self.batch = None
            self.iter = iter(self.loader)
            return 
        with torch.cuda.stream(self.stream):
            for key in self.batch:
                if isinstance(self.batch[key], torch.Tensor):
                    self.batch[key] = self.batch[key].to(self.device, non_blocking=True)

    def next(self):
        clock = time()
        batch = self.batch
        if batch is not None:
            for key in batch:
                if batch[key] is not None:
                    if isinstance(batch[key], torch.Tensor):
                        batch[key].record_stream(torch.cuda.current_stream())
        self.preload()
        return batch, time()-clock

    def next_without_none(self):
        batch, time = self.next()
        if batch is None:
            batch, time = self.next()
        return batch, time

class LMDBDataset(Dataset):
    def __init__(self, lmdb_dir, sequence_length, chunk_size,action_dim, start_ratio, end_ratio):
        super(LMDBDataset).__init__()
        self.sequence_length = sequence_length
        self.chunk_size = chunk_size
        self.action_dim = action_dim
        self.dummy_rgb_static = torch.zeros(sequence_length, 3, ORIGINAL_STATIC_RES, ORIGINAL_STATIC_RES, dtype=torch.uint8)
        self.dummy_rgb_gripper = torch.zeros(sequence_length, 3, ORIGINAL_GRIPPER_RES, ORIGINAL_GRIPPER_RES, dtype=torch.uint8)
        self.dummy_arm_state = torch.zeros(sequence_length, 6)
        self.dummy_gripper_state =  torch.zeros(sequence_length, 2)
        self.dummy_actions = torch.zeros(sequence_length,chunk_size, action_dim)
        self.dummy_mask = torch.zeros(sequence_length)
        self.lmdb_dir = lmdb_dir
        self.left_num = 0
        self.right_num = 0
 
        env = lmdb.open(lmdb_dir, readonly=True, create=False, lock=False)
        with env.begin() as txn:
            dataset_len = loads(txn.get('cur_step'.encode())) + 1
            self.start_step = int(dataset_len * start_ratio) 
            self.end_step = int(dataset_len * end_ratio) - sequence_length
        env.close()

    def open_lmdb(self):
        self.env = lmdb.open(self.lmdb_dir, readonly=True, create=False, lock=False)
        self.txn = self.env.begin()

    def __getitem__(self, idx):
        if hasattr(self, 'env') == 0:
            self.open_lmdb()

        idx = idx + self.start_step

        rgb_static = self.dummy_rgb_static.clone()
        rgb_gripper = self.dummy_rgb_gripper.clone()
        arm_state = self.dummy_arm_state.clone()
        gripper_state = self.dummy_gripper_state.clone()
        actions = self.dummy_actions.clone()
        mask = self.dummy_mask.clone()

        cur_episode = loads(self.txn.get(f'cur_episode_{idx}'.encode()))
        inst_token = loads(self.txn.get(f'inst_token_{cur_episode}'.encode()))
        inst = loads(self.txn.get(f'inst_{cur_episode}'.encode()))
        inst_emb = loads(self.txn.get(f'inst_emb_{cur_episode}'.encode()))
        for i in range(self.sequence_length):
            new_idx = idx + i
            if loads(self.txn.get(f'cur_episode_{new_idx}'.encode())) == cur_episode:
                mask[i] = 1
                rgb_static[i] = decode_jpeg(loads(self.txn.get(f'rgb_static_{new_idx}'.encode())))
                rgb_gripper[i] = decode_jpeg(loads(self.txn.get(f'rgb_gripper_{new_idx}'.encode())))
                robot_obs = loads(self.txn.get(f'robot_obs_{new_idx}'.encode()))
                arm_state[i, :6] = robot_obs[:6]
                gripper_state[i, ((robot_obs[-1] + 1) / 2).long()] = 1
                future_2d_actions = loads(self.txn.get(f'traj_2d_{new_idx}'.encode()))
                if len(future_2d_actions) < self.chunk_size/3:
                    if future_2d_actions[:, 0].max() - future_2d_actions[:, 0].min() < 10 and future_2d_actions[:, 1].max() - future_2d_actions[:, 1].min() < 10:
                        mask[i] = 0
                else:
                    if future_2d_actions[:, 0].max() - future_2d_actions[:, 0].min() < 6 and future_2d_actions[:, 1].max() - future_2d_actions[:, 1].min() < 6:
                        mask[i] = 0
                # 图像增强，针对move_slider_right/left做特殊处理
                if "slide the door" in inst or "sliding door" in inst:
                    if mask[i] == 1 and len(future_2d_actions) > 5:
                        # import cv2   
                        # rgb_static_rgb = cv2.cvtColor(rgb_static[i].permute(1, 2, 0).numpy(), cv2.COLOR_BGR2RGB)
                        # for point_2d in future_2d_actions[:,:2]:
                        #     cv2.circle(rgb_static_rgb, tuple(point_2d.int().tolist()), radius=3, color=(0, 255, 255), thickness=-1)
                        # # 把inst的文字放在图片左下角 放在左下角！
                        # cv2.putText(rgb_static_rgb, inst, (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (0, 0, 0), 1)
                        # cv2.imshow('before padding RGB Static Image', rgb_static_rgb)  # 注意这里需要调整回 HWC 格式
                        # 计算最后action 五个点的平均斜率
                        last_five_points = future_2d_actions[-5:, :]
                        deltas = last_five_points[1:, :] - last_five_points[:-1, :]# 计算连续点之间的差值
                        average_delta = deltas.float().mean(dim=0)# 计算平均斜率
                        num_padding_points = 5 # 生成5个额外点
                        padding_points = []

                        for padding_idx in range(1, num_padding_points + 1):
                            new_point = last_five_points[-1, :] + padding_idx * average_delta
                            # 合法性0-200
                            new_point = new_point.clamp(0, 200)
                            padding_points.append(new_point)
                        # 将生成的点与原始数据连接
                        padding_points = torch.stack(padding_points, dim=0).to(torch.int64)
                        future_2d_actions = torch.cat([future_2d_actions, padding_points], dim=0)
                        # if future_2d_actions[-1][0] > 500 or future_2d_actions[-1][0] < 0:
                        #     print("fsc test")

                actions[i,:,:] = resample_sequence(future_2d_actions[:,:2], self.chunk_size)
                # 图像增强 针对light bulb 做特殊处理 && led 做特殊处理
                # if "lightbulb" in inst or "light bulb" in inst or "switch" in inst or "yellow light" in inst or "yellow lamp" in inst

                if "lightbulb" in inst or "light bulb" in inst or "switch" in inst or "yellow light" in inst or "yellow lamp" in inst\
                    or "led" in inst:
                    if random.random() > 0.5:
                        rgb_static[i] = transforms.functional.hflip(rgb_static[i])
                        actions[i][:,0] = rgb_static[i].shape[-1] - actions[i][:,0]
                    # if "led" in inst:
                    # # visualization 轨迹
                    #     visulization_image(rgb_static[i],actions[i],inst)
                    #     cv2.waitKey(0)
                # 图像增强 针对push right left 做特殊处理
                colors = ["pink", "blue", "red"]
                directions = ["right", "left"]
                exclude_words = ["rotate","turn"]
                include_conditions = [(color, direction) for color in colors for direction in directions]
                if any(contains_words(inst, include_words=cond, exclude_words=exclude_words) for cond in include_conditions):
                    if random.random() > 0.5:
                        rgb_static[i] = transforms.functional.hflip(rgb_static[i])
                        actions[i][:,0] = rgb_static[i].shape[-1] - actions[i][:,0]
                        # 更新方向
                        if "left" in inst:
                            inst = inst.replace("left", "right")
                        elif "right" in inst:
                            inst = inst.replace("right", "left")
                    # 二次剔除垃圾样本： 首位点计算距离，作为方向
                    if mask[i] == 1 and "right" in inst :
                        if  actions[i][-1][0] - actions[i][0][0] < 0:
                            mask[i] = 0 
                    if mask[i] == 1 and "left" in inst :
                        if  actions[i][-1][0] - actions[i][0][0] > 0:
                            mask[i] = 0

                # if mask[i] == 1:
                #     # visualization 轨迹
                #     visulization_image(rgb_static[i],actions[i],inst)
                #     cv2.waitKey(0)
                
        return {
            'rgb_static': rgb_static,
            'rgb_gripper': rgb_gripper,
            'inst':inst,
            'inst_token': inst_token,
            'inst_emb': inst_emb,
            'arm_state': arm_state,
            'gripper_state': gripper_state,
            'actions': actions,
            'mask': mask,
        }

    def __len__(self):
        return self.end_step - self.start_step

if __name__ == '__main__':
    from traj_func import PreProcess
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 96
    num_workers = 1
    preprocessor = PreProcess(
        rgb_static_pad = 10, # 去除位置敏感性
        rgb_gripper_pad = 4,
        rgb_shape = [224,224], 
        rgb_mean = [0.485, 0.456, 0.406],
        rgb_std =  [0.229, 0.224, 0.225],
        device = device
    )
    train_dataset = LMDBDataset(
        # lmdb_dir = "/home/DATASET_PUBLIC/calvin/task_D_D/calvin_lmdb_V1", 
        # lmdb_dir = "/home/DATASET_PUBLIC/calvin/task_ABC_D/calvin_lmdb_V1/",
        lmdb_dir = "/home/DATASET_PUBLIC/calvin/task_ABCD_D/calvin_lmdb_V1/",
        sequence_length = 1, 
        chunk_size = 30,# 最长不超过65
        action_dim = 2, # x,y,gripper_state
        start_ratio = 0,
        end_ratio = 0.09, 
    )
    val_dataset = LMDBDataset(
        # lmdb_dir = "/home/DATASET_PUBLIC/calvin/task_D_D/calvin_lmdb_V1", 
        # lmdb_dir = "/home/DATASET_PUBLIC/calvin/task_ABC_D/calvin_lmdb_V1/",
        lmdb_dir = "/home/DATASET_PUBLIC/calvin/task_ABCD_D/calvin_lmdb_V1/",
        sequence_length = 1, 
        chunk_size = 30,# 最长不超过65
        action_dim = 2,
        start_ratio = 0.99,
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
    test_prefetcher = DataPrefetcher(val_loader, device)

    from tqdm import tqdm
    for epoch in range(1):
        with tqdm(total=len(train_loader), desc=f"Train Epoch {epoch+1}", leave=False) as pbar:
            batch, load_time = train_prefetcher.next()
            while batch is not None:
                batch, load_time = train_prefetcher.next()
                image = batch['rgb_static']
                naction = batch['actions']
                rgb_static_norm,rgb_gripper_norm,naction_transformed = preprocessor.rgb_process(batch['rgb_static'], batch["rgb_gripper"],batch['actions'],train=False)
                # visualization croped image
                # Convert tensor to NumPy array for visualization
                import cv2
                for batch_idx in range(image.shape[0]):
                    for seq_idx in range(image.shape[1]):
                        rgb_static_rgb = cv2.cvtColor(image[batch_idx][seq_idx].permute(1, 2, 0).cpu().numpy(), cv2.COLOR_BGR2RGB)
                        # for point_2d in naction[batch_idx,seq_idx,:,:]:
                        #     cv2.circle(rgb_static_rgb, tuple(point_2d.int().tolist()), radius=3, color=(0, 0, 255), thickness=-1)
                        # cv2.putText(rgb_static_rgb, batch["inst"][batch_idx], (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (0, 0, 0), 1)
                        cv2.imshow('Ori RGB Static Image', rgb_static_rgb)  # 注意这里需要调整回 HWC 格式
                        
                        rgb_static_reshape = preprocessor.rgb_recovery(rgb_static_norm)
                        rgb_static_np = cv2.cvtColor(rgb_static_reshape[batch_idx][seq_idx].permute(1, 2, 0).cpu().numpy(), cv2.COLOR_BGR2RGB)

                        for point_2d in naction_transformed[batch_idx,seq_idx,:,:]:
                            cv2.circle(rgb_static_np, tuple(point_2d.int().tolist()), radius=3, color=(0, 0, 255), thickness=-1)
                        cv2.imshow("Cropped Image", rgb_static_np)
                        cv2.waitKey(0)
                pbar.update(1) 
        