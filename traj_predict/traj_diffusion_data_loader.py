import io
import gc
from time import time
import lmdb
from pickle import loads
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.io import decode_jpeg
from torch.utils.data import DataLoader
ORIGINAL_STATIC_RES = 200
ORIGINAL_GRIPPER_RES = 84
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
    
    # 将结果四舍五入为整数
    resampled_sequence = torch.round(resampled_sequence).int()
    
    return resampled_sequence
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
                self.batch[key] = self.batch[key].to(self.device, non_blocking=True)

    def next(self):
        clock = time()
        batch = self.batch
        if batch is not None:
            for key in batch:
                if batch[key] is not None:
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
 

                actions[i,:,:] = resample_sequence(future_2d_actions[:,:2], self.chunk_size)
                
                # 可视化 action ori 和 downsample的
                # visualization 轨迹
                # import cv2
                # rgb_static_rgb = cv2.cvtColor(rgb_static[i].permute(1, 2, 0).numpy(), cv2.COLOR_BGR2RGB)
                # for point_2d in future_2d_actions[:,:2]:
                #     cv2.circle(rgb_static_rgb, tuple(point_2d.tolist()), radius=3, color=(0, 255, 255), thickness=-1)
                # for point_2d in actions[i,:,:]:
                #     cv2.circle(rgb_static_rgb, tuple(point_2d.int().tolist()), radius=3, color=(0, 0, 255), thickness=-1)
                # # 把inst的文字放在图片左下角 放在左下角！
                
                # cv2.putText(rgb_static_rgb, inst, (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (0, 0, 0), 1)
                # cv2.imshow('Processed RGB Static Image', rgb_static_rgb)  # 注意这里需要调整回 HWC 格式
                # cv2.waitKey(0)


        return {
            'rgb_static': rgb_static,
            'rgb_gripper': rgb_gripper,
            'inst_token': inst_token,
            'arm_state': arm_state,
            'gripper_state': gripper_state,
            'actions': actions,
            'mask': mask,
        }

    def __len__(self):
        return self.end_step - self.start_step
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 8
    num_workers = 1
    train_dataset = LMDBDataset(
        lmdb_dir = "/home/DATASET_PUBLIC/calvin/calvin_debug_dataset/calvin_lmdb", 
        sequence_length = 1, 
        chunk_size = 30,# 最长不超过65
        action_dim = 2, # x,y,gripper_state
        start_ratio = 0,
        end_ratio = 0.9, 
    )
    val_dataset = LMDBDataset(
        lmdb_dir = "/home/DATASET_PUBLIC/calvin/calvin_debug_dataset/calvin_lmdb", 
        sequence_length = 1, 
        chunk_size = 30,# 最长不超过65
        action_dim = 2,
        start_ratio = 0,
        end_ratio = 0.9, 
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
        shuffle=True,
        prefetch_factor=2,
        persistent_workers=True,
    ) 
    train_prefetcher = DataPrefetcher(train_loader, device)
    test_prefetcher = DataPrefetcher(val_loader, device)
    for epoch in range(10):
        batch, load_time = train_prefetcher.next()
        while batch is not None:
            
            print(batch)
            batch, load_time = train_prefetcher.next()
        