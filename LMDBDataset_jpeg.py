import io
import gc
from time import time
import lmdb
from pickle import loads
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.io import decode_jpeg
import torch.nn.functional as F
from torch.utils.data import DataLoader
from traj_predict.traj_func import PreProcess
ORIGINAL_STATIC_RES = 200
ORIGINAL_GRIPPER_RES = 84
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
    def __init__(self, lmdb_dir, sequence_length, chunk_size, action_mode, action_dim, start_ratio, end_ratio):
        super(LMDBDataset).__init__()
        self.sequence_length = sequence_length
        self.chunk_size = chunk_size
        self.action_mode = action_mode
        self.action_dim = action_dim
        self.dummy_rgb_static = torch.zeros(sequence_length, 3, ORIGINAL_STATIC_RES, ORIGINAL_STATIC_RES, dtype=torch.uint8)
        self.dummy_rgb_gripper = torch.zeros(sequence_length, 3, ORIGINAL_GRIPPER_RES, ORIGINAL_GRIPPER_RES, dtype=torch.uint8)
        self.dummy_arm_state = torch.zeros(sequence_length, 6)
        self.dummy_gripper_state =  torch.zeros(sequence_length, 2)
        self.dummy_actions = torch.zeros(sequence_length, chunk_size, action_dim)
        self.dummy_actions_2d = torch.zeros(sequence_length,30, 2)
        self.dummy_traj_2d_preds = torch.zeros(sequence_length,100, 2) # 最多100个2d点
        self.dummy_mask = torch.zeros(sequence_length, chunk_size)
        self.lmdb_dir = lmdb_dir
        env = lmdb.open(lmdb_dir, readonly=True, create=False, lock=False)
        with env.begin() as txn:
            dataset_len = loads(txn.get('cur_step'.encode())) + 1
            self.start_step = int(dataset_len * start_ratio) 
            self.end_step = int(dataset_len * end_ratio) - sequence_length - chunk_size
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
        actions_2d = self.dummy_actions_2d.clone()
        traj_2d_preds =self.dummy_traj_2d_preds.clone()

        cur_episode = loads(self.txn.get(f'cur_episode_{idx}'.encode()))
        inst_token = loads(self.txn.get(f'inst_token_{cur_episode}'.encode()))
        for i in range(self.sequence_length):
            if loads(self.txn.get(f'cur_episode_{idx+i}'.encode())) == cur_episode:
                rgb_static[i] = decode_jpeg(loads(self.txn.get(f'rgb_static_{idx+i}'.encode())))
                rgb_gripper[i] = decode_jpeg(loads(self.txn.get(f'rgb_gripper_{idx+i}'.encode())))
                robot_obs = loads(self.txn.get(f'robot_obs_{idx+i}'.encode()))
                arm_state[i, :6] = robot_obs[:6]
                gripper_state[i, ((robot_obs[-1] + 1) / 2).long()] = 1
                for j in range(self.chunk_size):
                    if loads(self.txn.get(f'cur_episode_{idx+i+j}'.encode())) == cur_episode:
                        mask[i, j] = 1
                        if self.action_mode == 'ee_rel_pose':
                            actions[i, j] = loads(self.txn.get(f'rel_action_{idx+i+j}'.encode()))
                        elif self.action_mode == 'ee_abs_pose':
                            actions[i, j] = loads(self.txn.get(f'abs_action_{idx+i+j}'.encode()))
                        actions[i, j, -1] = (actions[i, j, -1] + 1) / 2
                traj_2d_preds_len =len(loads(self.txn.get(f'traj_2d_{idx+i}'.encode()))[:,:2])
                traj_2d_preds[i,:traj_2d_preds_len] = loads(self.txn.get(f'traj_2d_{idx+i}'.encode()))[:,:2]
                traj_2d_preds[i,traj_2d_preds_len:] = traj_2d_preds[i,traj_2d_preds_len-1] # 补齐长度
                future_2d_actions = loads(self.txn.get(f'traj_2d_init_{cur_episode}'.encode())) # 只用当前episode的2d action 为了和inference保持一致 且快速验证
                actions_2d[i] = resample_sequence(future_2d_actions[:,:2], 30)
                
        return {
            'rgb_static': rgb_static,
            'rgb_gripper': rgb_gripper,
            'inst_token': inst_token,
            'arm_state': arm_state,
            'gripper_state': gripper_state,
            'actions': actions,
            'mask': mask,
            'actions_2d': actions_2d,
            'traj_2d_preds': traj_2d_preds,
        }

    def __len__(self):
        return self.end_step - self.start_step
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    import json
     # Preparation
    cfg = json.load(open('configs_2dTraj_test.json'))
    preprocessor = PreProcess(
        cfg['rgb_static_pad'],
        cfg['rgb_gripper_pad'],
        cfg['rgb_shape'],
        cfg['rgb_mean'],
        cfg['rgb_std'],
        device,
    )
    train_dataset = LMDBDataset(
        cfg['LMDB_path'], 
        cfg['seq_len'], 
        cfg['chunk_size'], 
        cfg['action_mode'],
        cfg['act_dim'],
        start_ratio = 0,
        end_ratio = 0.9, 
    )
    test_dataset = LMDBDataset(
        cfg['LMDB_path'], 
        cfg['seq_len'], 
        cfg['chunk_size'], 
        cfg['action_mode'],
        cfg['act_dim'],
        start_ratio = 0.9,
        end_ratio = 1, 
    )
    train_loader = DataLoader(
        train_dataset, 
        batch_size=cfg['bs_per_gpu'], # to be flattened in prefetcher  
        num_workers=cfg['workers_per_gpu'],
        pin_memory=True, # Accelerate data reading
        # shuffle=True,
        shuffle=False,
        prefetch_factor=cfg['prefetch_factor'],
        persistent_workers=True,
    ) 
    test_loader = DataLoader(
        test_dataset, 
        batch_size=cfg['bs_per_gpu'], # to be flattened in prefetcher  
        num_workers=cfg['workers_per_gpu'],
        pin_memory=True, # Accelerate data reading
        shuffle=True,
        prefetch_factor=cfg['prefetch_factor'],
        persistent_workers=True,
    ) 
    train_prefetcher = DataPrefetcher(train_loader, device)
    test_prefetcher = DataPrefetcher(test_loader, device)
    for epoch in range(10):
        batch, load_time = train_prefetcher.next()
        while batch is not None:
            action_2d = batch['actions_2d']
            rgb_static,rgb_gripper,actions_2d_transformed= preprocessor.rgb_process(batch['rgb_static'], batch["rgb_gripper"],batch['actions_2d'],train=True)     
            print(batch)
            batch, load_time = train_prefetcher.next()