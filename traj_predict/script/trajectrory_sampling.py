import sys
sys.path.insert(0, '/gpfsdata/home/shichao/EmbodiedAI/manipulation/calvin')
import os
import io
import argparse
import lmdb
from pickle import dumps, loads
import numpy as np
import torch
from torchvision.transforms.functional import resize
from torchvision.io import encode_jpeg
import clip
from einops import rearrange, repeat
import matplotlib.pyplot as plt
import cv2
import pickle
from calvin_env.calvin_env.camera.static_camera import StaticCamera
def worldtopixel_point(trajectory):
    point_2dprojection = []
    look_at = [ -0.026242351159453392, -0.0302329882979393, 0.3920000493526459]
    look_from =  [ 2.871459009488717, -2.166602199425597, 2.555159848480571]
    up_vector = [ 0.4041403970338857, 0.22629790978217404, 0.8862616969685161]
    staticamera_instance = StaticCamera(fov=10,aspect=1,nearval=0.01,farval=10,width=200,height=200,look_at=look_at,look_from=look_from,up_vector = up_vector,cid = None,name = None)
    for point in trajectory:
        w_point = point[0:3] + [1]
        p_point = staticamera_instance.project(w_point)
        p_point = tuple(int(x) for x in p_point) + (int(point[-1]),)
        point_2dprojection.append(p_point) 
    return point_2dprojection

def trajectory_visiable(traj,ann,traj_save_path):
    x = [point[0] for point in traj]
    y = [point[1] for point in traj]
    z = [point[2] for point in traj]
    # Plotting the 3D trajectory
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot
    ax.scatter(x, y, z,color='b')
    
    # start catch stop
    ax.scatter(x[0], y[0], z[0], c='g', label='Start', s=100)  # Start point in green
    ax.scatter(x[-1], y[-1], z[-1], c='r', label='End', s=100)  # End point in red


    # Line plot
    ax.plot(x, y, z, color='r')

    # Labels
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_zlabel('Z Coordinate')

    # Title
    ax.set_title(ann)

    # Save the plot
    plt.savefig(traj_save_path)  
    plt.close(fig)  # Close the figure window
    # plt.show()



def trajectory_visiable_combine(traj,ds_traj,ann,traj_save_path):
    x = [point[0] for point in traj]
    y = [point[1] for point in traj]
    z = [point[2] for point in traj]

    ds_x = [point[0] for point in ds_traj]
    ds_y = [point[1] for point in ds_traj]
    ds_z = [point[2] for point in ds_traj]
    # Plotting the 3D trajectory
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot
    ax.scatter(x[1:-1], y[1:-1], z[1:-1],color='b')
    
    # start catch stop
    ax.scatter(ds_x[1:-1], ds_y[1:-1], ds_z[1:-1],color='b',label= 'griper',s=100) # gripper
    ax.scatter(x[0], y[0], z[0], c='g', label='Start', s=100)  # Start point in green
    ax.scatter(x[-1], y[-1], z[-1], c='r', label='End', s=100)  # End point in red


    # Line plot
    ax.plot(x, y, z, color='r')
    # Line plot downsampling
    ax.plot(ds_x,ds_y,ds_z,color = 'y')

    # Labels
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_zlabel('Z Coordinate')

    # Title
    ax.set_title(ann)

    # Save the plot
    plt.savefig(traj_save_path)  
    plt.close(fig)  # Close the figure window
    # plt.show()

def trajectory_visiable_downsample(traj):
    # traj 优化,取收尾2个点，外加夹爪反转的时刻点
    # 起始点
    downsampled_trajectory = [traj[0]]
    for i in range(1, len(traj)):
        # Check if the last value (gripper action) has changed direction
        if traj[i][-1] * traj[i - 1][-1] < 0:
            # If direction changed, add this point to the downsampled trajectory
            downsampled_trajectory.append(traj[i])

    # Add the last point if it's not already in the list
    if downsampled_trajectory[-1] != traj[-1]:
        downsampled_trajectory.append(traj[-1])

    return downsampled_trajectory

def save_to_lmdb(output_dir, input_dir):
    env = lmdb.open(output_dir, map_size=int(3e12), readonly=False, lock=False) # maximum size of memory map is 3TB
    annotations = np.load(os.path.join(input_dir, 'lang_annotations/auto_lang_ann.npy'), allow_pickle=True).tolist()['language']['ann'] # 文字描述
    start_end_ids = np.load(os.path.join(input_dir, 'lang_annotations/auto_lang_ann.npy'), allow_pickle=True).tolist()['info']['indx'] # 每一集的开始结尾
    with env.begin(write=True) as txn:
        if txn.get('cur_step'.encode()) is not None:
            cur_step = loads(txn.get('cur_step'.encode())) + 1
            cur_episode = loads(txn.get(f'cur_episode_{cur_step - 1}'.encode())) + 1
        else:
            cur_step = 0
            cur_episode = 0

        for index, (start, end) in enumerate(start_end_ids):
            print(f'{index/len(start_end_ids)}')
            inst = annotations[index]
            txn.put(f'inst_{cur_episode}'.encode(), dumps(inst))
            with torch.no_grad():
                inst_token = clip.tokenize(inst)
                inst_emb = model_clip.encode_text(inst_token.cuda()).cpu()
            txn.put(f'inst_token_{cur_episode}'.encode(), dumps(inst_token[0]))
            txn.put(f'inst_emb_{cur_episode}'.encode(), dumps(inst_emb[0]))
            for i in range(start, end+1):
                frame = np.load(os.path.join(input_dir, f'episode_{i:07}.npz'))
                txn.put('cur_step'.encode(), dumps(cur_step))
                txn.put(f'cur_episode_{cur_step}'.encode(), dumps(cur_episode))
                txn.put(f'done_{cur_step}'.encode(), dumps(False))
                rgb_static = torch.from_numpy(rearrange(frame['rgb_static'], 'h w c -> c h w'))
                txn.put(f'rgb_static_{cur_step}'.encode(), dumps(encode_jpeg(rgb_static)))
                rgb_gripper = torch.from_numpy(rearrange(frame['rgb_gripper'], 'h w c -> c h w'))
                txn.put(f'rgb_gripper_{cur_step}'.encode(), dumps(encode_jpeg(rgb_gripper)))
                txn.put(f'abs_action_{cur_step}'.encode(), dumps(torch.from_numpy(frame['actions'])))
                txn.put(f'rel_action_{cur_step}'.encode(), dumps(torch.from_numpy(frame['rel_actions'])))
                txn.put(f'scene_obs_{cur_step}'.encode(), dumps(torch.from_numpy(frame['scene_obs'])))
                txn.put(f'robot_obs_{cur_step}'.encode(), dumps(torch.from_numpy(frame['robot_obs'])))
                # 加入剩余轨迹
                traj_index = i
                traj = []
                while traj_index >= start and traj_index <= end:
                    t_file_traj = np.load(os.path.join(input_dir, f'episode_{traj_index:07}.npz'))
                    traj.append(t_file_traj['actions.npy'].tolist())
                    traj_index = traj_index + 1 
                ds_traj = traj# trajectory_visiable_downsample(traj)
                ds_2d_traj = worldtopixel_point(ds_traj)
                traj_2d_tensor = torch.tensor(ds_2d_traj)
                txn.put(f'traj_2d_{cur_step}'.encode(), pickle.dumps(traj_2d_tensor))
                if i == start:
                    # 起始点
                    txn.put(f'traj_2d_init_{cur_episode}'.encode(), pickle.dumps(traj_2d_tensor))

                # visualization 轨迹
                # rgb_static_rgb = cv2.cvtColor(rgb_static.permute(1, 2, 0).numpy(), cv2.COLOR_BGR2RGB)
                # for point_2d in ds_2d_traj:
                #     cv2.circle(rgb_static_rgb, point_2d[:-1], radius=3, color=(0, 255, 255), thickness=-1)
                # cv2.imshow('Processed RGB Static Image', rgb_static_rgb)  # 注意这里需要调整回 HWC 格式
                # cv2.waitKey(0)
                cur_step += 1
            txn.put(f'done_{cur_step-1}'.encode(), dumps(True))
            cur_episode += 1
    env.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Transfer CALVIN dataset to lmdb format.")
    parser.add_argument("--input_dir", default='/home/DATASET_PUBLIC/calvin/calvin_debug_dataset/', type=str, help="Original dataset directory.")
    parser.add_argument("--output_dir", default='/home/DATASET_PUBLIC/calvin/calvin_debug_dataset/calvin_lmdb', type=str, help="Original dataset directory.")
    args = parser.parse_args()
    model_clip, _ = clip.load('ViT-B/32', device='cuda:0')
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    save_to_lmdb(args.output_dir, os.path.join(args.input_dir, 'training'))
    save_to_lmdb(args.output_dir, os.path.join(args.input_dir, 'validation'))