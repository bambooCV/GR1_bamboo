# MIT License

# Copyright (c) 2021 Oier Mees
# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Code to evaluate Calvin."""
import argparse
import json
import logging
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
# os.environ['CUDA_VISIBLE_DEVICES'] = '2,3,4,5'
# os.environ['CUDA_VISIBLE_DEVICES'] = '4,5,6,7'
# os.environ['CUDA_VISIBLE_DEVICES'] = '6,7,8,9'
# os.environ['CUDA_VISIBLE_DEVICES'] = '5'
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
from pathlib import Path
import sys
import time
import copy
from moviepy.editor import ImageSequenceClip
from accelerate import Accelerator
from datetime import timedelta
from accelerate.utils import InitProcessGroupKwargs
from traj_predict.model.TrajPredictPolicy import TrajPredictPolicy
# This is for using the locally installed repo clone when using slurm
from calvin_agent.models.calvin_base_model import CalvinBaseModel

sys.path.insert(0, Path(__file__).absolute().parents[2].as_posix())

from calvin_agent.evaluation.multistep_sequences import get_sequences_saved,get_sequences
from calvin_agent.evaluation.utils import (
    count_success,
    get_env_state_for_initial_condition,
    get_log_dir,
)
import hydra
import numpy as np
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from termcolor import colored
import torch
from tqdm.auto import tqdm

from evaluation.calvin_evaluation_traj import GR1CalvinEvaluation

from utils.calvin_utils import print_and_save,print_and_save_json
import clip
from PreProcess import PreProcess
import models.vision_transformer as vits 

# from models.gr1_2d_prompt import GR1
from models.gr1_2d_prompt_splitquery import GR1

import cv2
logger = logging.getLogger(__name__)
cvshow_flag = False
os.environ["FFMPEG_BINARY"] = "auto-detect"
os.environ['CALVIN_ROOT'] = "/gpfsdata/home/shichao/EmbodiedAI/manipulation/calvin"
CALVIN_ROOT = os.environ['CALVIN_ROOT']

def make_env(dataset_path, observation_space, device):
    val_folder = Path(dataset_path) / "validation"
    from evaluation.calvin_env_wrapper_raw import CalvinEnvWrapperRaw
    env = CalvinEnvWrapperRaw(val_folder, observation_space, device)
    return env


def evaluate_policy(model, env, eval_sr_path, eval_result_path, ep_len, num_sequences, num_procs, procs_id, eval_dir=None, debug=False, json_loaded=True):
    conf_dir = Path(f"{CALVIN_ROOT}/calvin_models") / "conf"
    task_cfg = OmegaConf.load(conf_dir / "callbacks/rollout/tasks/new_playtable_tasks.yaml")
    task_oracle = hydra.utils.instantiate(task_cfg)
    val_annotations = OmegaConf.load(conf_dir / "annotations/new_playtable_validation.yaml") # language instruction template：turn_off_led: ["press the button to turn off the led light"]
    eval_dir = get_log_dir(eval_dir)
    if json_loaded:
        if debug == True:
            eval_sequences = get_sequences_saved(num_sequences,filename="eval_episode_fail_case.json")
        else:
            eval_sequences = get_sequences_saved(num_sequences,filename="eval_episode_1000.json")
        # eval_sequences = get_sequences_saved(num_sequences,filename="eval_episode_fail_case.json")
    else:
        eval_sequences = get_sequences(num_sequences)
    num_seq_per_procs = num_sequences // num_procs
    eval_sequences = eval_sequences[num_seq_per_procs*procs_id:num_seq_per_procs*(procs_id+1)]

    results = []
    if not debug:
        eval_sequences = tqdm(eval_sequences, position=0, leave=True)

    sequence_i = 0
    if json_loaded:
        for item in eval_sequences:
            initial_state = item["state"]
            eval_sequence = item["result"]
            result = evaluate_sequence(env, model, task_oracle, initial_state, eval_sequence, val_annotations, debug, eval_dir, sequence_i, ep_len)
            results.append(result)
            if not debug:
                success_list = count_success(results)
                eval_sr_path_rank = eval_sr_path + f"_{procs_id}"
                with open(eval_sr_path_rank, 'a') as f:
                    line =f"{procs_id} id {sequence_i}/{num_sequences}: "
                    for sr in success_list:
                        line += f"{sr:.3f} | "
                    sequence_i += 1
                    line += "\n"
                    f.write(line)
                eval_sequences.set_description(
                    f"{procs_id} id "+ " ".join([f"{i + 1}/5 : {v * 100:.1f}% |" for i, v in enumerate(success_list)]) + "|"
                )
            else:
                sequence_i += 1

        path_parts = eval_result_path.rsplit('.', 1)
        base_path = path_parts[0]
        extension = path_parts[1]
        eval_result_path = f"{base_path}_{procs_id}.{extension}"
        print_and_save_json(results, eval_sequences, eval_result_path, None)
    else:
        for initial_state, eval_sequence in eval_sequences:

            result = evaluate_sequence(env, model, task_oracle, initial_state, eval_sequence, val_annotations, debug, eval_dir, sequence_i, ep_len)
            results.append(result)
            if not debug:
                success_list = count_success(results)
                with open(eval_sr_path, 'a') as f:
                    line =f"{sequence_i}/{num_sequences}: "
                    for sr in success_list:
                        line += f"{sr:.3f} | "
                    sequence_i += 1
                    line += "\n"
                    f.write(line)
                eval_sequences.set_description(
                    " ".join([f"{i + 1}/5 : {v * 100:.1f}% |" for i, v in enumerate(success_list)]) + "|"
                )
            else:
                sequence_i += 1
        print_and_save(results, eval_sequences, eval_result_path, None)
    return results


def evaluate_sequence(env, model, task_checker, initial_state, eval_sequence, val_annotations, debug, eval_dir, sequence_i, ep_len):
    robot_obs, scene_obs = get_env_state_for_initial_condition(initial_state)

    env.reset(robot_obs=robot_obs, scene_obs=scene_obs)

    success_counter = 0
    if debug:
        time.sleep(1)
        print()
        print()
        print(f"Evaluating sequence: {' -> '.join(eval_sequence)}")
        print("Subtask: ", end="")
    for subtask_i, subtask in enumerate(eval_sequence):
        success = rollout(env, model, task_checker, subtask, val_annotations, debug, eval_dir, subtask_i, sequence_i, ep_len)
        if success:
            success_counter += 1
        else:
            return success_counter

    return success_counter


def rollout(env, model, task_oracle, subtask, val_annotations, debug, eval_dir, subtask_i, sequence_i, ep_len):
    if debug:
        print(f"{subtask} ", end="")
        time.sleep(0.5)
    obs = env.get_obs()
    lang_annotation = val_annotations[subtask][0]
    
    model.reset()

    start_info = env.get_info()
    if debug:
        img_list = []
    unfinished = 0
    diff_flag = False
    draw_flag = False
    for step in range(ep_len):
        if unfinished == 0:
            action,re_out_action,traj_2d_pred = model.step(obs, lang_annotation,step,diff_flag,debug,draw_flag)
            if diff_flag:
                diff_flag = False
            if draw_flag:
                draw_flag = False

            unfinished = action.shape[0]
        obs, _, _, current_info = env.step(action[-unfinished])
        unfinished -= 1
        if debug:

            # inference traj inference
            img_copy = copy.deepcopy(obs['rgb_obs']['rgb_static'])
            img_copy_vis = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)
            point_2d_resized = re_out_action * 200/224
            # print(point_2d_resized)
            # print(action)
            for point_2d in point_2d_resized :
                cv2.circle(img_copy_vis, tuple(point_2d.int().tolist()), radius=3, color=(0, 0, 255), thickness=-1)
                cv2.circle(img_copy, tuple(point_2d.int().tolist()), radius=3, color=(255, 0, 0), thickness=-1)
            point_2d_pred = traj_2d_pred * 200
            # print(point_2d_pred)
            for point_2d in point_2d_pred :
                cv2.circle(img_copy_vis, tuple(point_2d.int().tolist()), radius=3, color=(255, 0, 0), thickness=-1)
                cv2.circle(img_copy, tuple(point_2d.int().tolist()), radius=3, color=(0, 0, 255), thickness=-1)
            if cvshow_flag:
                cv2.imshow("Pred Image Final", img_copy_vis)          
                while True:
                    key = cv2.waitKey(0) & 0xFF  # 获取按键
                    if key == ord('q'):  # 检查是否按下 'q' 键
                        step = 360
                    if key == ord('r') and step > 10:
                        diff_flag = True
                    if key == ord('d'): 
                        draw_flag = True
                    break
                if step == 360:
                    break
            img_list.append(img_copy)
        # check if current step solves a task
        current_task_info = task_oracle.get_task_info_for_set(start_info, current_info, {subtask})
        if len(current_task_info) > 0:
            if debug:
                print(colored("success", "green"), end=" ")
            current_frame = step
            while step < current_frame + 10:
                # 多走10帧
                if unfinished == 0:
                    action,re_out_action,traj_2d_pred = model.step(obs, lang_annotation,step,diff_flag,debug)
                    if diff_flag:
                        diff_flag = False
                    unfinished = action.shape[0]
                obs, _, _, current_info = env.step(action[-unfinished])
                unfinished -= 1
                step = step + 1
                if debug:
                    # inference traj inference
                    img_copy = copy.deepcopy(obs['rgb_obs']['rgb_static'])
                    img_copy_vis = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)
                    point_2d_resized = re_out_action * 200/224
                    # print(point_2d_resized)
                    # print(action)
                    for point_2d in point_2d_resized :
                        cv2.circle(img_copy_vis, tuple(point_2d.int().tolist()), radius=3, color=(0, 0, 255), thickness=-1)
                        cv2.circle(img_copy, tuple(point_2d.int().tolist()), radius=3, color=(255, 0, 0), thickness=-1)
                    point_2d_pred = traj_2d_pred * 200
                    # print(point_2d_pred)
                    for point_2d in point_2d_pred :
                        cv2.circle(img_copy_vis, tuple(point_2d.int().tolist()), radius=3, color=(255, 0, 0), thickness=-1)
                        cv2.circle(img_copy, tuple(point_2d.int().tolist()), radius=3, color=(0, 0, 255), thickness=-1)
                    if cvshow_flag:
                        cv2.imshow("Pred Image Final", img_copy_vis)          
                        cv2.waitKey(0) 
                    img_list.append(img_copy)
            if debug:
                # print(colored("success", "green"), end=" ")
                clip = ImageSequenceClip(img_list, fps=30)
                clip.write_gif(os.path.join(eval_dir, f'{sequence_i}-{subtask_i}-{subtask}-succ.gif'), fps=30)
                # print(obs["robot_obs"])
            return True
    if debug:
        if len(img_list) > 10:
            print(colored("fail", "red"), end=" ")
            clip = ImageSequenceClip(img_list, fps=30)
            clip.write_gif(os.path.join(eval_dir, f'{sequence_i}-{subtask_i}-{subtask}-fail.gif'), fps=30)
    return False


def main():
    # Preparation
    cfg = json.load(open('task_ABC_D_configs_eval_2dTraj_fail_case.json'))
    # The timeout here is 36000s to wait for other processes to finish the simulation
    kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=360000))
    acc = Accelerator(mixed_precision="bf16",kwargs_handlers=[kwargs])
    device = acc.device
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
    if cfg['fwd_pred'] and cfg['fwd_pred_hand']:
        training_target = ['act_pred', 'fwd_pred', 'fwd_pred_hand']
    else:
        training_target = ['act_pred']
    model = GR1(
        model_clip,
        model_mae,
        state_dim=cfg['state_dim'],
        act_dim=cfg['act_dim'],
        hidden_size=cfg['embed_dim'],
        sequence_length=cfg['seq_len'],
        chunk_size=cfg['chunk_size'],
        training_target=training_target,
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
        use_2d_traj=True,
        n_layer=cfg['n_layer'],
        n_head=cfg['n_head'],
        n_inner=4*cfg['embed_dim'],
        activation_function=cfg['activation_function'],
        n_positions=cfg['n_positions'],
        resid_pdrop=cfg['dropout'],
        attn_pdrop=cfg['dropout'],
    ).to(device)  # for fused optimizer
    model_traj = TrajPredictPolicy(model_mae,model_clip).to(device)

    
    # 预训练模型读入
    if cfg['load_bytedance_ckpt']:
        model.load_state_dict(torch.load(cfg['bytedance_ckpt_path'],map_location=device)['state_dict'], strict=False)
        acc.print('load ', cfg['bytedance_ckpt_path'] )
    elif os.path.isfile(cfg['save_path']+'GR1_{}.pth'.format(cfg['load_epoch'])):
        state_dict = torch.load(cfg['save_path']+'GR1_{}.pth'.format(cfg['load_epoch']),map_location=device)['state_dict'] 
        model.load_state_dict(state_dict, strict=False)
        acc.print('load ', cfg['save_path']+'GR1_{}.pth'.format(cfg['load_epoch']))
    if cfg['compile_model']:
        model = torch.compile(model)

    # 预训练模型读入
    # model_path_traj = "Save/diffusion_2D_trajectory/update4_with_pad10/ddp_task_ABC_D_best_checkpoint_103_e85.pth"
    model_path_traj = "Save/diffusion_2D_trajectory/update6_done/ddp_task_ABC_D_best_checkpoint_120_e57.pth"
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
        

    model,model_traj = acc.prepare(model, model_traj,device_placement=[True,True])
    observation_space = {
        'rgb_obs': ['rgb_static', 'rgb_gripper'], 
        'depth_obs': [], 
        'state_obs': ['robot_obs'], 
        'actions': ['rel_actions'], 
        'language': ['language']}
    eval_dir = cfg['save_path']+f'eval{torch.cuda.current_device()}/'
    os.makedirs(eval_dir, exist_ok=True)
    env = make_env('./fake_dataset', observation_space, device)
    
    eva = GR1CalvinEvaluation(model, model_traj,cfg, preprocessor, device)
    model.eval()
    model_traj.eval()

    
    avg_reward = torch.tensor(evaluate_policy(
        eva, 
        env,
        cfg['save_path']+'success_rate_e{}.txt'.format(cfg['load_epoch']), 
        cfg['save_path']+'result_e{}.txt'.format(cfg['load_epoch']), 
        cfg['ep_len'],
        cfg['num_sequences'],
        acc.num_processes,
        acc.process_index,
        eval_dir,
        debug=cfg['record_evaluation_video'],
    )).float().mean().to(device)
    acc.wait_for_everyone()
    avg_reward = acc.gather_for_metrics(avg_reward).mean()
    if acc.is_main_process:
        print('average success rate ', avg_reward)

if __name__ == "__main__":
    main()
