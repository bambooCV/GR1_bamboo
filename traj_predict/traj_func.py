import torch
from torchvision.transforms.v2 import Resize 
from PIL import Image
import torch.nn.functional as F
from torchvision.transforms.v2 import Resize

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
def resize_points(points, original_size, new_size):
    """
    根据图像的resize比例调整点坐标。
    
    参数:
    - points: 形状为 (b, seq, t, 2) 的点坐标张量
    - original_size: 原始图像的大小，(h, w)
    - new_size: 调整后的图像大小，(new_h, new_w)
    
    返回:
    - 调整后的点坐标张量，形状为 (b, seq, t, 2)
    """
    original_h, original_w = original_size
    new_h, new_w = new_size
    
    # 计算缩放比例
    scale_x = new_w / original_w
    scale_y = new_h / original_h
    
    # 调整点的坐标
    points[..., 0] = points[..., 0] * scale_x
    points[..., 1] = points[..., 1] * scale_y
    
    return points


def RandomShiftsAug(x, pad):
    x = x.float()
    b, t, c, h, w = x.size() # torch.Size([10, 10, 3, 200, 200])
    assert h == w
    x = x.view(b*t, c, h, w)  # reshape x to [B*T, C, H, W] torch.Size([100, 3, 200, 200])
    padding = tuple([pad] * 4) # (10,10,10,10)
    x = F.pad(x, padding, "replicate") # 上下左右各10 torch.Size([100, 3, 220, 220])
    h_pad, w_pad = h + 2*pad, w + 2*pad  #220，220 calculate the height and width after padding
    eps = 1.0 / (h_pad)  # 1/220
    arange = torch.linspace(-1.0 + eps, 1.0 - eps, h_pad, device=x.device, dtype=x.dtype)[:h] # 以步长eps 范围是[-1,1] 长度为220，而后取前200个点
    arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2) # [200,200,1]
    base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)#torch.Size([200, 200, 2])
    base_grid = base_grid.unsqueeze(0).repeat(b*t, 1, 1, 1)#torch.Size([100, 200, 200, 2]) (x,y)原始图像每个像素的位置，且位置呗标准化到[-1，1]

    shift = torch.randint(0, 2 * pad + 1, size=(b, 1, 1, 1, 2), device=x.device, dtype=x.dtype) # x,y 方向的随机平移量
    # shift = torch.zeros(size=(b, 1, 1, 1, 2), device=x.device, dtype=x.dtype)# 这里设定为0，即不做平移
    shift = shift.repeat(1, t, 1, 1, 1)  # repeat the shift for each image in the sequence 同一个batch里的squence 同一个时间步的shift
    shift = shift.view(b*t, 1, 1, 2)  # reshape shift to match the size of base_grid torch.Size([100, 1, 1, 2])
    shift *= 2.0 / (h_pad)

    grid = base_grid + shift
    output = F.grid_sample(x, grid, padding_mode="zeros", align_corners=False)
    output = output.view(b, t, c, h, w)  # reshape output back to [B, T, C, H, W]
    shift = shift.view(b, t, 1, 2)  # reshape shift back to [B, T, 1, 2]
    return output,shift
def shifts_2d_action_to_Aug(action_2d, shift, pad, w, h):
    # 获取张量的形状
    b, seq, t, dim = action_2d.shape

    # 将原图padding后像坐标标准化 [-1, 1]
    # norm_x_o = ((action_2d[..., 0] + pad) / (w + 2 * pad - 1)) * 2 - 1
    # norm_y_o = ((action_2d[..., 1] + pad) / (h + 2 * pad - 1)) * 2 - 1
    # ori_shift = (pad - 1) / (2 * pad + w - 1)
    norm_x_o = ((action_2d[..., 0]) / (w  - 1)) * 2 - 1
    norm_y_o = ((action_2d[..., 1]) / (h  - 1)) * 2 - 1
    ori_shift = (pad - 1) / (w - 1)
    # 获取shift值
    shift_x = shift[..., 0] 
    shift_y = shift[..., 1]

    # 计算新图像中对应的标准化坐标
    new_norm_x_o = norm_x_o - shift_x + ori_shift
    new_norm_y_o = norm_y_o - shift_y + ori_shift

    # 将标准化坐标转换回实际坐标
    new_point_2d_x = ((new_norm_x_o + 1) / 2) * (w - 1)
    new_point_2d_y = ((new_norm_y_o + 1) / 2) * (h - 1)

    # 合并新的坐标
    augmented_action_2d = torch.stack([new_point_2d_x, new_point_2d_y], dim=-1)

    return augmented_action_2d
    
class PreProcess(): 
    def __init__(
            self,
            rgb_static_pad,
            rgb_gripper_pad,
            rgb_shape, 
            rgb_mean, 
            rgb_std, 
            device,
        ):
        self.resize = Resize(rgb_shape, interpolation=Image.BICUBIC, antialias=True).to(device)
        self.rgb_static_pad = rgb_static_pad
        self.rgb_gripper_pad = rgb_gripper_pad
        self.rgb_mean = torch.tensor(rgb_mean, device=device).view(1, 1, -1, 1, 1)
        self.rgb_std = torch.tensor(rgb_std, device=device).view(1, 1, -1, 1, 1)
    
    def rgb_process(self, rgb_static, rgb_gripper, action_2d,train=False):
        rgb_static_ori = rgb_static.clone()
        rgb_static = rgb_static.float()*(1/255.)
        rgb_gripper = rgb_gripper.float()*(1/255.)
        if train:
            rgb_static,shift = RandomShiftsAug(rgb_static, self.rgb_static_pad)
            new_action_2d = shifts_2d_action_to_Aug(action_2d, shift,self.rgb_static_pad,rgb_static.shape[-1],rgb_static.shape[-2])
            rgb_gripper,_ = RandomShiftsAug(rgb_gripper, self.rgb_gripper_pad)
        else:
            new_action_2d = action_2d
        new_action_2d = resize_points(new_action_2d, (200,200), (224,224))
        rgb_static = self.resize(rgb_static)
        rgb_gripper = self.resize(rgb_gripper)
        # import cv2, numpy as np
        # for batch_idx in range(rgb_static.shape[0]):
        #     for seq_idx in range(rgb_static.shape[1]):
        #         rgb_static_rgb = cv2.cvtColor(rgb_static_ori[batch_idx][seq_idx].permute(1, 2, 0).cpu().numpy(), cv2.COLOR_BGR2RGB)
        #         for point_2d in action_2d[batch_idx,seq_idx,:,:]:
        #             cv2.circle(rgb_static_rgb, tuple(point_2d.int().tolist()), radius=3, color=(0, 0, 255), thickness=-1)
        #         cv2.imshow('original RGB Static Image', rgb_static_rgb)  # 注意这里需要调整回 HWC 格式
        
        #         rgb_static_resized = (rgb_static * 255).clamp(0, 255).byte()
        #         rgb_static_rgb = cv2.cvtColor(rgb_static_resized[batch_idx][seq_idx].permute(1, 2, 0).cpu().numpy(), cv2.COLOR_BGR2RGB)
        #         for point_2d in new_action_2d[batch_idx,seq_idx,:,:]:
        #             cv2.circle(rgb_static_rgb, tuple(point_2d.int().tolist()), radius=3, color=(0, 0, 255), thickness=-1)
        #         cv2.imshow('resize RGB Static Image', rgb_static_rgb)  # 注意这里需要调整回 HWC 格式
        #         cv2.waitKey(0)
        new_action_2d = normalize_data(new_action_2d)
        # torchvision Normalize forces sync between CPU and GPU, so we use our own
        rgb_static = (rgb_static - self.rgb_mean) / (self.rgb_std + 1e-6)
        rgb_gripper = (rgb_gripper - self.rgb_mean) / (self.rgb_std + 1e-6)
        return rgb_static, rgb_gripper,new_action_2d

    
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