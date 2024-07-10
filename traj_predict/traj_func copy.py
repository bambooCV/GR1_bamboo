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
def apply_shifts_to_coords(coords, shifts, h, w):
    """
    将标准化的平移量应用于原始图像的坐标。

    参数：
    coords: 原始图像的坐标，形状为 [batch_size, time_steps, num_points, 2]
    shifts: 标准化的平移量张量，形状为 [batch_size, time_steps, 2]
    h: 原始图像的高度
    w: 原始图像的宽度

    返回：
    new_coords: 应用平移后的新坐标，形状为 [batch_size, time_steps, num_points, 2]
    """
    # 将标准化平移量转换为像素坐标系的平移量
    pixel_shifts = convert_shifts_to_pixel_coords(shifts, h, w)
    
    # 扩展平移量的维度以匹配坐标的维度
    pixel_shifts = pixel_shifts.unsqueeze(2)  # 形状变为 [batch_size, time_steps, 1, 2]
    
    # 将平移量应用于原始坐标
    new_coords = coords + pixel_shifts
    
    return new_coords
def convert_shifts_to_pixel_coords(shifts, h, w):
    """
    将标准化的平移量转换为原始图像的像素坐标。

    参数：
    shifts: 标准化的平移量张量，形状为 [batch_size, time_steps, 2]
    h: 原始图像的高度
    w: 原始图像的宽度

    返回：
    pixel_shifts: 像素坐标系下的平移量张量，形状为 [batch_size, time_steps, 2]
    """
    # 转换平移量
    shift_y = shifts[..., 1] * (h / 2.0)
    shift_x = shifts[..., 0] * (w / 2.0)
    
    # 拼接转换后的平移量
    pixel_shifts = torch.stack([shift_x, shift_y], dim=-1)
    
    return pixel_shifts
def calculate_transformed_coordinates_batch(coords, pad, h, w, shifts):
    """
    根据原始图像上的一系列坐标点和 RandomShiftsAug 函数的处理，
    计算变换后的坐标点。

    参数：
    coords: 原始坐标点张量 [batch_size, time_steps, num_points, 2]
    pad: 填充大小
    h, w: 原始图像高度和宽度
    shifts: 随机平移量张量 (形状: [batch_size, time_steps, 2])

    返回：
    new_coords: 变换后的坐标点张量 [batch_size, time_steps, num_points, 2]
    """
    h_pad, w_pad = h + 2 * pad, w + 2 * pad
    eps = 1.0 / h_pad
    arange = torch.linspace(-1.0 + eps, 1.0 - eps, h_pad, device=coords.device, dtype=coords.dtype)

    batch_size, time_steps, num_points, _ = coords.size()
    new_coords = torch.zeros_like(coords)

    for b in range(batch_size):
        for t in range(time_steps):
            shift = shifts[b, t] * (2.0 / h_pad)
            for p in range(num_points):
                x, y = coords[b, t, p]
                # 计算填充后的坐标
                x_padded = int(x + pad)
                y_padded = int(y + pad)
                
                # 生成基础网格坐标 gx 和 gy
                if x_padded >= h_pad:
                    x_padded = h_pad - 1
                if y_padded >= w_pad:
                    y_padded = w_pad - 1
                
                gx = arange[x_padded]
                gy = arange[y_padded]
                
                # 应用随机平移量
                gx_shifted = gx + shift[0]
                gy_shifted = gy + shift[1]
                
                # 将网格坐标转换回图像坐标
                new_x = ((gx_shifted + 1) / 2) * h - pad
                new_y = ((gy_shifted + 1) / 2) * w - pad
                
                new_coords[b, t, p] = torch.tensor([new_x, new_y])

    return new_coords


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

            # 获取平移量：
            shift_amounts = shift * (rgb_static.shape[-2] + 2 * self.rgb_static_pad) / 2  # 将 [-1, 1] 范围的平移量转换为像素值

            rgb_gripper,_ = RandomShiftsAug(rgb_gripper, self.rgb_gripper_pad)
        import cv2,numpy as np

        for batch_idx in range(rgb_static.shape[0]):
            for seq_idx in range(rgb_static.shape[1]):
                rgb_static_rgb_shift = cv2.cvtColor(rgb_static[batch_idx][seq_idx].permute(1, 2, 0).cpu().numpy(), cv2.COLOR_BGR2RGB)
                x_o, y_o = 16, 20

                # 将原图padding后像坐标标准化
                norm_x_o = ((x_o+10) / (220 - 1)) * 2 - 1 # [-1, 1]
                norm_y_o = ((y_o+10) / (220 - 1)) * 2 - 1 # [-1, 1]

                ori_shift = ((10) / (220 - 1))
                # 计算新图像中对应的标准化坐标
                shift_x, shift_y = shift[batch_idx, seq_idx, 0, 0].item(), shift[batch_idx, seq_idx, 0, 1].item()
                new_norm_x_o = norm_x_o - shift_x + ori_shift
                new_norm_y_o = norm_y_o - shift_y + ori_shift

                # 将标准化坐标转换回实际坐标
                new_point_2d_x = ((new_norm_x_o + 1) / 2) * (200 - 1)
                new_point_2d_y = ((new_norm_y_o + 1) / 2) * (200 - 1)
                print(shift_x,shift_y)
                print(new_point_2d_x,new_point_2d_y)
                cv2.circle(rgb_static_rgb_shift, (int(new_point_2d_x), int(new_point_2d_y)), radius=3, color=(0, 0, 255), thickness=-1)
                cv2.imshow('Augmented RGB Static Image', rgb_static_rgb_shift)  # 注意这里需要调整回 HWC 格式
                
                
                rgb_static_rgb = cv2.cvtColor(rgb_static_ori[batch_idx][seq_idx].permute(1, 2, 0).cpu().numpy(), cv2.COLOR_BGR2RGB)
                point_2d = torch.tensor([16, 20], dtype=torch.float32)
                cv2.circle(rgb_static_rgb, tuple(point_2d.int().tolist()), radius=3, color=(0, 0, 255), thickness=-1)
                cv2.imshow('original RGB Static Image', rgb_static_rgb)  # 注意这里需要调整回 HWC 格式
                cv2.waitKey(0)
        
        
        
        # rgb_static = self.resize(rgb_static)
        # # 计算resize后的坐标点
        # orig_h, orig_w = rgb_static_ori.shape[-2], rgb_static_ori.shape[-1]
        # new_h, new_w = rgb_static.shape[-2], rgb_static.shape[-1]
        # resize_scale_x = new_w / orig_w
        # resize_scale_y = new_h / orig_h
        # new_action_2d_resized = new_action_2d.clone()
        # new_action_2d_resized[..., 0] *= resize_scale_x
        # new_action_2d_resized[..., 1] *= resize_scale_y
        # rgb_gripper = self.resize(rgb_gripper)
        # visualization resized image
        
        # Convert tensor to NumPy array for visualization
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
        
        # visualization croped image


        
        # torchvision Normalize forces sync between CPU and GPU, so we use our own
        rgb_static = (rgb_static - self.rgb_mean) / (self.rgb_std + 1e-6)
        rgb_gripper = (rgb_gripper - self.rgb_mean) / (self.rgb_std + 1e-6)
        return rgb_static, rgb_gripper

    
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