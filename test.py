import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# 加载图像
img = Image.open('evaluation/Image_file/image.png')  # 确保路径正确

# 重新调整高斯分布的参数以覆盖整个图像区域
mean_full = [img.width / 2, img.height / 2]  # 保持中心
cov_full = [[img.width*2, 0], [0, img.height*2]]  # 控制散布范围

# 生成新的高斯分布的点
x_full, y_full = np.random.multivariate_normal(mean_full, cov_full, 30).T
x = [149,156, 166, 173, 179, 185, 190,200, 205, 210, 215, 220, 225, 235,243,254,270,281,294,305,316,328,332,333,328,329,328,328]
y = [350,339, 330, 320, 312, 304, 291,285, 272, 257, 251, 242, 230, 222,217,209,209,208,202,196,184,174,155,138,123,108,93,79]
x_full = np.array([x])
        
y_full = np.array([y])
# 设置图像大小匹配原始图像大小
fig, ax = plt.subplots()
ax.imshow(img)
ax.scatter(x_full, y_full, alpha=0.5, edgecolor='w', linewidth=0.5, s=30, color='red')  # 绘制高斯点
ax.axis('off')  # 隐藏坐标轴

# 保存图片
plt.savefig('evaluation/Image_file/framework_gaussian.png', bbox_inches='tight', pad_inches=0)
plt.close()

# import torch.nn.functional as F
# import matplotlib.pyplot as plt
# from PIL import Image
# def resample_sequence(sequence, target_length):
#     """
#     使用插值将 sequence 重新采样到 target_length，并将结果四舍五入为整数
#     :param sequence: 原始序列，形状为 [N, 2]
#     :param target_length: 目标序列长度
#     :return: 重新采样后的序列，形状为 [target_length, 2]
#     """
#     # 确保 sequence 是 float 类型
#     sequence = sequence.float()
#     sequence = sequence.unsqueeze(0).permute(0, 2, 1)  # 调整形状为 (1, 2, N)
#     resampled_sequence = F.interpolate(sequence, size=target_length, mode='linear', align_corners=True)
#     resampled_sequence = resampled_sequence.permute(0, 2, 1).squeeze(0)  # 调整回原始形状 (target_length, 2)
#     # 将结果四舍五入为整数
#     resampled_sequence = torch.round(resampled_sequence).int()
    
#     return resampled_sequence
# point_2d_resized = resample_sequence(point_2d_resized,120)
# point_2d_resized = resample_sequence(point_2d_resized,30)
# x_full = point_2d_resized[:, 0].cpu().numpy()
# y_full = point_2d_resized[:, 1].cpu().numpy()
# fig, ax = plt.subplots()
# ax.imshow(img_copy)
# ax.scatter(
#     x_full, y_full, 
#     alpha=0.8,  # 设置透明度
#     edgecolor='black',  # 边框颜色
#     linewidth=1.0,  # 边框宽度
#     s=60,  # 点的大小
#     color='Yellowgreen'  # 点的颜色
# )
# ax.axis('off')  # 隐藏坐标轴
# ax.cla()
