# import numpy as np
# import matplotlib.pyplot as plt
# from PIL import Image

# # 加载图像
# img = Image.open('evaluation/Image_file/image.png')  # 确保路径正确

# # 重新调整高斯分布的参数以覆盖整个图像区域
# mean_full = [img.width / 2, img.height / 2]  # 保持中心
# cov_full = [[img.width*2, 0], [0, img.height*2]]  # 控制散布范围

# # 生成新的高斯分布的点
# x_full, y_full = np.random.multivariate_normal(mean_full, cov_full, 30).T
# # x = [149,156, 166, 173, 179, 185, 190,200, 205, 210, 215, 220, 225, 235,243,254,270,281,294,305,316,328,332,333,328,329,328,328]
# # y = [350,339, 330, 320, 312, 304, 291,285, 272, 257, 251, 242, 230, 222,217,209,209,208,202,196,184,174,155,138,123,108,93,79]
# # x_full = np.array([x])
        
# # y_full = np.array([y])
# # 设置图像大小匹配原始图像大小
# fig, ax = plt.subplots()
# ax.imshow(img)
# ax.scatter(x_full, y_full, alpha=0.5, edgecolor='w', linewidth=0.5, s=30, color='lightblue')  # 绘制高斯点
# ax.axis('off')  # 隐藏坐标轴

# # 保存图片
# plt.savefig('evaluation/Image_file/framework_gaussian.png', bbox_inches='tight', pad_inches=0)
# plt.close()

# import cv2
# import numpy as np
# img = cv2.imread('evaluation/Image_file/image.png')
# if img is None:
#     print("Error: Image not found. Please check the path.")
# else:
#     # Set Gaussian distribution parameters to cover the whole image area
#     mean_full = [img.shape[1] / 2, img.shape[0] / 2]  # Center of the image
#     cov_full = [[img.shape[1]*2, 0], [0, img.shape[0]*2]]  # Spread range

#     # Generate points from the Gaussian distribution
#     # points = np.random.multivariate_normal(mean_full, cov_full, 30).astype(int)
#     # Manually set x and y points
#     x = [147, 142, 145, 149, 154, 160, 166, 173, 184, 195, 212, 225, 244, 260, 280, 300, 318, 331, 336, 343, 343, 342, 340, 338, 330, 312]
#     y = [350, 327, 300, 284, 269, 255, 246, 235, 225, 216, 206, 200, 192, 190, 187, 184, 179, 169, 160, 151, 137, 121, 105, 92, 77, 78]

#     # Combine x and y into points array
#     points = np.array(list(zip(x, y)))
#     # Draw points on the image
#     for point in points:
#         # Draw a white circle as the background
#         cv2.circle(img, (point[0], point[1]), 6, (255, 255, 255), -1)  # White circle
#         # Draw a light blue circle on top
#         cv2.circle(img, (point[0], point[1]), 5, (235, 206, 135), -1)  # Sky blue circle

#     # Save the result
#     cv2.imwrite('evaluation/Image_file/framework_gaussian.png', img)


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














# import matplotlib.pyplot as plt
# import numpy as np

# # 数据用于绘图
# categories = ['task_D_D']  # 类别
# baseline = 3.0 # 设定基线值为无提示的情况
# values = np.array([
#     [3.34],  # DTP w/o prompt
#     [3.55],  # DTP
# ])

# # 创建条形图
# # Creating the bar plot with specified figure size
# fig, ax = plt.subplots(figsize=(4, 8))  # Adjust figure size to be more rectangular
# width = 0.1  # Reduce the width of the bars
# spacing = 0.2  # Space between bars, adjusted for visual distinction
# x = np.arange(len(categories))  # the label locations

# # 绘制两组数据的条形
# ax.bar(x - spacing/2, values[0] - baseline, width, bottom=baseline, label='DTP w/o prompt', color='blue')
# ax.bar(x + spacing/2, values[1] - baseline, width, bottom=baseline, label='DTP', color='red')

# # 添加一些文本标签，标题以及自定义的x轴刻度标签等
# ax.set_xlabel('Performance Metrics in Task_D_D')
# ax.set_ylabel('Average Length')
# # ax.set_title('Performance Metrics in Task_D_D')
# ax.set_xticks([])
# # ax.set_xticklabels(categories)
# ax.legend()

# # 添加网格线
# ax.grid(True, which='both', linestyle='--', linewidth=0.5)

# fig.tight_layout()

# plt.show()


# import json

# # 从文件中读取数据
# with open('Save/task10_ABCD_D/GR1_2D_trajectory_0830_V2/result_e45_0.txt', 'r', encoding='utf-8') as file:
#     data = json.load(file)

# # 计算成功率
# for task, info in data["null"]["task_info"].items():
#     success_rate = (info["success"] / info["total"]) * 100
#     print(f"{task}: {success_rate:.2f}%")
    
    

# result 结果自动化统计
# import json

# 初始化一个字典来存储总的数据
# total_data = {"null": {"task_info": {}}}

# # 读取每个文件并合并数据
# for i in range(4):
#     with open(f'Save/task10_ABCD_D/GR1_2D_trajectory_0830_V2/result_e50_{i}.txt', 'r', encoding='utf-8') as file:
#         data = json.load(file)
        
#         for task, info in data["null"]["task_info"].items():
#             if task not in total_data["null"]["task_info"]:
#                 total_data["null"]["task_info"][task] = {"success": 0, "total": 0}
            
#             total_data["null"]["task_info"][task]["success"] += info["success"]
#             total_data["null"]["task_info"][task]["total"] += info["total"]

# # 计算成功率并输出
# for task in sorted(total_data["null"]["task_info"].keys()):
#     info = total_data["null"]["task_info"][task]
#     success_rate = (info["success"] / info["total"]) * 100 if info["total"] > 0 else 0
#     print(f"{task}: {success_rate:.2f}%")

#图标
import matplotlib.pyplot as plt
import numpy as np

# 数据
tasks = [
    "Interact With Blocks\n", 
    "Interact With Blocks Based Env\n", 
    "Interact With Articulated Objects\n"
]
baseline_success_rate = [60.24, 63.24, 89.58]  # 基线成功率
ours_success_rate = [56.22, 74.68, 94.12]  # 我们的成功率

# 绘图
x = np.arange(len(tasks))  # 任务的数量
width = 0.35  # 柱子的宽度

fig, ax = plt.subplots(figsize=(10, 6))

# 绘制 baseline 和 ours 的柱状图
baseline_bars = ax.bar(x - width/2, baseline_success_rate, width, label='Baseline', color='green')
ours_bars = ax.bar(x + width/2, ours_success_rate, width, label='Ours', color='red')

# 添加文本标签
ax.set_ylabel('Successful Rate (%)',fontsize=16)
ax.set_title('Trained on 10% Data from ABCD->D Setting',fontsize=16)
ax.set_xticks(x)  # 确保标签和x对齐
ax.set_xticklabels(tasks, ha="center",fontsize=16)  # 设置标签为水平显示
ax.legend(fontsize=24) 

# 添加每个柱子的数值标签
def add_labels(bars):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}%', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom',fontsize=14)

add_labels(baseline_bars)
add_labels(ours_bars)

# 自动调整布局
fig.tight_layout()

# 显示图像
plt.show()


print("fsc test")

