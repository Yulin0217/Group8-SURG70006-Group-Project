# import numpy as np

# pose1 = np.load('dvrk_pose/pose_1.npy',allow_pickle=True)

# print('pose',pose1)


import os
import numpy as np

# 定义存储文件夹
output_folder = "./dvrk_pose_txt"
os.makedirs(output_folder, exist_ok=True)

# 读取原始pose.txt
with open("pose.txt", "r") as file:
    content = file.read()

# 按照[[开头分割每组数据
blocks = content.strip().split("[[")
blocks = [block for block in blocks if block.strip()]  # 去除空块

# 遍历每组数据并处理
for idx, block in enumerate(blocks):
    # 去除多余字符，解析矩阵数据
    block = block.replace("[", "").replace("]", "").replace(";", "").split("\n")
    block = [line.replace(",", "").strip() for line in block if line.strip()]  # 去除逗号和空格
    try:
        matrix_data = [list(map(float, line.split())) for line in block]
    except ValueError as e:
        print(f"Error parsing block {idx + 1}: {e}")
        continue

    # 提取旋转矩阵 (3x3) 和平移向量 (3x1)
    rotation_matrix = np.array(matrix_data[:3])  # 前3行是旋转矩阵
    translation_vector = np.array(matrix_data[3])  # 第4行是平移向量

    # 创建4x4齐次变换矩阵
    homogeneous_matrix = np.eye(4)
    homogeneous_matrix[:3, :3] = rotation_matrix
    homogeneous_matrix[:3, 3] = translation_vector

    # 保存到TXT文件，增加精度
    output_path = os.path.join(output_folder, f"pose_{idx + 1}.txt")
    np.savetxt(output_path, homogeneous_matrix, fmt="%.12f", delimiter=" ")
    print(f"Saved: {output_path}")







