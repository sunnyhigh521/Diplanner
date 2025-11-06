# ======================================================================
# Copyright (c) 2025 sql
# PCA Lab, NJUST
# All rights reserved.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# ======================================================================

import torch
# 导入 PerceptNet，并假设你的 percept_net.py 中定义了 BasicBlock
from .percept_net import PerceptNet, BasicBlock 
import torch.nn as nn
import numpy as np # For create_occupancy_grid function
import cv2 # For create_occupancy_grid function

# --- Data Preprocessing Function (You should integrate this into your data loading pipeline) ---
def create_occupancy_grid(ranges, angle_min, angle_increment, 
                          output_height, output_width, 
                          grid_resolution_override=None, grid_size_meters_override=None):
    """
    将激光雷达 ranges 数据转换为一个指定像素尺寸的占用栅格图。
    
    Args:
        ranges (np.array): 激光雷达距离测量数组。
        angle_min (float): 最小扫描角度 (弧度)。
        angle_increment (float): 角度增量 (弧度)。
        output_height (int): 目标栅格图的高度 (像素)。
        output_width (int): 目标栅格图的宽度 (像素)。
        grid_resolution_override (float, optional): 如果指定，强制使用此分辨率。
        grid_size_meters_override (float, optional): 如果指定，强制使用此地图尺寸。
    
    Returns:
        np.array: 生成的占用栅格图 (output_height, output_width)，值在 0-1 之间。
    """
    num_ranges = len(ranges)
    angles = np.arange(num_ranges) * angle_increment + angle_min

    # --- Auto-calculate or use overridden map parameters ---
    if grid_size_meters_override is None:
        # A common assumption is to cover a circular area with a radius of e.g., 15m.
        # So a square map of 30x30 meters might be suitable.
        grid_size_meters = 30.0 
    else:
        grid_size_meters = grid_size_meters_override

    if grid_resolution_override is None:
        # Calculate theoretical resolution based on desired output width
        # Adjust as needed. A common value is 0.05m/pixel or 0.1m/pixel
        grid_resolution = grid_size_meters / max(output_height, output_width) * 1.5 # Adjusted for typical view
    else:
        grid_resolution = grid_resolution_override

    # Initial grid dimensions (can be rectangular if grid_size_meters is different for x/y)
    initial_grid_pixels_height = int(grid_size_meters / grid_resolution)
    initial_grid_pixels_width = int(grid_size_meters / grid_resolution) 
    
    occupancy_grid = np.zeros((initial_grid_pixels_height, initial_grid_pixels_width), dtype=np.float32)

    # Convert polar to Cartesian coordinates
    # Filter out invalid distances (inf, NaN) or values beyond max effective range
    valid_indices = np.isfinite(ranges) & (ranges > 0.01) & (ranges < grid_size_meters / 2.0) 
    
    x_coords = ranges[valid_indices] * np.cos(angles[valid_indices])
    y_coords = ranges[valid_indices] * np.sin(angles[valid_indices])

    # Map Cartesian coordinates to grid pixels
    # Robot is at the center of the grid map
    center_x_pixel = initial_grid_pixels_width // 2
    center_y_pixel = initial_grid_pixels_height // 2 

    pixel_x = (x_coords / grid_resolution + center_x_pixel).astype(int)
    pixel_y = (y_coords / grid_resolution + center_y_pixel).astype(int)

    # Filter out pixels outside map boundaries
    valid_pixels = (pixel_x >= 0) & (pixel_x < initial_grid_pixels_width) & \
                   (pixel_y >= 0) & (pixel_y < initial_grid_pixels_height)
    
    # Mark as occupied
    occupancy_grid[pixel_y[valid_pixels], pixel_x[valid_pixels]] = 1.0 

    # Resize to target dimensions
    if occupancy_grid.shape[0] != output_height or occupancy_grid.shape[1] != output_width:
        occupancy_grid = cv2.resize(occupancy_grid, (output_width, output_height), 
                                     interpolation=cv2.INTER_LINEAR) # INTER_LINEAR or INTER_AREA

    return occupancy_grid

# --- PlannerNet Class Definition ---
class PlannerNet(nn.Module):
    # 修改 __init__ 方法以接收输入通道数，并传递 block 参数给 PerceptNet
    def __init__(self, in_channels=6, encoder_channel=64, k=5):
        super().__init__()
        # 修正 PerceptNet 实例化：传入 block 和 in_channels 参数
        # 这里的 BasicBlock 是一个假设，请根据你实际的 percept_net.py 中的类来替换
        self.encoder = PerceptNet(block=BasicBlock, layers=[2, 2, 2, 2])
        self.decoder = Decoder(1024, encoder_channel, k)

    # --- MODIFICATION START ---
    # 修改 forward 方法以接收图像、LiDAR地图和行人地图
    def forward(self, x_image,goal,scan_map, ped_map): 
        # x_image: 图像数据 (N, 3, H, W)
        # scan_map: LiDAR 扫描数据 (N, 1, H, W)
        # ped_map: 行人地图 (N, 2, H, W)
        
        # 将所有特征图拼接在通道维度上
        # 将拼接后的输入送入编码器
        x = self.encoder(x_image,scan_map, ped_map) 
        x, c = self.decoder(x, goal)
        return x, c
    # --- MODIFICATION END ---


class Decoder(nn.Module):
    def __init__(self, in_channels, goal_channels, k=5):
        super().__init__()
        self.k = k
        self.relu    = nn.ReLU(inplace=True)
        self.fg      = nn.Linear(3, goal_channels)
        self.sigmoid = nn.Sigmoid()

        self.conv1 = nn.Conv2d((in_channels + goal_channels), 512, kernel_size=5, stride=1, padding=1)
        self.conv2 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=0);

        self.fc1   = nn.Linear(256 * 128, 1024) 
        self.fc2   = nn.Linear(1024, 512)
        self.fc3   = nn.Linear(512,  k*3*3)
        
        self.frc1 = nn.Linear(1024, 128)
        self.frc2 = nn.Linear(128, 3)

    def forward(self, x, goal):
        # compute goal encoding
        goal = self.fg(goal[:, 0:3])
        goal = goal[:, :, None, None].expand(-1, -1, x.shape[2], x.shape[3])
        # cat x with goal in channel dim
        x = torch.cat((x, goal), dim=1)
        # compute x
        try:
            x = self.relu(self.conv1(x))  # size = (N, 512, x.H/32, x.W/32)
            x = self.relu(self.conv2(x))  # size = (N, 512, x.H/60, x.W/60)
            x = torch.flatten(x, 1)

            f = self.relu(self.fc1(x))

            x = self.relu(self.fc2(f))
            x = self.fc3(x)
            x = x.reshape(-1, self.k, 3)

            c = self.relu(self.frc1(f))
            c = self.sigmoid(self.frc2(c))
        except Exception as e:
            import pdb
            pdb.set_trace()

        return x, c