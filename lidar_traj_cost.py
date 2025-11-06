# ======================================================================
# Copyright (c) 2025 sql
# PCA Lab, NJUST
# All rights reserved.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# ======================================================================

import torch
import pypose as pp
from traj_opt import TrajOpt
import torch.nn.functional as F
from LidarCostUtil import ESDFGenerator
import numpy as np
import scipy.ndimage
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
from .LidarCostMap import LidarCostMap
import time
torch.set_default_dtype(torch.float32)
import os
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 设置非交互式后端 - 确保不显示图像
import matplotlib.pyplot as plt
import torch
import shutil
class LidarTrajCost:
    def __init__(self, gpu_id=0):
        # self.tsdf_map = TSDF_Map(gpu_id)
        self.opt = TrajOpt()
        self.is_map = False
        self.rate = 20
        self.max_linear_speed = 1.0
        self.max_angular_speed = 0.35
        self.PI = np.pi
        return None

    def TransformPoints(self, odom, points):
        batch_size, num_p, _ = points.shape
        world_ps = pp.identity_SE3(batch_size, num_p, device=points.device, requires_grad=points.requires_grad)
        world_ps.tensor()[:, :, 0:3] = points
        world_ps = pp.SE3(odom[:, None, :]) @ pp.SE3(world_ps)
        return world_ps
    
    def SetMap(self, root_path, map_name):
        self.tsdf_map.ReadTSDFMap(root_path, map_name)
        self.is_map = True
        return
    def transform_to_local_coordinates(self,points_global, robot_pose):
        # Ensure robot_pose is a tensor for consistent operations
        if not isinstance(robot_pose, torch.Tensor):
            robot_pose = torch.tensor(robot_pose, dtype=points_global.dtype, device=points_global.device)

        robot_x, robot_y, robot_yaw = robot_pose[0], robot_pose[1], robot_pose[2]

        # 1. 平移：将机器人位置作为新的原点
        translated_x = points_global[:, 0] - robot_x
        translated_y = points_global[:, 1] - robot_y

        # 2. 旋转：绕新原点旋转，使机器人航向角对齐局部X轴
        # 旋转矩阵的逆：
        # [ cos(theta)  sin(theta)]
        # [-sin(theta)  cos(theta)]
        cos_yaw = torch.cos(-robot_yaw) # Negative yaw for inverse rotation
        sin_yaw = torch.sin(-robot_yaw)

        rotated_x = translated_x * cos_yaw - translated_y * sin_yaw
        rotated_y = translated_x * sin_yaw + translated_y * cos_yaw

        return torch.stack((rotated_x, rotated_y), dim=-1)
            # 转换点的代码...


    # 将 map_and_update_map 改造为函数，并返回一个表示第一个 batch 是否被更新的标志
    def map_and_update_map(self,trajectory_tensor, current_now_pose_batch, lidar_map_list):
        batch_size = trajectory_tensor.shape[0] 
        first_batch_updated = False 

        for b in range(batch_size):
            lidar_map_obj = lidar_map_list[b]
            current_now_pose = current_now_pose_batch[b] # 机器人当前的 (x_odom, y_odom, yaw_odom)

            map_width_obj = int(lidar_map_obj.info['width'])
            map_height_obj = int(lidar_map_obj.info['height'])

            current_agent_trajectories = trajectory_tensor[b] # (num_agents, num_timesteps, 2)
            num_agents = current_agent_trajectories.shape[0]

            for agent_idx in range(num_agents):
                agent_trajectory_odom = current_agent_trajectories[agent_idx] # 智能体轨迹在odom坐标系下 (num_timesteps, 2)

                if torch.sum(torch.abs(agent_trajectory_odom)) == 0:
                    continue

                # --- 关键修改：将智能体轨迹从 Odom 坐标系转换到机器人局部坐标系 ---
                # current_now_pose 的形状是 (3,) 即 (robot_x_odom, robot_y_odom, robot_yaw_odom)
                agent_trajectory_local = self.transform_to_local_coordinates(agent_trajectory_odom, current_now_pose)
                # 现在 agent_trajectory_local 中的点是相对于机器人当前位置和航向的坐标

                # lidar_map_obj.Pos2Ind 现在应该接收局部坐标系的轨迹点
                norm_inds, valid_mask = lidar_map_obj.Pos2Ind(agent_trajectory_local, current_now_pose)

                valid_norm_inds = norm_inds[valid_mask]

                if valid_norm_inds.numel() > 0:
                    inds_pixel_x = ((valid_norm_inds[:, 0] + 1) / 2 * (map_width_obj - 1)).round().long()
                    inds_pixel_y = ((valid_norm_inds[:, 1] + 1) / 2 * (map_height_obj - 1)).round().long()

                    temp_map_update = np.zeros_like(lidar_map_obj.data, dtype=np.float32)
                    radius = 3 

                    center_x_coords = inds_pixel_x.cpu().numpy()
                    center_y_coords = inds_pixel_y.cpu().numpy()

                    dx_vals = np.arange(-radius, radius + 1)
                    dy_vals = np.arange(-radius, radius + 1)
                    DX, DY = np.meshgrid(dx_vals, dy_vals)
                    valid_offsets_mask = DX**2 + DY**2 <= radius**2
                    
                    all_relative_offsets = np.stack([DX[valid_offsets_mask], DY[valid_offsets_mask]], axis=-1)

                    expanded_points = (np.array([center_x_coords, center_y_coords]).T[:, np.newaxis, :] + 
                                    all_relative_offsets[np.newaxis, :, :])

                    expanded_points_flat = expanded_points.reshape(-1, 2)

                    final_x = np.clip(expanded_points_flat[:, 0], 0, map_width_obj - 1)
                    final_y = np.clip(expanded_points_flat[:, 1], 0, map_height_obj - 1)

                    temp_map_update[final_y, final_x] = 1.0

                    if np.any(temp_map_update > 0): 
                        old_data = lidar_map_obj.data.copy()
                        lidar_map_obj.data = np.maximum(lidar_map_obj.data, temp_map_update)
                        if not np.array_equal(old_data, lidar_map_obj.data):
                            if b == 0: 
                                first_batch_updated = True
        return first_batch_updated


    def visualize_costmaps_with_traj(self, gt_costmap_list, waypoints, save_dir=None, now_pose=None):
        """
        可视化未来五帧成本图，并叠加轨迹点。

        参数:
            gt_costmap_list (List[LidarCostMap]): 未来五个时间戳的成本图对象。
            waypoints (Tensor or np.ndarray): 预测轨迹，形状为 [5, 2]，局部坐标系下。
            save_dir (str or None): 如果不为 None，则将图像保存到该目录下。
            now_pose (np.ndarray or None): 当前位姿，可选，用于标注原点位置。
        """
        # 转换为 tensor
        if not isinstance(waypoints, torch.Tensor):
            waypoints = torch.tensor(waypoints, dtype=torch.float32)

        for i, cost_map in enumerate(gt_costmap_list):
            plt.figure(figsize=(6, 6))
            plt.imshow(cost_map.data, origin='lower', cmap='gray')
            plt.title(f"Future Costmap t+{i+1}")

            # 绘制预测轨迹点
            for pt in waypoints:
                if pt.numel() != 2:
                    continue  # 跳过异常点

                pt = pt.to(torch.float32)
                grid = cost_map.Pos2Ind(pt, now_pose)  # 使用当前车的位置作为参考
                if grid is not None:
                    plt.plot(grid[1], grid[0], 'rx')  # 注意：imshow 是 row,col 格式

            # 绘制当前原点位置（蓝点）
            if now_pose is not None:
                grid = cost_map.Pos2Ind(torch.tensor([0.0, 0.0,0.0]), now_pose)
                if grid is not None:
                    plt.plot(grid[1], grid[0], 'bo', label='origin')

            plt.legend()
            plt.grid(True)

            # 保存或展示图像
            if save_dir is not None:
                os.makedirs(save_dir, exist_ok=True)
                plt.savefig(f"{save_dir}/costmap_t+{i+1}.png", bbox_inches='tight')
                plt.close()
            else:
                plt.show()
    def compute_traj_cost(self,trajs, now_poses, local_costmaps, gt_poses):
        B, T, _ = trajs.shape
        device = trajs.device
        costs = torch.zeros(B, device=device)

        for b in range(B):
            costmap = local_costmaps[b]
            now_pose0 = now_poses[b]  # 初始坐标系的 pose
            traj = trajs[b]  # [T, 2]

            for t in range(T):
                # 获取第 t 帧的参考位姿（odom）
                gt_pose_t = gt_poses[b, t].tolist()  # [x, y, yaw]

                # 当前轨迹点，原本是相对 now_pose0 的局部坐标系
                wp_rel = traj[t]  # Tensor [2]

                # 将 wp_rel 变换到世界坐标系（即从初始帧出发的世界坐标）
                wp_world = local_to_world(wp_rel, now_pose0)  # numpy [2]

                # 再将其转换为相对 gt_pose_t 的局部坐标
                wp_local_t = world_to_local(wp_world, gt_pose_t)  # numpy [2]

                # 转成 Tensor
                wp_local_t = torch.tensor(wp_local_t, dtype=torch.float32, device=device)

                # 将该点转换为 costmap 坐标系下的索引并取 cost
                try:
                    ind = costmap.Pos2Ind(wp_local_t.unsqueeze(0),torch)  # [1, 2]
                    ix, iy = ind[0, 0], ind[0, 1]
                    cost = costmap.data[iy, ix]
                    costs[b] += cost
                except Exception as e:
                    print(f"Error in costmap sampling for traj {b} step {t}: {e}")
                    costs[b] += 1000.0  # 设置为大值以惩罚出界轨迹

        return costs


    def local_to_world(self,local_pt, ref_pose):
        """
        将点 local_pt 从 ref_pose 局部坐标系转换到世界坐标系
        输入:
            local_pt: Tensor 或 list [x, y]
            ref_pose: list [x, y, yaw]
        输出:
            world_pt: numpy [2]
        """
        x, y = local_pt
        tx, ty, theta = ref_pose
        R = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta),  np.cos(theta)]
        ])
        return R @ np.array([x, y]) + np.array([tx, ty])


    def world_to_local(self,world_pt, ref_pose):
        """
        将 world_pt 从世界坐标系转换为 ref_pose 局部坐标系
        输入:
            world_pt: numpy [2]
            ref_pose: list [x, y, yaw]
        输出:
            local_pt: numpy [2]
        """
        tx, ty, theta = ref_pose
        R_inv = np.array([
            [np.cos(theta), np.sin(theta)],
            [-np.sin(theta), np.cos(theta)]
        ])
        delta = np.array(world_pt) - np.array([tx, ty])
        return R_inv @ delta
    
        
    def CostofTraj_lobby(self, trajic_future, lidar_map, waypoints, odom, goal, ahead_dist, now_pose, transform_matrics,
                          alpha=8, beta=1.0, gamma=2.0, obstacle_thred=0.3, inflate_px=3,
                          epsilon=1.0, zeta=1.0, eta=1.0,
                          gloss_weight=5):
        
        batch_size, T_waypoints, _ = waypoints.shape
        device = waypoints.device
        save_dir = "/opt/data/private/ros/trajPre/FIREDA/THUD_Robot/iPlanner/img"
        
        # 确保所有张量在正确的设备和数据类型上
        odom = odom.to(device).to(torch.float32) if isinstance(odom, torch.Tensor) else odom
        goal = goal.to(device).to(torch.float32)
        now_pose = now_pose.to(device).to(torch.float32)
        waypoints = waypoints.to(device).to(torch.float32)
        
        # 初始化损失
        oloss = torch.tensor(0.0, device=device)
        hloss = torch.tensor(0.0, device=device)
        gloss = torch.tensor(0.0, device=device)
        mloss = torch.tensor(0.0, device=device)
        total_smoothness_loss = torch.tensor(0.0, device=device)
        fear_labels = torch.zeros(batch_size, 1, device=device)
        
        # 确保我们有全局成本图
        if lidar_map is None or len(lidar_map) != batch_size:
            print(f"Warning: Invalid lidar_map provided. Using default identity matrix.")
            cost_grids = torch.eye(4).unsqueeze(0).repeat(batch_size, 1, 1, 1).to(device)
            map_origins = [(-3.0, -4.45) for _ in range(batch_size)]
            map_resolutions = [0.05 for _ in range(batch_size)]
            map_widths = [160 for _ in range(batch_size)]
            map_heights = [160 for _ in range(batch_size)]
        else:
            # 创建成本图张量
            cost_grids = []
            map_origins = []
            map_resolutions = []
            map_widths = []
            map_heights = []
            
            for b in range(batch_size):
                cost_map_obj = lidar_map[b]
                map_origin = cost_map_obj.info['origin']['position']
                map_origins.append((map_origin['x'], map_origin['y']))
                map_resolutions.append(cost_map_obj.info['resolution'])
                map_widths.append(cost_map_obj.info['width'])
                map_heights.append(cost_map_obj.info['height'])
                
                prob_map = cost_map_obj.data
                obs_mask = ((prob_map > obstacle_thred*100)).astype(np.uint8)
                esdf = scipy.ndimage.distance_transform_edt(1 - obs_mask)
                normalized_cost = np.exp(-esdf / 6.0)
                cost_map = torch.tensor(normalized_cost, dtype=torch.float32, device=device)
                
                if cost_map.dim() == 2:
                    cost_map = cost_map.unsqueeze(0)
                elif cost_map.dim() == 3:
                    pass
                else:
                    print(f"Warning: Invalid cost map shape {cost_map.shape}. Using identity matrix.")
                    cost_map = torch.eye(4, device=device).unsqueeze(0)
                
                cost_grids.append(cost_map)
            cost_grids = torch.stack(cost_grids, dim=0)
        
        # 将轨迹点从局部坐标系转换到全局坐标系
        global_waypoints = []
        for b in range(batch_size):
            if transform_matrics is not None and len(transform_matrics) > b:
                transform_matrix = transform_matrics[b]
            else:
                transform_matrix = torch.eye(4, device=device)
            
            transform_matrix = transform_matrix.to(torch.float32)
            
            waypoints_homogeneous = torch.cat([
                waypoints[b], 
                torch.ones(T_waypoints, 1, device=device)
            ], dim=1)
            
            global_points = torch.matmul(transform_matrix, waypoints_homogeneous.t()).t()
            global_waypoints.append(global_points[:, :3])

        global_waypoints = torch.stack(global_waypoints, dim=0)

        # 存储像素坐标用于膨胀处理
        pixel_coords = []
        for b in range(batch_size):
            map_origin_x = map_origins[b][0]
            map_origin_y = map_origins[b][1]
            map_resolution = map_resolutions[b]
            
            pixel_x = (global_waypoints[b, :, 0] - map_origin_x) / map_resolution
            pixel_y = (global_waypoints[b, :, 1] - map_origin_y) / map_resolution
            
            pixel_coords.append((pixel_x, pixel_y))

        # === 修正：对轨迹点进行膨胀处理并计算总成本 ===
        # 创建一个新的膨胀后的成本张量，用于存储每个轨迹点的总成本
        inflated_oloss_M = torch.zeros_like(waypoints[:,:,0])
        
        for b in range(batch_size):
            map_width = map_widths[b]
            map_height = map_heights[b]
            
            pixel_x, pixel_y = pixel_coords[b]
            
            pixel_x_int = torch.clamp(pixel_x.round().long(), 0, map_width - 1)
            pixel_y_int = torch.clamp(pixel_y.round().long(), 0, map_height - 1)
            
            for t in range(T_waypoints):
                x = pixel_x_int[t].item()
                y = pixel_y_int[t].item()
                
                min_x = max(0, x - inflate_px)
                max_x = min(map_width - 1, x + inflate_px)
                min_y = max(0, y - inflate_px)
                max_y = min(map_height - 1, y + inflate_px)
                
                inflated_region = cost_grids[b, 0, min_y:max_y+1, min_x:max_x+1]
                
                if inflated_region.numel() > 0:
                    # 修正：直接使用总成本
                    total_cost = torch.max(inflated_region)
                    # 更新膨胀后的成本值，存储的是总成本
                    inflated_oloss_M[b, t] = total_cost
                else:
                    inflated_oloss_M[b, t] = 0.0
        
        # 计算障碍物成本
        oloss = torch.mean(torch.mean(inflated_oloss_M, dim=1)) # 对每个批次的所有点成本求和，再对所有批次取平均
        
        # 高度成本（暂时设为0，因为全局成本图可能不包含高度信息）
        hloss = torch.tensor(0.0, device=device)
        
        # 目标成本
        gloss = torch.norm(goal[:, :3] - waypoints[:, -1, :], dim=1)
        gloss = torch.mean(gloss)
        
        # 运动学成本
        desired_wp = self.opt.TrajGeneratorFromPFreeRot(goal[:, None, 0:3], step=1.0 / (T_waypoints - 1))
        desired_ds = torch.norm(desired_wp[:, 1:, :] - desired_wp[:, :-1, :], dim=2)
        wp_ds = torch.norm(waypoints[:, 1:, :] - waypoints[:, :-1, :], dim=2)
        mloss = torch.abs(desired_ds - wp_ds)
        mloss = torch.sum(mloss, dim=1)
        mloss = torch.mean(mloss, dim=0)
        
        # 平滑度成本
        if T_waypoints >= 3:
            p0 = waypoints[:, :-2, :2]
            p1 = waypoints[:, 1:-1, :2]
            p2 = waypoints[:, 2:, :2]
            vec1 = p1 - p0
            vec2 = p2 - p1
            vec1_norm = vec1 / (torch.norm(vec1, dim=2, keepdim=True) + 1e-6)
            vec2_norm = vec2 / (torch.norm(vec2, dim=2, keepdim=True) + 1e-6)
            cos_angles = torch.sum(vec1_norm * vec2_norm, dim=2)
            angle_diffs = torch.acos(torch.clamp(cos_angles, -1.0, 1.0))
            curvature_loss = angle_diffs.mean()
        else:
            curvature_loss = torch.tensor(0.0, device=device)
        
        velocities = waypoints[:, 1:, :2] - waypoints[:, :-1, :2]
        speed = torch.norm(velocities, dim=2)
        if T_waypoints > 2:
            velocity_change = speed[:, 1:] - speed[:, :-1]
            velocity_change_loss = torch.mean(torch.abs(velocity_change))
        else:
            velocity_change_loss = torch.tensor(0.0, device=device)
        
        current_x = now_pose[:, 0]
        current_y = now_pose[:, 1]
        current_yaw = now_pose[:, 2]
        waypoint0_x = waypoints[:, 0, 0]
        waypoint0_y = waypoints[:, 0, 1]
        delta_x_initial = waypoint0_x - current_x
        delta_y_initial = waypoint0_y - current_y
        traj_initial_angle = torch.atan2(delta_y_initial, delta_x_initial)
        initial_angle_diff = torch.abs(current_yaw - traj_initial_angle)
        initial_angle_diff = torch.fmod(initial_angle_diff + self.PI, 2 * self.PI) - self.PI
        angle_smooth_loss = torch.mean(torch.abs(initial_angle_diff))
        
        total_smoothness_loss = epsilon * curvature_loss + zeta * velocity_change_loss + eta * angle_smooth_loss
        
        # 恐惧标签计算
        wp_ds_for_fear = torch.norm(waypoints[:, 1:, :2] - waypoints[:, :-1, :2], dim=2)
        goal_dists = torch.cumsum(wp_ds_for_fear, dim=1)
        
        # 修正：使用膨胀后的成本
        fear_loss_M_raw = inflated_oloss_M[:, 1:]
        
        mask_ahead_dist = (goal_dists <= ahead_dist).to(fear_loss_M_raw.dtype)
        floss_M_masked = fear_loss_M_raw * mask_ahead_dist
        
        fear_labels = torch.max(floss_M_masked, dim=1, keepdim=True)[0]
        fear_labels = (fear_labels > obstacle_thred).to(torch.float32)
        
        start_point_loss = torch.norm(waypoints[:, 0, :2], dim=1).mean()
        
        # 可视化（可选）
        # 可视化（可选）
        visualize = False
        if visualize:
            # 定义保存目录
            save_dir = "/opt/data/private/ros/trajPre/FIREDA/THUD_Robot/iPlanner/img"
            os.makedirs(save_dir, exist_ok=True)  # 确保目录存在

            # 检查目录是否可写
            if not os.access(save_dir, os.W_OK):
                print(f"ERROR: Directory {save_dir} is not writable!")
                # 尝试使用临时目录作为备选
                save_dir = "/tmp/visualizations/"
                os.makedirs(save_dir, exist_ok=True)
                print(f"Using temporary directory instead: {save_dir}")
            
            # 检查磁盘空间
            total, used, free = shutil.disk_usage("/")
            print(f"Disk space: Total: {total//(2**30)} GB, Used: {used//(2**30)} GB, Free: {free//(2**30)} GB")
            
            for b in range(min(batch_size, 5)):  # 只可视化前5个批次
                # 获取成本图
                cost_map = cost_grids[b].squeeze().cpu().numpy()
                
                # 获取全局轨迹点
                global_points = global_waypoints[b].cpu().detach().numpy()
                
                # 获取当前批次的地图参数
                map_origin_x = map_origins[b][0]
                map_origin_y = map_origins[b][1]
                map_resolution = map_resolutions[b]
                map_width = map_widths[b]
                map_height = map_heights[b]
                
                # 获取变换矩阵
                if transform_matrics is not None and len(transform_matrics) > b:
                    transform_matrix = transform_matrics[b].cpu().detach().numpy()
                else:
                    transform_matrix = np.eye(4)
                
                # 创建可视化图像
                fig = plt.figure(figsize=(12, 8))
                
                # 绘制成本图
                ax1 = fig.add_subplot(1, 2, 1)
                im1 = ax1.imshow(cost_map, cmap='viridis', origin='lower')
                plt.colorbar(im1, ax=ax1, label='Cost Value')
                ax1.set_title('Global Cost Map')
                
                # 将全局坐标转换为像素坐标
                pixel_x = (global_points[:, 0] - map_origin_x) / map_resolution
                pixel_y = (global_points[:, 1] - map_origin_y) / map_resolution
                
                # 添加边界检查
                pixel_x = np.clip(pixel_x, 0, map_width - 1)
                pixel_y = np.clip(pixel_y, 0, map_height - 1)
                
                # 绘制预测轨迹点
                ax1.scatter(pixel_x, pixel_y, c='red', s=30, label='Predicted Waypoints')
                
                # 连接预测轨迹点
                ax1.plot(pixel_x, pixel_y, 'r-', linewidth=2, alpha=0.7)
                
                # 标记预测轨迹起点和终点
                if len(pixel_x) > 0:
                    ax1.scatter(pixel_x[0], pixel_y[0], c='green', s=100, marker='o', label='Predicted Start')
                    ax1.scatter(pixel_x[-1], pixel_y[-1], c='blue', s=100, marker='*', label='Predicted Goal')
                
                # 获取并绘制GT轨迹
                # 获取并绘制GT轨迹
                if odom is not None:
                    # 确保odom是张量
                    if isinstance(odom, (list, tuple)):
                        gt_odom = torch.tensor(odom[b], device=device, dtype=torch.float32)
                    else:
                        gt_odom = odom[b]
                    
                    # 检查gt_odom的形状
                    print(f"gt_odom shape: {gt_odom.shape}")  # 调试信息
                    
                    # 确保gt_odom至少有2列
                    if gt_odom.dim() == 1:
                        # 如果是一维张量，转换为二维
                        gt_odom = gt_odom.unsqueeze(0)
                    
                    # 获取变换矩阵
                    if transform_matrics is not None and len(transform_matrics) > b:
                        transform_matrix = transform_matrics[b]  # 已经是张量
                    else:
                        transform_matrix = torch.eye(4, device=device)
                    
                    # 确保变换矩阵是张量
                    if not isinstance(transform_matrix, torch.Tensor):
                        transform_matrix = torch.tensor(transform_matrix, device=device, dtype=torch.float32)
                    
                    # 使用相同的变换矩阵将GT轨迹转换到全局坐标系
                    # 从gt_odom中提取x, y，并添加z=0和齐次坐标1
                    gt_points = gt_odom[:, :2]  # 取前两列：x, y
                    gt_points_z = torch.zeros(gt_points.shape[0], 1, device=device)  # z坐标设为0
                    gt_points_homogeneous = torch.cat([
                        gt_points, 
                        gt_points_z,
                        torch.ones(gt_points.shape[0], 1, device=device)
                    ], dim=1)  # 形状为 (n, 4)
                    
                    # 检查形状
                    print(f"gt_points_homogeneous shape: {gt_points_homogeneous.shape}")  # 应该是 (n, 4)
                    
                    # 确保变换矩阵和点都在同一个设备上，并且数据类型一致
                    transform_matrix = transform_matrix.to(device).to(torch.float32)
                    gt_points_homogeneous = gt_points_homogeneous.to(device).to(torch.float32)
                    
                    # 执行矩阵乘法
                    # 变换矩阵 (4,4) * 点矩阵 (4, n) -> (4, n)
                    gt_global_points = torch.matmul(transform_matrix, gt_points_homogeneous.t())
                    
                    # 转置得到 (n, 4)，然后取前两列（x, y）
                    gt_global_points = gt_global_points.t()[:, :2].cpu().detach().numpy()
                    
                    # 将GT坐标转换为像素坐标
                    gt_pixel_x = (gt_global_points[:, 0] - map_origin_x) / map_resolution
                    gt_pixel_y = (gt_global_points[:, 1] - map_origin_y) / map_resolution
                    
                    # 添加边界检查
                    gt_pixel_x = np.clip(gt_pixel_x, 0, map_width - 1)
                    gt_pixel_y = np.clip(gt_pixel_y, 0, map_height - 1)
                    
                    # 绘制GT轨迹点
                    ax1.scatter(gt_pixel_x, gt_pixel_y, c='cyan', s=20, label='GT Waypoints')
                    
                    # 连接GT轨迹点
                    ax1.plot(gt_pixel_x, gt_pixel_y, 'c-', linewidth=1.5, alpha=0.7)
                    
                    # 标记GT轨迹起点和终点
                    if len(gt_pixel_x) > 0:
                        ax1.scatter(gt_pixel_x[0], gt_pixel_y[0], c='yellow', s=80, marker='s', label='GT Start')
                        ax1.scatter(gt_pixel_x[-1], gt_pixel_y[-1], c='magenta', s=80, marker='d', label='GT Goal')
                    
                    # 添加航向角箭头（可选）
                    if gt_odom.shape[1] >= 3:  # 确保有航向角数据
                        # 获取航向角
                        gt_yaw = gt_odom[:, 2].cpu().detach().numpy()
                        
                        # 计算箭头方向
                        arrow_length = 5  # 箭头长度（像素）
                        arrow_dx = arrow_length * np.cos(gt_yaw)
                        arrow_dy = arrow_length * np.sin(gt_yaw)
                        
                        # 绘制箭头
                        for i in range(len(gt_pixel_x)):
                            ax1.arrow(
                                gt_pixel_x[i], gt_pixel_y[i],
                                arrow_dx[i], arrow_dy[i],
                                head_width=2, head_length=3, fc='cyan', ec='cyan'
                            )
                ax1.legend()
                ax1.set_title('Trajectory on Global Cost Map')
                
                # 绘制成本热力图
                ax2 = fig.add_subplot(1, 2, 2)
                im2 = ax2.imshow(cost_map, cmap='viridis', origin='lower')
                plt.colorbar(im2, ax=ax2, label='Cost Value')
                
                # 创建轨迹成本热力图
                traj_cost = np.zeros_like(cost_map)
                for i in range(len(pixel_x)):
                    x, y = int(pixel_x[i]), int(pixel_y[i])
                    if 0 <= x < map_width and 0 <= y < map_height:
                        # 在轨迹点周围创建高斯分布
                        for dx in range(-3, 4):
                            for dy in range(-3, 4):
                                nx, ny = x + dx, y + dy
                                if 0 <= nx < map_width and 0 <= ny < map_height:
                                    dist = np.sqrt(dx**2 + dy**2)
                                    weight = np.exp(-dist/2.0)
                                    # 确保索引在范围内
                                    if ny < traj_cost.shape[0] and nx < traj_cost.shape[1]:
                                        traj_cost[ny, nx] = max(traj_cost[ny, nx], weight)
                
                # 叠加轨迹热力图
                ax2.imshow(traj_cost, cmap='hot', alpha=0.5, origin='lower')
                ax2.set_title('Trajectory Cost Heatmap')
                
                # 保存图像
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                save_path = os.path.join(save_dir, f"global_traj_cost_{timestamp}_b{b}.png")
                
                try:
                    # 直接保存图像而不显示
                    fig.savefig(save_path)
                    print(f"Saved global trajectory visualization to: {save_path}")
                    
                    # 验证文件是否创建
                    if os.path.exists(save_path):
                        file_size = os.path.getsize(save_path)
                        print(f"File created successfully: {save_path} ({file_size} bytes)")
                    else:
                        print(f"ERROR: File not created: {save_path}")
                        
                        # 尝试保存到临时目录
                        temp_path = f"/tmp/global_traj_cost_{timestamp}_b{b}.png"
                        fig.savefig(temp_path)
                        print(f"Saved to temporary location: {temp_path}")
                        
                        if os.path.exists(temp_path):
                            file_size = os.path.getsize(temp_path)
                            print(f"Temporary file created: {temp_path} ({file_size} bytes)")
                        else:
                            print(f"ERROR: Failed to save even to temporary location")
                except Exception as e:
                    print(f"ERROR saving image: {e}")
                    import traceback
                    traceback.print_exc()
                    
                    # 尝试保存一个简单的测试图
                    try:
                        test_fig = plt.figure()
                        plt.plot([1, 2, 3], [1, 2, 3])
                        test_path = os.path.join(save_dir, "test_plot.png")
                        test_fig.savefig(test_path)
                        plt.close(test_fig)
                        print(f"Test plot saved to: {test_path}")
                        
                        if os.path.exists(test_path):
                            file_size = os.path.getsize(test_path)
                            print(f"Test file created: {test_path} ({file_size} bytes)")
                        else:
                            print(f"ERROR: Failed to save test plot")
                    except Exception as e2:
                        print(f"ERROR saving test plot: {e2}")
                
                # 关闭图形以释放内存
                plt.close(fig)
            
            # 打印Matplotlib版本信息
            print(f"Matplotlib version: {matplotlib.__version__}")
    
        # 计算总成本
        total_smoothness_loss = 0
        final_total_cost = (
            alpha * oloss +
            beta * hloss +
            gamma * mloss +
            gloss_weight * gloss +
            0.0035 * total_smoothness_loss +
            gloss_weight * start_point_loss
        )
        
        return final_total_cost, fear_labels,total_smoothness_loss
    
    
    # #####  TODO 这里是做了地图更新的。这段代码需要保留
    # def CostofTraj_lobby(self, trajic_future, lidar_map, waypoints, odom, goal, ahead_dist, now_pose, gt_cost_map_list,
    #                  alpha=5, beta=1.0, gamma=2.0, obstacle_thred=0.6, inflate_px=3,
    #                  epsilon=1.0, zeta=1.0, eta=1.0,
    #                  gloss_weight=10):

    #     batch_size, T_waypoints, _ = waypoints.shape
    #     device = waypoints.device

    #     odom = odom.to(device).to(torch.float32) if isinstance(odom, torch.Tensor) else odom
    #     goal = goal.to(device).to(torch.float32)
    #     now_pose = now_pose.to(device).to(torch.float32)
    #     trajic_future = trajic_future.to(device).to(torch.float32)
    #     waypoints = waypoints.to(device).to(torch.float32)

    #     oloss = torch.tensor(0.0, device=device)
    #     hloss = torch.tensor(0.0, device=device)
    #     gloss = torch.tensor(0.0, device=device)
    #     mloss = torch.tensor(0.0, device=device)
    #     total_smoothness_loss = torch.tensor(0.0, device=device)
    #     fear_labels = torch.zeros(batch_size, 1, device=device)

    #     first_batch_map_was_updated = self.map_and_update_map(trajic_future, now_pose, lidar_map)

    #     cost_grids = []
    #     for b in range(batch_size):
    #         prob_map = lidar_map[b].data
    #         obs_mask = (prob_map > obstacle_thred).astype(np.uint8)
    #         esdf = scipy.ndimage.distance_transform_edt(1 - obs_mask)
    #         normalized_cost = np.exp(-esdf / 2.0)
    #         cost_tensor = torch.tensor(normalized_cost, dtype=torch.float32, device=device).unsqueeze(0)
    #         cost_grids.append(cost_tensor)
    #     cost_grids = torch.stack(cost_grids, dim=0)

    #     norm_inds_list = []
    #     for i in range(batch_size):
    #         # Similarly, `is_global_coords` parameter removed
    #         norm_inds, valid_mask = lidar_map[i].Pos2Ind(waypoints[i, :, :2], now_pose[i])
    #         valid_norm_inds = norm_inds[valid_mask].unsqueeze(1)

    #         if valid_norm_inds.shape[0] < T_waypoints:
    #             pad_len = T_waypoints - valid_norm_inds.shape[0]
    #             padding = -2 * torch.ones(pad_len, 1, 2, dtype=torch.float32, device=device)
    #             valid_norm_inds = torch.cat((valid_norm_inds, padding), dim=0)
    #         norm_inds_list.append(valid_norm_inds)

    #     max_length = max(inds.shape[0] for inds in norm_inds_list)
    #     for i in range(len(norm_inds_list)):
    #         cur_len = norm_inds_list[i].shape[0]
    #         if cur_len < max_length:
    #             padding = -2 * torch.ones(max_length - cur_len, 1, 2, dtype=torch.float32, device=device)
    #             norm_inds_list[i] = torch.cat((norm_inds_list[i], padding), dim=0)

    #     norm_inds_waypoints = torch.stack(norm_inds_list, dim=0)

    #     map_H, map_W = (80, 80)
    #     pixel_offset_x = 2.0 / map_W
    #     pixel_offset_y = 2.0 / map_H
    #     radius_x = inflate_px * pixel_offset_x
    #     radius_y = inflate_px * pixel_offset_y

    #     offsets_np = np.array([
    #         [0.0, 0.0],
    #         [-radius_x, 0.0], [radius_x, 0.0],
    #         [0.0, -radius_y], [0.0, radius_y],
    #         [-radius_x, -radius_y], [-radius_x, radius_y],
    #         [radius_x, -radius_y], [radius_x, radius_y],
    #     ], dtype=np.float32)
    #     offsets = torch.tensor(offsets_np, device=device).unsqueeze(0).unsqueeze(0)

    #     grid = norm_inds_waypoints.squeeze(2).unsqueeze(2) + offsets
        
    #     B, cur_T, K, _ = grid.shape
    #     grid = grid.reshape(B, cur_T * K, 1, 2)

    #     oloss_M = F.grid_sample(cost_grids, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
    #     oloss_M = oloss_M.squeeze(1).squeeze(2).reshape(B, cur_T, K)
    #     oloss_M = torch.mean(oloss_M, dim=2)
    #     oloss = torch.mean(torch.sum(oloss_M, dim=1), dim=0)

    #     ground_array = np.zeros((80, 80), dtype=np.float32)
    #     ground_map = torch.from_numpy(ground_array.T).unsqueeze(0).float().to(device)
    #     ground_map = ground_map.expand(batch_size, -1, -1).unsqueeze(1).to(device)
    #     height_grids = ground_map

    #     hloss_M = F.grid_sample(height_grids, grid, mode='bilinear', padding_mode='border', align_corners=False)
    #     hloss_M = hloss_M.squeeze(1).squeeze(2).reshape(B, cur_T, K)
    #     hloss_M = torch.mean(hloss_M, dim=2)
    #     hloss_M = torch.abs(waypoints[:, :cur_T, 2] - hloss_M) 
    #     hloss = torch.mean(torch.sum(hloss_M, dim=1), dim=0)

    #     gloss = torch.norm(goal[:, :3] - waypoints[:, -1, :], dim=1)
    #     gloss = torch.mean(gloss)  # 去掉log

    #     desired_wp = self.opt.TrajGeneratorFromPFreeRot(goal[:, None, 0:3], step=1.0 / (T_waypoints - 1))
    #     desired_ds = torch.norm(desired_wp[:, 1:, :] - desired_wp[:, :-1, :], dim=2)
    #     wp_ds = torch.norm(waypoints[:, 1:, :] - waypoints[:, :-1, :], dim=2)
    #     mloss = torch.abs(desired_ds - wp_ds)
    #     mloss = torch.sum(mloss, dim=1)
    #     mloss = torch.mean(mloss, dim=0)

    #     if T_waypoints >= 3:
    #         p0 = waypoints[:, :-2, :2]
    #         p1 = waypoints[:, 1:-1, :2]
    #         p2 = waypoints[:, 2:, :2]
    #         vec1 = p1 - p0
    #         vec2 = p2 - p1
    #         vec1_norm = vec1 / (torch.norm(vec1, dim=2, keepdim=True) + 1e-6)
    #         vec2_norm = vec2 / (torch.norm(vec2, dim=2, keepdim=True) + 1e-6)
    #         cos_angles = torch.sum(vec1_norm * vec2_norm, dim=2)
    #         angle_diffs = torch.acos(torch.clamp(cos_angles, -1.0, 1.0))
    #         curvature_loss = angle_diffs.mean()
    #     else:
    #         curvature_loss = torch.tensor(0.0, device=device)

    #     velocities = waypoints[:, 1:, :2] - waypoints[:, :-1, :2]
    #     speed = torch.norm(velocities, dim=2)
    #     if T_waypoints > 2:
    #         velocity_change = speed[:, 1:] - speed[:, :-1]
    #         velocity_change_loss = torch.mean(torch.abs(velocity_change))
    #     else:
    #         velocity_change_loss = torch.tensor(0.0, device=device)

    #     current_x = now_pose[:, 0]
    #     current_y = now_pose[:, 1]
    #     current_yaw = now_pose[:, 2]
    #     waypoint0_x = waypoints[:, 0, 0]
    #     waypoint0_y = waypoints[:, 0, 1]
    #     delta_x_initial = waypoint0_x - current_x
    #     delta_y_initial = waypoint0_y - current_y
    #     traj_initial_angle = torch.atan2(delta_y_initial, delta_x_initial)
    #     initial_angle_diff = torch.abs(current_yaw - traj_initial_angle)
    #     initial_angle_diff = torch.fmod(initial_angle_diff + self.PI, 2 * self.PI) - self.PI
    #     angle_smooth_loss = torch.mean(torch.abs(initial_angle_diff))

    #     total_smoothness_loss = epsilon * curvature_loss + zeta * velocity_change_loss + eta * angle_smooth_loss

    #     wp_ds_for_fear = torch.norm(waypoints[:, 1:, :2] - waypoints[:, :-1, :2], dim=2)
    #     goal_dists = torch.cumsum(wp_ds_for_fear, dim=1)
        
    #     floss_M = oloss_M[:, 1:]
        
    #     mask_ahead_dist = (goal_dists <= ahead_dist).to(floss_M.dtype)
    #     floss_M_masked = floss_M * mask_ahead_dist

    #     fear_labels = torch.max(floss_M_masked, dim=1, keepdim=True)[0]
    #     fear_labels = (fear_labels > obstacle_thred).to(torch.float32)

    #     start_point_loss = torch.norm(waypoints[:, 0, :2], dim=1).mean()

    #     final_total_cost = (
    #         alpha * oloss +
    #         beta * hloss +
    #         gamma * mloss +
    #         gloss_weight * gloss +
    #         0.0035 * total_smoothness_loss +
    #         gloss_weight * start_point_loss
    #     )

    #     return final_total_cost, fear_labels


    
    


    # def CostofTraj_lobby(self, trajic_future, lidar_map, waypoints, odom, goal, ahead_dist, now_pose, gt_cost_map, future_odom_list,
    #                         alpha=5, beta=1.0, gamma=2.0, obstacle_thred=0.2, inflate_px=3,
    #                         epsilon=1.0, zeta=1.0, eta=1.0, gloss_weight=15):

    #         batch_size, T_waypoints, _ = waypoints.shape
    #         device = waypoints.device

    #         odom = odom.to(device).float() if isinstance(odom, torch.Tensor) else odom
    #         goal = goal.to(device).float()
    #         now_pose = now_pose.to(device).float()
    #         trajic_future = trajic_future.to(device).float()
    #         waypoints = waypoints.to(device).float()

    #         oloss = torch.tensor(0.0, device=device)
    #         hloss = torch.tensor(0.0, device=device)
    #         gloss = torch.tensor(0.0, device=device)
    #         mloss = torch.tensor(0.0, device=device)
    #         total_smoothness_loss = torch.tensor(0.0, device=device)
    #         fear_labels = torch.zeros(batch_size, 1, device=device)

    #         self.map_and_update_map(trajic_future, now_pose, lidar_map)

    #         map_H, map_W = 80, 80
    #         pixel_offset_x = 2.0 / map_W
    #         pixel_offset_y = 2.0 / map_H
    #         radius_x = inflate_px * pixel_offset_x
    #         radius_y = inflate_px * pixel_offset_y

    #         offsets = torch.tensor([
    #             [0.0, 0.0],
    #             [-radius_x, 0.0], [radius_x, 0.0],
    #             [0.0, -radius_y], [0.0, radius_y],
    #             [-radius_x, -radius_y], [-radius_x, radius_y],
    #             [radius_x, -radius_y], [radius_x, radius_y],
    #         ], dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)  # [1,1,9,2]

    #         oloss_all = []
    #         fear_scores = []
    #         for b in range(batch_size):
    #             oloss_per_batch = []
    #             fear_per_batch = []

    #             for t in range(T_waypoints):
    #                 cost_map_obj = gt_cost_map[b][t]
    #                 obs_mask_np = (cost_map_obj.data > obstacle_thred).astype(np.uint8)
    #                 esdf_np = scipy.ndimage.distance_transform_edt(1 - obs_mask_np)
    #                 soft_cost_np = np.exp(-esdf_np / 2.0)
    #                 cost_tensor = torch.tensor(soft_cost_np, dtype=torch.float32, device=device).unsqueeze(0)

    #                 # --- 关键坐标转换 ---
    #                 # 1. 提取当前轨迹点局部坐标（相对于 now_pose）
    #                 local_wp = waypoints[b, t, :2].detach().cpu().numpy()

    #                 # 2. 提取对应位姿
    #                 now_pose_b = now_pose[b].detach().cpu().numpy()  # now_pose[b]: [x, y, yaw]
    #                 # import pdb
    #                 # pdb.set_trace()  # 调试用，检查 now_pose_b 的值
    #                 try:
    #                     odom_bt = future_odom_list[b, t].detach().cpu().numpy()    # odom[b,t]: [x, y, yaw]
    #                 except Exception as e:
    #                     print(f"Error accessing future_odom_list at batch {b}, time {t}: {e}")
    #                     import pdb
    #                     pdb.set_trace()
    #                 # 3. 局部(相对now_pose) -> 世界
    #                 world_wp = self.local_to_world(local_wp, now_pose_b)

    #                 # 4. 世界 -> odom[t]局部坐标系
    #                 transformed_wp = self.world_to_local(world_wp, odom_bt)

    #                 # 转回 tensor
    #                 transformed_wp_tensor = torch.tensor(transformed_wp, dtype=torch.float32, device=device).unsqueeze(0)

    #                 norm_ind, valid_mask = cost_map_obj.Pos2Ind(transformed_wp_tensor, future_odom_list[b, t])
    #                 if not valid_mask[0]:
    #                     norm_ind = torch.tensor([[-2.0, -2.0]], device=device)

    #                 grid = norm_ind.unsqueeze(1) + offsets  # [1,9,2]
    #                 grid = grid.unsqueeze(1)  # [1,1,9,2]

    #                 cost_tensor_4d = cost_tensor.unsqueeze(0)  # [1,1,H,W]
    #                 grid = grid.squeeze(1)  # [1,9,2]

    #                 sampled_cost = F.grid_sample(cost_tensor_4d, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
    #                 sampled_cost = sampled_cost.view(-1)

    #                 mean_cost = sampled_cost.sum()
    #                 max_cost = sampled_cost.max()

    #                 oloss_per_batch.append(mean_cost)
    #                 fear_per_batch.append(max_cost)
    #             oloss_all.append(torch.stack(oloss_per_batch).sum())
    #             fear_scores.append(torch.stack(fear_per_batch).max())
    #         oloss = torch.stack(oloss_all).mean()
    #         fear_labels = (torch.stack(fear_scores).unsqueeze(1) > obstacle_thred).float()

    #         ground_map = torch.zeros((batch_size, 1, map_H, map_W), dtype=torch.float32, device=device)
    #         hloss_all = []
    #         for b in range(batch_size):
    #             hloss_per_batch = []
    #             for t in range(T_waypoints):
    #                 waypoint_t = waypoints[b, t, :2].unsqueeze(0)
    #                 now_pose_bt = future_odom_list[b, t] if odom.dim() == 3 else now_pose[b]  # Use gt pose at time t
    #                 norm_ind, valid_mask = gt_cost_map[b][t].Pos2Ind(waypoint_t, now_pose_bt)
    #                 if not valid_mask[0]:
    #                     norm_ind = torch.tensor([[-2.0, -2.0]], device=device)
    #                 grid = norm_ind.unsqueeze(1) + offsets  # [1,9,2]
    #                 sampled_height = F.grid_sample(ground_map[b:b+1], grid, mode='bilinear', padding_mode='border', align_corners=False)
    #                 mean_height = sampled_height.view(-1).mean()
    #                 hloss_per_batch.append(mean_height)
    #             hloss_traj = torch.abs(waypoints[b, :, 2] - torch.tensor(hloss_per_batch, device=device))
    #             hloss_all.append(hloss_traj.sum())
    #         hloss = torch.stack(hloss_all).mean()

    #         gloss_dist = torch.norm(goal[:, :3] - waypoints[:, -1, :], dim=1)
    #         gloss = torch.mean(torch.log(gloss_dist + 1.0))

    #         desired_wp = self.opt.TrajGeneratorFromPFreeRot(goal[:, None, 0:3], step=1.0 / (T_waypoints - 1))
    #         desired_ds = torch.norm(desired_wp[:, 1:, :] - desired_wp[:, :-1, :], dim=2)
    #         wp_ds = torch.norm(waypoints[:, 1:, :] - waypoints[:, :-1, :], dim=2)
    #         mloss = torch.abs(desired_ds - wp_ds).sum(dim=1).mean()

    #         if T_waypoints >= 3:
    #             p0 = waypoints[:, :-2, :2]
    #             p1 = waypoints[:, 1:-1, :2]
    #             p2 = waypoints[:, 2:, :2]
    #             vec1 = p1 - p0
    #             vec2 = p2 - p1
    #             vec1_norm = vec1 / (vec1.norm(dim=2, keepdim=True) + 1e-6)
    #             vec2_norm = vec2 / (vec2.norm(dim=2, keepdim=True) + 1e-6)
    #             cos_angles = (vec1_norm * vec2_norm).sum(dim=2)
    #             angle_diffs = torch.acos(torch.clamp(cos_angles, -1.0, 1.0))
    #             curvature_loss = angle_diffs.mean()
    #         else:
    #             curvature_loss = torch.tensor(0.0, device=device)

    #         speed = waypoints[:, 1:, :2] - waypoints[:, :-1, :2]
    #         speed_norm = speed.norm(dim=2)
    #         if T_waypoints > 2:
    #             velocity_change = speed_norm[:, 1:] - speed_norm[:, :-1]
    #             velocity_change_loss = velocity_change.abs().mean()
    #         else:
    #             velocity_change_loss = torch.tensor(0.0, device=device)

    #         delta = waypoints[:, 0, :2] - now_pose[:, :2]
    #         traj_init_angle = torch.atan2(delta[:, 1], delta[:, 0])
    #         angle_diff = now_pose[:, 2] - traj_init_angle
    #         angle_diff = (angle_diff + self.PI) % (2 * self.PI) - self.PI
    #         angle_smooth_loss = angle_diff.abs().mean()

    #         total_smoothness_loss = epsilon * curvature_loss + zeta * velocity_change_loss + eta * angle_smooth_loss

    #         fear_loss = torch.stack(oloss_all).mean()

    #         start_point_loss = torch.norm(waypoints[:, 0, :2], dim=1).mean()

    #         if batch_size == 1:
    #             with torch.no_grad():
    #                 self.visualize_costmaps_with_traj(
    #                     gt_costmap_list=gt_cost_map[0],
    #                     waypoints=waypoints[0].detach().cpu().numpy(),
    #                     save_dir="./costmap_viz",
    #                     now_pose=now_pose[0].detach().cpu().numpy() if now_pose is not None else None
    #                 )

    #         final_total_cost = (
    #             alpha * oloss +
    #             beta * hloss +
    #             gamma * mloss +
    #             gloss_weight * gloss +
    #             0.0035 * total_smoothness_loss +
    #             gloss_weight * start_point_loss
    #         )

    #         return final_total_cost, fear_labels










    # def CostofTraj(self, trajic_future, valid_mask, waypoints, odom, goal, ahead_dist, esdf_map=None, alpha=0.5, beta=1.0, gamma=2.0, delta=5.0, obstacle_thred=0.5):
    #     total_cost = torch.tensor(0.0, device=waypoints.device)
    #     fear_labels = torch.tensor(0.0, device=waypoints.device)

    #     if esdf_map is not None:
    #         world_ps = waypoints

    #         # 获取 norm_inds，确保它是 PyTorch 张量
    #         norm_inds, _ = esdf_map.Pos2Ind(world_ps)

    #         # 如果 norm_inds 是 NumPy 数组，则转换为 PyTorch 张量
    #         if isinstance(norm_inds, np.ndarray):
    #             norm_inds = torch.from_numpy(norm_inds).float().to(waypoints.device)
    #         # 如果 norm_inds 已经是 PyTorch 张量，确保它在正确的设备上
    #         elif isinstance(norm_inds, torch.Tensor):
    #             norm_inds = norm_inds.to(waypoints.device)
    #         else:
    #             raise TypeError(f"norm_inds must be a NumPy array or PyTorch tensor, but got {type(norm_inds)}")

    #         # 确保 norm_inds 的数据类型是 float32
    #         norm_inds = norm_inds.float()

    #         # Obstacle Cost
    #         # 将 NumPy 数组转换为 PyTorch 张量，并确保在正确的设备上
    #         cost_grid = torch.from_numpy(np.expand_dims(esdf_map.esdf_map.T, axis=(0, 1))).float().to(waypoints.device)
    #         oloss_M = F.grid_sample(cost_grid, norm_inds[:, None, :, :], mode='bicubic', padding_mode='border', align_corners=False).squeeze(1).squeeze(1)
    #         oloss = torch.mean(torch.sum(oloss_M, axis=1))

    #         # Terrain Height Loss
    #         # 将 NumPy 数组转换为 PyTorch 张量，并确保在正确的设备上
    #         height_grid = torch.from_numpy(esdf_map.ground_array.T).float().unsqueeze(0).unsqueeze(0).to(waypoints.device)
    #         hloss_M = F.grid_sample(height_grid, norm_inds[:, None, :, :], mode='bicubic', padding_mode='border', align_corners=False).squeeze(1).squeeze(1)
            
    #         hloss_M = torch.abs(waypoints[ :, 2] - hloss_M)
    #         hloss = torch.mean(torch.sum(hloss_M, axis=1))

    #         # Goal Cost
    #         gloss = torch.norm(goal[ :3] - waypoints[-1, :], dim=0)
    #         gloss = torch.mean(torch.log(gloss + 1.0))

    #         # Motion Loss
            
    #         desired_wp = self.opt.TrajGeneratorFromPFreeRot(goal[ None,0:3], step=1.0 / (len(waypoints) - 1))
    #         desired_ds = torch.norm(desired_wp[ 1:len(waypoints), :] - desired_wp[ 0:len(waypoints)-1, :], dim=1)
    #         wp_ds = torch.norm(waypoints[ 1:, :] - waypoints[ :-1, :], dim=1)
    #         mloss = torch.abs(desired_ds - wp_ds)
    #         mloss = torch.sum(mloss, axis=0)
    #         mloss = torch.mean(mloss)

    #         # Fear Labels
    #         goal_dists = torch.cumsum(wp_ds, dim=0, dtype=wp_ds.dtype)
    #         floss_M = torch.clone(oloss_M)[0,1:]
    #         floss_M[goal_dists > ahead_dist] = 0.0
    #         fear_labels = torch.max(floss_M, 0, keepdim=True)[0]
    #         fear_labels = (fear_labels > obstacle_thred).to(torch.float32)

    #         # Total Cost
    #         total_cost = alpha * oloss + beta * hloss + gamma * mloss + delta * gloss

    #     return total_cost, fear_labels