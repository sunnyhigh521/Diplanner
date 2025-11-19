import os
import PIL
import torch
import numpy as np
import pypose as pp
from PIL import Image
from pathlib import Path
import json
from random import sample
from operator import itemgetter
from torch.utils.data import Dataset, DataLoader
from trajicPre.GS_module import GumbelSocialTransformer
from trajicPre.GS_module.crowd_nav_interface_parallel import CrowdNavPredInterfaceMultiEnv
from scipy.spatial.transform import Rotation as R
from .LidarCostMap import LidarCostMap 
from torchvision import transforms
import cv2
import numpy as np
import numpy.matlib 
# other imports like torch, os, etc.
torch.set_default_dtype(torch.float32)
import csv
# 确保这些路径和参数与你的实际设置匹配
load_path = "/opt/data/private/ros/trajPre/markov/trajicPre/results/100-gumbel_social_transformer-faster_lstm-lr_0.001-init_temp_0.5-edge_head_0-ebd_64-snl_1-snh_8-seed_1000_rand/sj"
device = torch.device("cuda")
import LidarCostUtil # Re-import LidarCostUtil
import pickle
checkpointdir = '/opt/data/private/ros/trajPre/CrowdNav_Prediction_AttnGraph/gst_updated/results/100-gumbel_social_transformer-faster_lstm-lr_0.001-init_temp_0.5-edge_head_0-ebd_64-snl_1-snh_8-seed_1000_rand/sj/checkpoint/args.pickle'
with open(checkpointdir,'rb') as f:
    args = pickle.load(f)
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
import json
from random import sample
from operator import itemgetter
import math


def is_in_costmap(odom, costmap_meta):
    x, y = odom[0], odom[1]
    origin_x = costmap_meta['origin']['position']['x']
    origin_y = costmap_meta['origin']['position']['y']
    resolution = costmap_meta['resolution']
    width = costmap_meta['width']
    height = costmap_meta['height']
    max_x = origin_x + resolution * width
    max_y = origin_y + resolution * height
    return origin_x <= x < max_x and origin_y <= y < max_y


def parse_costmap_meta(costmap_path):
    meta = {}
    with open(costmap_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith("info.resolution:"):
                meta['resolution'] = float(line.strip().split(":")[-1])
            elif line.startswith("info.width:"):
                meta['width'] = int(line.strip().split(":")[-1])
            elif line.startswith("info.height:"):
                meta['height'] = int(line.strip().split(":")[-1])
            elif line.startswith("info.origin.position.x:"):
                meta.setdefault('origin', {})['position'] = meta.get('origin', {}).get('position', {})
                meta['origin']['position']['x'] = float(line.strip().split(":")[-1])
            elif line.startswith("info.origin.position.y:"):
                meta['origin']['position']['y'] = float(line.strip().split(":")[-1])
    return meta


def get_corresponding_costmap(odom, odom_timestamp, costmap_timestamps, costmap_dir):
    odom_timestamp = int(odom_timestamp)
    # 1. 找到小于等于里程计时间戳的最大成本图时间戳 T1
    T1 = None
    for ts in costmap_timestamps:
        if ts <= odom_timestamp:
            T1 = ts
        else:
            break  # 时间戳列表是升序的，一旦超过就退出

    # 如果找不到任何小于等于的，直接返回 None
    if T1 is None:
        return None

    # 2. 找 T1 的下一个时间戳 T2（如果存在）
    idx_T1 = costmap_timestamps.index(T1)
    T2 = costmap_timestamps[idx_T1 + 1] if idx_T1 + 1 < len(costmap_timestamps) else None

    # 3. 检查 T1 是否覆盖当前位置
    costmap_path_T1 = os.path.join(costmap_dir, f"local_costmaps/{T1}.txt")
    if os.path.exists(costmap_path_T1):
        meta_T1 = parse_costmap_meta(costmap_path_T1)
        if is_in_costmap(odom, meta_T1):
            return T1

    # 4. 如果 T1 不行，检查 T2 是否覆盖当前位置
    if T2 is not None:
        costmap_path_T2 = os.path.join(costmap_dir, f"local_costmaps/{T2}.txt")
        if os.path.exists(costmap_path_T2):
            meta_T2 = parse_costmap_meta(costmap_path_T2)
            if is_in_costmap(odom, meta_T2):
                return T2

    # 5. 都不满足
    return T1


class MultiEpochsDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._DataLoader__initialized = False
        self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()
        self.predictor = CrowdNavPredInterfaceMultiEnv(load_path=load_path, device=device, config = args, num_env = 1)


    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler(object):
    """ Sampler that repeats forever.
    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)

# --- Helper function: Converts raw LiDAR ranges to an occupancy grid ---
def create_occupancy_grid(ranges, angle_min, angle_increment, 
                          output_height, output_width, 
                          grid_resolution_override=None, grid_size_meters_override=None):
    """
    Converts raw LiDAR ranges data into a specified pixel dimension occupancy grid map.
    
    Args:
        ranges (np.array): Array of LiDAR distance measurements.
        angle_min (float): Minimum scan angle (radians).
        angle_increment (float): Angular increment (radians).
        output_height (int): Desired height of the output grid map (pixels).
        output_width (int): Desired width of the output grid map (pixels).
        grid_resolution_override (float, optional): If specified, forces this resolution.
        grid_size_meters_override (float, optional): If specified, forces this map size (e.g., 30.0 for a 30x30m map).
    
    Returns:
        np.array: Generated occupancy grid map (output_height, output_width) with values between 0-1.
    """
    num_ranges = len(ranges)
    angles = np.arange(num_ranges) * angle_increment + angle_min

    # Auto-calculate or use overridden map parameters
    if grid_size_meters_override is None:
        # A common assumption is to cover a circular area with a radius of e.g., 15m.
        # So a square map of 30x30 meters might be suitable.
        grid_size_meters = 30.0 
    else:
        grid_size_meters = grid_size_meters_override

    if grid_resolution_override is None:
        # Calculate theoretical resolution based on desired output dimensions
        # Adjusted for typical view, you might need to tune this constant (1.5)
        grid_resolution = grid_size_meters / max(output_height, output_width) * 1.5 
    else:
        grid_resolution = grid_resolution_override

    # Initial grid dimensions (can be rectangular if grid_size_meters is different for x/y)
    initial_grid_pixels_height = int(grid_size_meters / grid_resolution)
    initial_grid_pixels_width = int(grid_size_meters / grid_resolution) 
    
    occupancy_grid = np.zeros((initial_grid_pixels_height, initial_grid_pixels_width), dtype=np.float32)

    # Convert polar to Cartesian coordinates
    # Filter out invalid distances (inf, NaN) or values beyond max effective range
    # 0.01 is to filter out very close noise points. grid_size_meters / 2.0 corresponds to half map width/height.
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

    # Resize to target dimensions if necessary
    if occupancy_grid.shape[0] != output_height or occupancy_grid.shape[1] != output_width:
        occupancy_grid = cv2.resize(occupancy_grid, (output_width, output_height), 
                                    interpolation=cv2.INTER_LINEAR) # INTER_LINEAR or INTER_AREA

    return occupancy_grid

# --- PlannerData Class Definition ---
class PlannerData(Dataset):
    def __init__(self, root, max_episode, goal_step, train, ratio=0.9, max_depth=10.0, sensorOffsetX=0.0, transform=None, is_robot=True):
        super().__init__()
        self.LASER_CLIP = 30
        self.mu_ped_pos = -0.0001 
        self.std_ped_pos = 0.0391 
        self.mu_scan = 5.3850 
        self.std_scan = 4.2161 
        self.transform = transform
        self.is_robot = is_robot
        self.max_depth = max_depth
        img_path = root + "/final_image"
        img_filename_list = [str(s) for s in Path(img_path).rglob('*.npy')]
        img_filename_list.sort(key=lambda x: int(x.split("/")[-1][0:-4]))

        odom_path = os.path.join(root, "matching_odometry_data.txt")
        odom_list = []
        time_stamp_list = []
        with open(odom_path) as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip("\n").split(", ")
                time_stamp_list.append(line[0])
                odom = [float(element) for element in line[1:]]
                odom_list.append(odom)

        # --- LiDAR Raw Scan Data Loading ---
        self.RAW_LIDAR_SCAN_PATH = "/opt/data/private/ros/trajPre/FIREDA/THUD_Robot/iPlanner/iplanner/Final_data/TrainingData/lobby4_scan/all_interpolated_laser_scan_data.txt"
        self.raw_lidar_scan_lines = []
        try:
            with open(self.RAW_LIDAR_SCAN_PATH, 'r') as f:
                self.raw_lidar_scan_lines = f.readlines()
            if len(self.raw_lidar_scan_lines) != len(odom_list):
                print(f"Warning: Number of raw LiDAR scan lines ({len(self.raw_lidar_scan_lines)}) does not match odometry/image lines ({len(odom_list)}). Ensure data alignment.")
        except FileNotFoundError:
            print(f"Error: Raw LiDAR scan data file not found at {self.RAW_LIDAR_SCAN_PATH}")
            self.raw_lidar_scan_lines = []

        # --- Costmap related logic (retained as per request) ---
        costmap_timestamps_file = os.path.join(root, "final_cloud2/local_costmap_timestamps.txt")
        costmap_dir = os.path.join(root, "final_cloud2")
        self.costmap_base_path = os.path.join(costmap_dir, "test_cost_map6_local") # Base path for costmap files

        self.corresponding_timestamps = {}
        costmap_timestamps_list_sorted = [] # Renamed to avoid conflict with `costmap_timestamps` variable in loop
        if os.path.exists(costmap_timestamps_file):
            with open(costmap_timestamps_file, 'r') as f:
                costmap_timestamps_list_sorted = [int(line.strip()) for line in f.readlines()]
            costmap_timestamps_list_sorted.sort()

            for i, odom_timestamp in enumerate(time_stamp_list):
                odom = odom_list[i]
                selected_ts = get_corresponding_costmap(odom, odom_timestamp, costmap_timestamps_list_sorted, costmap_dir)
                if selected_ts is not None:
                    self.corresponding_timestamps[odom_timestamp] = selected_ts
        else:
            print(f"Warning: Costmap timestamps file not found at {costmap_timestamps_file}. Costmap-related logic might be incomplete.")
            self.corresponding_timestamps = {}

        # --- NEW: Load transform matrices from CSV file ---
        transforms_file = os.path.join(root, "/opt/data/private/ros/trajPre/FIREDA/THUD_Robot/iPlanner/iplanner/Final_data/TrainingData/lobby5_scan/transforms.csv")
        self.transform_matrices = {}
        if os.path.exists(transforms_file):
            try:
                with open(transforms_file, 'r') as f:
                    # 读取标题行
                    header = f.readline().strip().split(',')
                    # 手动处理每一行
                    for line_num, line in enumerate(f):
                        # 分割前三个字段
                        parts = line.strip().split(',', 3)  # 只分割前三个逗号
                        if len(parts) < 4:
                            print(f"Skipping invalid line {line_num}: {line}")
                            continue
                        
                        # 解析前三个字段
                        try:
                            timestamp_sec = parts[0]
                            timestamp_nsec = int(parts[1])
                            actual_timestamp_sec = parts[2]
                            matrix_str = parts[3]
                        except Exception as e:
                            print(f"Error parsing line {line_num}: {e}")
                            continue
                        
                        # 解析矩阵字符串
                        try:
                            # 分割行
                            rows = matrix_str.split(';')
                            # 分割每行中的元素
                            matrix_data = []
                            for r in rows:
                                # 分割每个元素
                                elements = r.split(',')
                                # 移除空元素
                                elements = [e.strip() for e in elements if e.strip()]
                                if elements:
                                    matrix_data.append(list(map(float, elements)))

                                
                            
                            # 转换为numpy数组
                            matrix = np.array(matrix_data)
                            
                            # 确保是4x4矩阵
                            if matrix.shape != (4, 4):
                                # 尝试修正
                                if matrix.size == 16:
                                    matrix = matrix.reshape(4, 4)
                                else:
                                    print(f"Error: Matrix has {matrix.size} elements, expected 16. Using identity matrix.")
                                    matrix = np.eye(4)
                        except Exception as e:
                            print(f"Error parsing matrix string on line {line_num}: {e}. Using identity matrix.")
                            matrix = np.eye(4)
                        
                        # 存储矩阵
                        self.transform_matrices[timestamp_nsec] = matrix
                        
                print(f"Loaded {len(self.transform_matrices)} transform matrices from {transforms_file}")
            except Exception as e:
                print(f"Error loading transform matrices: {e}")
        else:
            print(f"Warning: Transform matrices file not found at {transforms_file}")
        N = len(odom_list)
        json_path = os.path.join(root,"trajic.json")
        with open(json_path,'r',encoding="UTF-8") as file:
            trajic_json_data = json.load(file)
        trajics_by_timestamp = trajic_json_data

        self.img_filename = []
        self.odom_list    = []
        self.goal_list    = []
        self.lidar_list   = [] # This will store paths to the CURRENT costmap files
        self.gt_costmap_list = [] # NEW: To store paths for future 5 ground truth costmaps

        # NEW: Lists for transform matrices
        self.current_transform_list = []  # For current transform matrices
        self.future_transform_list = []   # For future 5 transform matrices (each element is a list of 5 matrices)

        self.past_trajic_list = []
        self.future_trajic_list = []
        self.ground_truth_list = []
        self.valid_mask_list = []
        self.people_vx_list=[]
        self.people_vy_list=[]
        max_num_ped = 7
        self.scan_list = []
        self.future_odom = []

        # Pre-load all costmap paths into a dictionary for efficient lookup
        all_costmap_paths_dict = {}
        if os.path.exists(self.costmap_base_path):
            for ts in costmap_timestamps_list_sorted: # Use the sorted list for iterating
                costmap_file = os.path.join(self.costmap_base_path, f"{ts}.txt")
                if os.path.exists(costmap_file):
                    all_costmap_paths_dict[str(ts)] = costmap_file

        for ahead in range(79, max_episode+1, goal_step+10):
            for i in range(N):
                odom = odom_list[i]
                gt_full = odom_list[i:min(i + ahead + 1, len(odom_list))]
                if len(gt_full) >= 5:
                    length = len(gt_full) - 1
                    indices = [
                        0,
                        length // 4,
                        length // 2,
                        (3 * length) // 4,
                        length
                    ]
                    try:
                        gt = [gt_full[j] for j in indices]
                    except Exception as e:
                        import pdb
                        pdb.set_trace()
                        print(e)
                else:
                    gt = gt_full + [gt_full[-1]] * (5 - len(gt_full))
                
                cart_pos = np.array([odom[0], odom[1]])
                cart_th = odom[2]

                goal_abs = np.array(odom_list[min(i + ahead, N - 1)])[0:2]

                goal_rel = goal_abs - cart_pos
                gt = np.array(gt)
                first_point = gt[0]
                gt_rel = gt[:, :2] - first_point[:2]
                
                R_inv = np.array([
                    [np.cos(cart_th), np.sin(cart_th)],
                    [-np.sin(cart_th), np.cos(cart_th)]
                ])

                goal = np.matmul(R_inv, goal_rel)
                goal = np.array([goal[0], goal[1], 0])
                gt_rel = np.array(gt_rel)
                R_inv_expanded = R_inv[np.newaxis, :, :]

                gt_rel_expanded = gt_rel[:, :, np.newaxis]
                gt_rotated = np.matmul(R_inv_expanded, gt_rel_expanded)
                gt = gt_rotated.squeeze(axis=2)
            
                try:
                    self.img_filename.append(img_filename_list[i])
                    self.odom_list.append(torch.tensor(odom, dtype=torch.float32))
                    self.goal_list.append(torch.tensor(goal, dtype=torch.float32))
                    self.scan_list.append(self.raw_lidar_scan_lines[i])
                    
                    # --- Current Costmap Path Appending ---
                    current_odom_timestamp_str = time_stamp_list[i]
                    current_costmap_ts_val = None  # 初始化变量
                    if current_odom_timestamp_str in self.corresponding_timestamps:
                        current_costmap_ts_val = self.corresponding_timestamps[current_odom_timestamp_str]
                        current_costmap_ts_str = str(current_costmap_ts_val)
                        if current_costmap_ts_str in all_costmap_paths_dict:
                            self.lidar_list.append(all_costmap_paths_dict[current_costmap_ts_str])
                        else:
                            print(f"Warning: Current costmap file not found for timestamp {current_costmap_ts_str} (derived from {current_odom_timestamp_str}). Appending a dummy path.")
                            self.lidar_list.append("")
                    else:
                        print(f"Warning: No corresponding costmap found for timestamp {current_odom_timestamp_str}. Appending a dummy path for current costmap.")
                        self.lidar_list.append("")
                    # --- End Current Costmap Path Appending ---

                    # --- NEW: Get transform matrix for current timestamp ---
                    current_transform = np.eye(4)  # Default identity matrix
                    
                    # 检查 current_costmap_ts_val 是否有效
                    if current_costmap_ts_val is not None and current_costmap_ts_val in self.transform_matrices:
                        current_transform = self.transform_matrices[current_costmap_ts_val]
                    else:
                        # 如果 current_costmap_ts_val 为 None 或不在字典中，使用单位矩阵
                        if current_costmap_ts_val is None:
                            print(f"Warning: No costmap timestamp for odom timestamp {current_odom_timestamp_str}. Using identity matrix.")
                        else:
                            print(f"Warning: Transform matrix not found for timestamp {current_costmap_ts_val}. Using identity matrix.")
                    self.current_transform_list.append(current_transform)
                    # --- End current transform matrix ---

                    # --- NEW: Get paths and transform matrices for future 5 ground truth costmaps (GtCostMap) ---
                    future_costmaps_paths_for_sample = []
                    future_transforms_for_sample = []  # NEW: List for future transform matrices
                    last_valid_path = None
                    last_valid_transform = np.eye(4)  # Default identity matrix

                    # 使用已经计算好的 indices，统一对 costmap 也用这些 index
                    future_odom_list = []
                    gt_indices = [i + j for j in indices]
                    last_valid_odom = None  # 新增：记录最后一个有效的odom
                    for idx in gt_indices:
                        if idx < N:
                            future_odom = odom_list[idx]
                            odom_tensor = torch.tensor(future_odom, dtype=torch.float32)
                            future_odom_list.append(odom_tensor)
                            last_valid_odom = odom_tensor  # 更新最后一个有效odom

                            future_odom_timestamp_str = time_stamp_list[idx]
                            # if future_odom_timestamp_str in self.corresponding_timestamps:
                            #     future_costmap_ts_val = self.corresponding_timestamps[future_odom_timestamp_str]
                            #     future_costmap_ts_str = str(future_costmap_ts_val)
                                # if future_costmap_ts_str in all_costmap_paths_dict:
                                #     path = all_costmap_paths_dict[future_costmap_ts_str]
                                #     future_costmaps_paths_for_sample.append(path)
                                #     last_valid_path = path
                                    
                                #     # NEW: Get transform matrix for this future timestamp
                                #     if future_costmap_ts_val in self.transform_matrices:
                                #         future_transform = self.transform_matrices[future_costmap_ts_val]
                                #         future_transforms_for_sample.append(future_transform)
                                #         last_valid_transform = future_transform
                                #     else:
                                #         print(f"Warning: Transform matrix not found for future timestamp {future_costmap_ts_val}. Using last valid transform.")
                                #         future_transforms_for_sample.append(last_valid_transform)
                                # else:
                                #     print(f"Warning: Future costmap file not found for timestamp {future_costmap_ts_str}. Using last valid path.")
                                #     future_costmaps_paths_for_sample.append(last_valid_path if last_valid_path else "")
                                #     future_transforms_for_sample.append(last_valid_transform)  # NEW: Append last valid transform
                            # else:
                            #     print(f"Warning: No corresponding costmap found for future timestamp {future_odom_timestamp_str}. Using last valid path.")
                            #     future_costmaps_paths_for_sample.append(last_valid_path if last_valid_path else "")
                            #     future_transforms_for_sample.append(last_valid_transform)  # NEW: Append last valid transform
                        else:
                            # 用最后一个有效的odom和路径补齐
                            if last_valid_odom is not None:
                                future_odom_list.append(last_valid_odom)
                            else:
                                print(f"Error: No valid odom available to pad future_odom_list.")
                                future_odom_list.append(torch.zeros_like(torch.tensor(odom_list[i], dtype=torch.float32)))
                            future_costmaps_paths_for_sample.append(last_valid_path if last_valid_path else "")
                            future_transforms_for_sample.append(last_valid_transform)  # NEW: Append last valid transform

                    # 保证长度为5：如果不足5个odom，再次补齐
                    while len(future_odom_list) < 5:
                        if last_valid_odom is not None:
                            future_odom_list.append(last_valid_odom)
                        else:
                            future_odom_list.append(torch.zeros_like(torch.tensor(odom_list[i], dtype=torch.float32)))
                        future_costmaps_paths_for_sample.append(last_valid_path if last_valid_path else "")
                        future_transforms_for_sample.append(last_valid_transform)  # NEW: Append last valid transform

                    self.future_odom.append(future_odom_list)
                    self.gt_costmap_list.append(future_costmaps_paths_for_sample)
                    self.future_transform_list.append(future_transforms_for_sample)  # NEW: Store future transform matrices

                    # --- End GtCostMap appending ---

                except Exception as e:
                    print(f"Error processing index {i}: {e}")
                    import pdb
                    pdb.set_trace()
                
                self.ground_truth_list.append(torch.tensor(gt, dtype=torch.float32))
                # Trajectory extraction for pedestrians
                peoples = trajics_by_timestamp[time_stamp_list[i]]
                past_trajics = [v["trajic_past"] for v in peoples.values()]
                future_trajics = [v["trajic_future"] for v in peoples.values()]
                people_vx = [v["vx"] for v in peoples.values()]  # x方向的行人速度
                people_vy = [v["vy"] for v in peoples.values()]  # y方向的行人速度

                past_np = np.array(past_trajics)[:, :, [0, 1]]
                future_np = np.array(future_trajics)[:, :, [0, 1]]
                past_np[past_np == -1] = -999
                future_np[future_np == -1] = -999

                num_ped = past_np.shape[0]

                # Truncate or pad
                if num_ped > max_num_ped:
                    past_np = past_np[:max_num_ped][:, 0:5]
                    future_np = future_np[:max_num_ped]
                    # Truncate vx and vy lists as well
                    people_vx = people_vx[:max_num_ped]
                    people_vy = people_vy[:max_num_ped]
                elif num_ped < max_num_ped:
                    pad_shape = (max_num_ped - num_ped, 5, 2)
                    pad_val = -999
                    pad_past = np.full(pad_shape, pad_val, dtype=np.float32)
                    pad_future = np.full(pad_shape, pad_val, dtype=np.float32)
                    # Pad vx and vy lists
                    pad_vx = np.full((max_num_ped - num_ped,), 0, dtype=np.float32)
                    pad_vy = np.full((max_num_ped - num_ped,), 0, dtype=np.float32)
                    try:
                        past_np = np.concatenate([past_np[:,0:5], pad_past], axis=0)
                        future_np = np.concatenate([future_np[:,0:5], pad_future], axis=0)
                        # Concatenate vx and vy pads
                        people_vx = np.concatenate([np.array(people_vx), pad_vx], axis=0)
                        people_vy = np.concatenate([np.array(people_vy), pad_vy], axis=0)
                    except Exception as e:
                        print(e)
                        import pdb
                        pdb.set_trace()

                mask = ((future_np != -999) & (future_np != -1)).all(axis=-1).astype(np.float32)
                self.past_trajic_list.append(past_np[:, 0:5])
                self.future_trajic_list.append(future_np[:, 0:5])
                self.valid_mask_list.append(mask)

                # 添加 vx 和 vy 到新的列表中
                self.people_vx_list.append(people_vx)
                self.people_vy_list.append(people_vy)


        indexfile = os.path.join(img_path, 'split.pt')
        is_generate_split = True
        N = len(self.img_filename)
        if os.path.exists(indexfile):
            train_index, test_index = torch.load(indexfile)
            if len(train_index)+len(test_index) == N:
                is_generate_split = False
            else:
                print("Data changed! Generate a new split file")
        if (is_generate_split):
            indices = range(N)
            train_index = sample(indices, int(ratio*N))
            test_index = np.delete(indices, train_index)
            torch.save((train_index, test_index), indexfile)
        if train == True:
            self.img_filename = itemgetter(*train_index)(self.img_filename)
            self.ground_truth_list    = itemgetter(*train_index)(self.ground_truth_list)
            self.goal_list    = itemgetter(*train_index)(self.goal_list)
            self.past_trajic_list = itemgetter(*train_index)(self.past_trajic_list)
            self.future_trajic_list = itemgetter(*train_index)(self.future_trajic_list)
            self.lidar_list = itemgetter(*train_index)(self.lidar_list) # Keep this for current costmaps
            self.gt_costmap_list = itemgetter(*train_index)(self.gt_costmap_list) # NEW: Split gt_costmap_list
            self.valid_mask_list = itemgetter(*train_index)(self.valid_mask_list)
            self.now_pos = itemgetter(*train_index)(self.odom_list)
            self.scan_list = itemgetter(*train_index)(self.scan_list) # Keep this for raw LiDAR scans
            self.future_odom = itemgetter(*train_index)(self.future_odom) # Keep this for raw LiDAR scans
            
            # NEW: Split transform matrices
            self.current_transform_list = itemgetter(*train_index)(self.current_transform_list)
            self.future_transform_list = itemgetter(*train_index)(self.future_transform_list)
            
            # 添加 vx 和 vy 列表到训练集中
            self.people_vx_list = itemgetter(*train_index)(self.people_vx_list)
            self.people_vy_list = itemgetter(*train_index)(self.people_vy_list)
        else:
            self.img_filename = itemgetter(*test_index)(self.img_filename)
            self.ground_truth_list    = itemgetter(*test_index)(self.ground_truth_list)
            self.goal_list    = itemgetter(*test_index)(self.goal_list)
            self.past_trajic_list = itemgetter(*test_index)(self.past_trajic_list)
            self.future_trajic_list = itemgetter(*test_index)(self.future_trajic_list)
            self.lidar_list = itemgetter(*test_index)(self.lidar_list) # Keep this for current costmaps
            self.gt_costmap_list = itemgetter(*test_index)(self.gt_costmap_list) # NEW: Split gt_costmap_list
            self.valid_mask_list = itemgetter(*test_index)(self.valid_mask_list)
            self.now_pos = itemgetter(*test_index)(self.odom_list)
            self.scan_list = itemgetter(*test_index)(self.scan_list) # Keep this for raw LiDAR scans
            self.future_odom = itemgetter(*test_index)(self.future_odom) # Keep this for raw LiDAR scans
            
            # NEW: Split transform matrices
            self.current_transform_list = itemgetter(*test_index)(self.current_transform_list)
            self.future_transform_list = itemgetter(*test_index)(self.future_transform_list)
            
            # 添加 vx 和 vy 列表到测试集中
            self.people_vx_list = itemgetter(*test_index)(self.people_vx_list)
            self.people_vy_list = itemgetter(*test_index)(self.people_vy_list)
        assert len(self.now_pos) == len(self.img_filename) and len(self.now_pos)==len(self.past_trajic_list), "Odom numbers should match with image and trajectory numbers"

        # Initialize ESDF generator once for efficiency, if it's used with LidarCostMap
        # If LidarCostMap does its own ESDF generation, this might be redundant or needs careful handling.
        self.esdf_generator = LidarCostUtil.ESDFGenerator(
            voxel_size=0.1, device="cpu", robot_height=1.5,
            ground_height=0.0, clear_dist=1.0
        )
    def compute_extrinsic_matrix(self,translation, rotation_quat):
        """
        Computes the extrinsic matrix (camera to world) from camera translation and rotation quaternion.
        """
        rot = R.from_quat(rotation_quat)
        rot_matrix = rot.as_matrix()
        extrinsic = np.eye(4)
        extrinsic[:3, :3] = rot_matrix
        extrinsic[:3, 3] = translation
        return extrinsic

    def transform_world_to_camera(self, world_traj_list, extrinsic_matrix):
        """
        Transforms trajectories from world coordinates to camera coordinates.
        Handles missing data points marked as [-999, -999, -999].
        """
        extrinsic_inv = np.linalg.inv(extrinsic_matrix)
        camera_traj_list = []

        for traj in world_traj_list:
            traj_cam = []
            for point in traj:
                if point[0] == -999 and point[1] == -999 and point[2] == -999:
                    traj_cam.append([-999, -999,-999])
                else:
                    point_h = np.append(point, 1.0)
                    point_cam_h = extrinsic_inv @ point_h
                    traj_cam.append(point_cam_h[:3])
            camera_traj_list.append(traj_cam)

        camera_traj_np = np.array(camera_traj_list, dtype=np.float32)
        return camera_traj_np


    def __len__(self):
        return len(self.img_filename)
    
    def __getitem__(self, idx):
            # --- Load .npy Depth Image ---
            image = np.load(self.img_filename[idx])  # [H, W], float32

            # --- Clear invalid values and out-of-range depths ---
            image = np.where(np.isfinite(image), image, 0.0)
            image[image > self.max_depth] = 0.0

            # --- Convert to tensor and resize to [360, 640] ---
            image_tensor = torch.from_numpy(image).float().unsqueeze(0)  # [1, H, W]
            image_tensor = torch.nn.functional.interpolate(
                image_tensor.unsqueeze(0), size=[360, 640], mode='bilinear', align_corners=False
            ).squeeze(0)  # [1, 360, 640]

            # --- Expand to pseudo-RGB if your PerceptNet expects 3 channels ---
            image_tensor = image_tensor.expand(3, -1, -1)  # [3, 360, 640]

            # --- Load Lidar Costmap ---
            lidar_path_for_costmap = self.lidar_list[idx]
            cost_map = None
            if lidar_path_for_costmap and os.path.exists(lidar_path_for_costmap):
                cost_map = LidarCostMap.from_file(lidar_path_for_costmap)
            else:
                print(f"Warning: Costmap file not found for index {idx} at '{lidar_path_for_costmap}'. Returning None.")
            # --- Remove Pdb debug point ---
            # --- Load and process raw LiDAR scan data ---
            scan_avg = np.zeros((20,80), dtype=np.float32)
            lidar_grid_tensor = torch.zeros((1, 1, 360, 640), dtype=torch.float32)
            for n in range(10):  # 一般 SEQ_LEN = 10
                # Ensure the index is within the valid range
                if idx + n >= len(self.scan_list):
                    print(f"Warning: Scan list index out of bounds at {idx + n}. Skipping remaining scans.")
                    break
                
                raw_scan_data = self.scan_list[idx + n]
                
                # --- FIX: Correctly parse and convert the scan data string ---
                try:
                    # Split the string and skip the first element (timestamp), then convert to float
                    scan_data_floats = [float(val) for val in raw_scan_data.split(',')[1:]]
                    scan_np_array = np.array(scan_data_floats, dtype=np.float32)
                except (ValueError, IndexError) as e:
                    print(f"Warning: Failed to parse LiDAR data at index {idx + n}. Error: {e}")
                    continue # Skip this time step
                
                # Check if the array has enough points
                if scan_np_array.size < 360 + 80 * 9: # Check for minimum expected size
                    continue

                # 使用完整的激光扫描数据
                full_scan = scan_np_array
                total_points = len(full_scan)

                # 计算每段点数（向上取整）
                points_per_segment = (total_points + 79) // 80  # 确保整除

                for i in range(80):
                    start_idx = i * points_per_segment
                    end_idx = min((i+1) * points_per_segment, total_points)
                    
                    segment = full_scan[start_idx:end_idx]
                    
                    if segment.size > 0:
                        scan_avg[2*n, i] = np.min(segment)
                        scan_avg[2*n+1, i] = np.mean(segment)
            
            scan_avg = scan_avg.reshape(1600)
            scan_avg_map = np.matlib.repmat(scan_avg,1,4)
            scan_map = scan_avg_map.reshape(6400)
            scan_avg_tensor = torch.from_numpy(scan_avg).float()
            
            # --- NEW: Create pedestrian velocity map (ped_map) ---
            # Renamed to distinguish between pos and vel
            ped_pos_map = np.zeros((2, 80, 80), dtype=np.float32)
            ped_vel_map = np.zeros((2, 80, 80), dtype=np.float32)
            current_ped_positions = self.past_trajic_list[idx][:, -1]  # Get the last position (current position)
            current_vx = self.people_vx_list[idx]
            current_vy = self.people_vy_list[idx]

            # Iterate over each pedestrian
            for i in range(len(current_ped_positions)):
                x, y = current_ped_positions[i]
                vx = current_vx[i]
                vy = current_vy[i]
                abs_X = x - self.now_pos[idx][0].item()  # 计算相对坐标
                abs_Y = y - self.now_pos[idx][1].item()  # 计算

                # Check for valid data (not the placeholder value -999)
                if (x != -999 and y != -999) or (x != 0 and y != 0):
                    # Assuming relative coordinates and map covers x:[0,20], y:[-10,10]
                    # Convert world coordinates to map coordinates
                    if abs_X >= 0 and abs_X <= 20 and np.abs(abs_Y) <= 10:
                        # Bin size: 0.25m
                        c = int(np.floor(-(abs_Y - 10) / 0.25))
                        r = int(np.floor(abs_X / 0.25))

                        # Boundary check to prevent index out of bounds
                        if r >= 80:
                            r = 79
                        if c >= 80:
                            c = 79
                        
                        # Fill the maps with position and velocity data
                        ped_pos_map[0, r, c] = abs_X
                        ped_pos_map[1, r, c] = abs_Y
                        ped_vel_map[0, r, c] = vx
                        ped_vel_map[1, r, c] = vy
            
            # 将 ped_map 转换为 PyTorch tensor
            ped_pos_map_tensor = torch.from_numpy(ped_pos_map).float()
            ped_vel_map_tensor = torch.from_numpy(ped_vel_map).float()

            # Normalization
            scan_map = (scan_map - self.mu_scan) / self.std_scan
            # Original code normalized ped_map_tensor with mu_ped_pos/std_ped_pos
            # I'll keep this but clarify the name.
            normalized_ped_map = (ped_vel_map_tensor - self.mu_ped_pos) / self.std_ped_pos
            # Removed pdb.set_trace()
            return image_tensor, \
                self.ground_truth_list[idx], \
                self.goal_list[idx], \
                self.past_trajic_list[idx], \
                self.future_trajic_list[idx], \
                cost_map, \
                self.now_pos[idx], \
                lidar_grid_tensor, \
                self.future_odom[idx], \
                scan_map, \
                normalized_ped_map, \
                self.current_transform_list[idx]
