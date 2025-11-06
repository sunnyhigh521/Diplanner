# ======================================================================
# Copyright (c) 2025 SQL
# PCA Lab, NJUST
# All rights reserved.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# ======================================================================

import torch
import torch.nn as nn
import numpy as np
import numpy.matlib
import os
import random
import torch.nn.functional as F
# For reproducibility, we seed the rng
SEED1 = 1337
NEW_LINE = "\n"

# --- Helper functions ---

def set_seed(seed):
    """
    This method seeds all the random number generators and makes
    the results deterministic.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

POINTS = 1080
IMG_SIZE = 80
SEQ_LEN = 10

class NavDataset(torch.utils.data.Dataset):
    def __init__(self, img_path, file_name):
        self.LASER_CLIP = 30
        self.mu_ped_pos = -0.0001
        self.std_ped_pos = 0.0391
        self.mu_scan = 5.3850
        self.std_scan = 4.2161
        # --- NEW: Radar normalization stats placeholder ---
        self.mu_radar = 0.0 # Placeholder: You NEED to calculate this from your radar data
        self.std_radar = 1.0 # Placeholder: You NEED to calculate this from your radar data
        # --- END NEW ---

        self.scan_file_names = []
        self.ped_file_names = []
        self.goal_file_names = []
        self.vel_file_names = []
        # --- NEW: Add radar file names list ---
        self.radar_file_names = []
        # --- END NEW ---

        fp_scan = open(img_path+'/scans_base/'+file_name+'.txt', 'r')
        fp_ped = open(img_path+'/peds_local/'+file_name+'.txt', 'r')
        fp_goal = open(img_path+'/sub_goals_local/'+file_name+'.txt', 'r')
        fp_vel = open(img_path+'/velocities/'+file_name+'.txt', 'r')
        # --- NEW: Open radar data file list ---
        fp_radar = open(img_path+'/radars/'+file_name+'.txt', 'r')
        # --- END NEW ---

        for line in fp_scan.read().split(NEW_LINE):
            if('.npy' in line):
                self.scan_file_names.append(img_path+'/scans_base/'+line)
        for line in fp_ped.read().split(NEW_LINE):
            if('.npy' in line):
                self.ped_file_names.append(img_path+'/peds_local/'+line)
        for line in fp_goal.read().split(NEW_LINE):
            if('.npy' in line):
                self.goal_file_names.append(img_path+'/sub_goals_local/'+line)
        for line in fp_vel.read().split(NEW_LINE):
            if('.npy' in line):
                self.vel_file_names.append(img_path+'/velocities/'+line)
        # --- NEW: Populate radar file names ---
        for line in fp_radar.read().split(NEW_LINE):
            if('.npy' in line):
                self.radar_file_names.append(img_path+'/radars/'+line)
        # --- END NEW ---

        fp_scan.close()
        fp_ped.close()
        fp_goal.close()
        fp_vel.close()
        # --- NEW: Close radar file ---
        fp_radar.close()
        # --- END NEW ---

        self.length = len(self.scan_file_names)
        # --- NEW: Assert all data lists have same length ---
        if not (self.length == len(self.ped_file_names) == \
                len(self.goal_file_names) == len(self.vel_file_names) == \
                len(self.radar_file_names)):
            raise ValueError("All data lists must have the same length!")
        # --- END NEW ---

        print("Dataset length: ", self.length)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        idx_s = idx if (idx + SEQ_LEN) < self.length else idx - SEQ_LEN

        # Create lidar historical map (Depth Map from scan_map):
        scan_avg = np.zeros((20, IMG_SIZE))
        for n in range(SEQ_LEN):
            scan = np.load(self.scan_file_names[idx_s + n])
            scan_tmp = scan[180:-180] # Use 720 points
            for i in range(IMG_SIZE):
                scan_avg[2*n, i] = np.min(scan_tmp[i*9:(i+1)*9])
                scan_avg[2*n+1, i] = np.mean(scan_tmp[i*9:(i+1)*9])
        scan_avg = scan_avg.reshape(1600)
        depth_map = np.matlib.repmat(scan_avg, 1, 4).reshape(IMG_SIZE, IMG_SIZE)

        # Create pedestrian kinematic maps:
        tracked_peds = np.load(self.ped_file_names[idx_s + SEQ_LEN - 1])
        ped_map = np.zeros((2, IMG_SIZE, IMG_SIZE)) # 2 channels for vx, vy
        for tracked_ped in tracked_peds:
            x, y, vx, vy = tracked_ped[0], tracked_ped[1], tracked_ped[2], tracked_ped[3]
            if(x >= 0 and x <= 20 and np.abs(y) <= 10):
                c = int(np.floor(-(y-10)/0.25))
                r = int(np.floor(x/0.25))
                if(r == IMG_SIZE): r = r - 1
                if(c == IMG_SIZE): c = c - 1
                ped_map[0,r,c] = vx
                ped_map[1,r,c] = vy

        # --- NEW: Load and preprocess radar map ---
        radar_map = np.load(self.radar_file_names[idx_s + SEQ_LEN - 1])
        # IMPORTANT: Ensure radar_map is (IMG_SIZE, IMG_SIZE) and single channel here
        if radar_map.shape != (IMG_SIZE, IMG_SIZE):
             print(f"Warning: Radar map shape {radar_map.shape} is not {IMG_SIZE}x{IMG_SIZE}. "
                   f"Please ensure your radar .npy files are preprocessed to this shape.")
             # Add specific reshaping/resizing logic here if needed, e.g.,
             # from skimage.transform import resize
             # radar_map = resize(radar_map, (IMG_SIZE, IMG_SIZE), anti_aliasing=True)
        # --- END NEW ---

        # Get the sub goal data:
        sub_goal = np.load(self.goal_file_names[idx_s + SEQ_LEN - 1])
        # Get the velocity data:
        vel = np.load(self.vel_file_names[idx_s + SEQ_LEN - 1])
        
        # Normalization:
        depth_map = (depth_map - self.mu_scan) / self.std_scan
        ped_map = (ped_map - self.mu_ped_pos) / self.std_ped_pos
        # --- NEW: Normalize radar map ---
        radar_map = (radar_map - self.mu_radar) / self.std_radar
        # --- END NEW ---

        # Transfer to pytorch tensor:
        # Add channel dimension for depth_map and radar_map as they are 2D images
        depth_tensor = torch.FloatTensor(depth_map).unsqueeze(0) # Becomes 1x80x80
        ped_tensor = torch.FloatTensor(ped_map) # Already 2x80x80
        radar_tensor = torch.FloatTensor(radar_map).unsqueeze(0) # Becomes 1x80x80
        sub_goal_tensor = torch.FloatTensor(sub_goal)
        vel_tensor = torch.FloatTensor(vel)

        data = {
            'depth_map': depth_tensor,
            'ped_map': ped_tensor,
            'radar_map': radar_tensor,
            'sub_goal': sub_goal_tensor,
            'velocity': vel_tensor,
        }
        return data

# --- ResNet blocks ---
def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

# --- The main model (PerceptNet now handles multi-modal input) ---
import torch
import torch.nn as nn
import torch.nn.functional as F

class PerceptNet(nn.Module): 

    def __init__(self, block, layers, num_classes=2, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(PerceptNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.initial_inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        
        # --- Depth Map Branch ---
        self.depth_conv1 = nn.Conv2d(3, self.initial_inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.depth_bn1 = norm_layer(self.initial_inplanes)
        self.depth_relu = nn.ReLU(inplace=True)
        self.depth_maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # ResNet layers for Depth Branch
        _depth_inplanes = self.initial_inplanes
        
        self.depth_layer1 = self._make_layer_branch(block, _depth_inplanes, 64, layers[0])
        _depth_inplanes = 64 * block.expansion
        
        self.depth_layer2 = self._make_layer_branch(block, _depth_inplanes, 128, layers[1], stride=2,
                                                    dilate=replace_stride_with_dilation[0])
        _depth_inplanes = 128 * block.expansion

        self.depth_layer3 = self._make_layer_branch(block, _depth_inplanes, 256, layers[2], stride=2,
                                                    dilate=replace_stride_with_dilation[1])
        _depth_inplanes = 256 * block.expansion

        self.depth_layer4 = self._make_layer_branch(block, _depth_inplanes, 512, layers[3], stride=2,
                                                    dilate=replace_stride_with_dilation[2])
        _depth_inplanes = 512 * block.expansion
        
        # 添加深度特征稳定层（不改变通道数）
        self.depth_stabilizer = nn.Sequential(
            nn.Conv2d(_depth_inplanes, _depth_inplanes, kernel_size=1),
            norm_layer(_depth_inplanes),
            nn.ReLU(inplace=True)
        )
        
        # 保存深度分支输出特征数
        self.depth_out_features = _depth_inplanes

        # --- Radar + Pedestrian Kinematics Branch ---
        # 添加雷达特征稳定层（不改变通道数）
        self.radar_stabilizer = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=1),
            norm_layer(16),
            nn.ReLU(inplace=True)
        )
        
        # 添加行人特征稳定层（不改变通道数）
        self.ped_stabilizer = nn.Sequential(
            nn.Conv2d(2, 48, kernel_size=1),
            norm_layer(48),
            nn.ReLU(inplace=True)
        )
        
        self.radar_ped_conv1 = nn.Conv2d(64, self.initial_inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.radar_ped_bn1 = norm_layer(self.initial_inplanes)
        self.radar_ped_relu = nn.ReLU(inplace=True)
        self.radar_ped_maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNet layers for Radar+Ped branch
        _radar_ped_inplanes = self.initial_inplanes
        
        self.radar_ped_layer1 = self._make_layer_branch(block, _radar_ped_inplanes, 64, layers[0])
        _radar_ped_inplanes = 64 * block.expansion

        self.radar_ped_layer2 = self._make_layer_branch(block, _radar_ped_inplanes, 128, layers[1], stride=2,
                                                         dilate=replace_stride_with_dilation[0])
        _radar_ped_inplanes = 128 * block.expansion
        
        self.radar_ped_layer3 = self._make_layer_branch(block, _radar_ped_inplanes, 256, layers[2], stride=2,
                                                         dilate=replace_stride_with_dilation[1])
        _radar_ped_inplanes = 256 * block.expansion
        
        self.radar_ped_layer4 = self._make_layer_branch(block, _radar_ped_inplanes, 512, layers[3], stride=2,
                                                         dilate=replace_stride_with_dilation[2])
        _radar_ped_inplanes = 512 * block.expansion
        
        # 添加雷达+行人特征稳定层（不改变通道数）
        self.radar_ped_stabilizer = nn.Sequential(
            nn.Conv2d(_radar_ped_inplanes, _radar_ped_inplanes*block.expansion, kernel_size=1),
            norm_layer(_radar_ped_inplanes*block.expansion),
            nn.ReLU(inplace=True)
        )
        
        # 保存雷达+行人分支输出特征数
        self.radar_ped_out_features = _radar_ped_inplanes
        
        # 添加融合稳定层（不改变通道数）
        self.fusion_stabilizer = nn.Sequential(
            nn.Conv2d(1024, 1024, kernel_size=1),
            norm_layer(1024),
            nn.ReLU(inplace=True)
        )

        # 其他层保持不变
        self.downsample3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=512, kernel_size=(1, 1), stride=(4,4), padding=(0, 0)),
            nn.BatchNorm2d(512)
        )
        self.downsample2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(1, 1), stride=(2,2), padding=(0, 0)),
            nn.BatchNorm2d(256)
        )
        self.conv2_2 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=(1, 1), stride=(1,1), padding=(0, 0)),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1,1), padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(1, 1), stride=(1,1), padding=(0, 0)),
            nn.BatchNorm2d(256)
        )
        self.conv3_2 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=(1, 1), stride=(1,1), padding=(0, 0)),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1,1), padding=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(1, 1), stride=(1,1), padding=(0, 0)),
            nn.BatchNorm2d(512)
        )
        self.relu2 = nn.ReLU(inplace=True)
        self.relu3 = nn.ReLU(inplace=True)
        
        # --- Fusion Layer ---
        # 计算融合层的输入特征数
        fusion_input_features = self.depth_out_features + self.radar_ped_out_features + 2  # +2 for sub_goal

        self.fusion_fc = nn.Sequential(
            nn.Linear(fusion_input_features, 512),  # First FC layer
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes)  # Final output layer for velocities
        )

        # --- Weight Initialization ---
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)

        # 稳定层初始化
        for stabilizer in [self.depth_stabilizer, self.radar_stabilizer, self.ped_stabilizer,
                           self.radar_ped_stabilizer, self.fusion_stabilizer]:
            for layer in stabilizer:
                if isinstance(layer, nn.Conv2d):
                    nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, 0)

        # Zero-initialize the last BN in each residual branch (optional)
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, block):
                    nn.init.constant_(m.conv2.weight, 0)

    # Modified _make_layer to take `current_inplanes` explicitly
    def _make_layer_branch(self, block, current_inplanes, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or current_inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(current_inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(current_inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        _inplanes_after_first_block = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(_inplanes_after_first_block, planes, groups=self.groups,
                                 base_width=self.base_width, dilation=self.dilation,
                                 norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, depth_map, radar_map, ped_map):
        # --- Depth Map Branch ---
        x_depth = self.depth_conv1(depth_map)
        x_depth = self.depth_bn1(x_depth)
        x_depth = self.depth_relu(x_depth)
        x_depth = self.depth_maxpool(x_depth)

        x_depth = self.depth_layer1(x_depth)
        x_depth = self.depth_layer2(x_depth)
        x_depth = self.depth_layer3(x_depth)
        x_depth = self.depth_layer4(x_depth)
        
        # 深度特征稳定（不改变通道数）
        depth_stable = self.depth_stabilizer(x_depth)

        # --- Radar + Pedestrian Kinematics Branch ---
        ped_map = ped_map.reshape(-1, 2, 80, 80)
        radar_map = radar_map.reshape(-1, 1, 80, 80)
        
        # 雷达特征稳定（不改变通道数）
        radar_stable = self.radar_stabilizer(radar_map)
        
        # 行人特征稳定（不改变通道数）
        ped_stable = self.ped_stabilizer(ped_map)
        # 拼接稳定后的特征
        radar_ped_in = torch.cat((radar_stable, ped_stable), dim=1)  # Shape will be (N, 32, H, W)
        
        x_radar_ped = self.radar_ped_conv1(radar_ped_in)
        x_radar_ped = self.radar_ped_bn1(x_radar_ped)
        x_radar_ped = self.radar_ped_relu(x_radar_ped)
        x_radar_ped = self.radar_ped_layer1(x_radar_ped)
        x_radar_ped = self.radar_ped_layer2(x_radar_ped)
        x_radar_ped = self.radar_ped_layer3(x_radar_ped)
        
        x_radar_ped = self.radar_ped_layer4(x_radar_ped)
        # 雷达+行人特征稳定（不改变通道数）
        radar_ped_stable = self.radar_ped_stabilizer(x_radar_ped)
        
        # 调整雷达+行人特征图尺寸以匹配深度特征图
        radar_ped_resized = F.interpolate(
            radar_ped_stable, 
            size=depth_stable.shape[2:], 
            mode='bilinear',
            align_corners=False
        )
        
        # 拼接特征
        fused_feature = torch.cat([depth_stable, radar_ped_resized], dim=1)
        
        
        return fused_feature