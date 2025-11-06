# ======================================================================
# Copyright (c) 2025 sql
# PCA Lab, NJUST
# All rights reserved.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# ======================================================================

import os
import tqdm
import time
import torch
import json
import wandb
import random
import argparse
import torch.optim as optim
from datetime import datetime
import torch.nn.functional as F
import torchvision.transforms as transforms

from .planner_net import PlannerNet
from .dataloader import PlannerData
from torch.utils.data import DataLoader
from torchutil import EarlyStopScheduler
from traj_cost import TrajCost
from traj_viz import TrajViz
from .lidar_traj_cost import LidarTrajCost
from LidarCostUtil import ESDFGenerator 
from .LidarCostMap import LidarCostMap
from fire_trajic_viz import visualize_batch
torch.set_default_dtype(torch.float32)

class PlannerNetTrainer:
    def __init__(self):
        self.root_folder = os.getenv('EXPERIMENT_DIRECTORY', os.getcwd())
        self.load_config()
        self.parse_args()
        self.prepare_model()
        self.prepare_data()
        if self.args.training == True:
            self.init_wandb()
        else:
            print("Testing Mode")
        
    def init_wandb(self):
        # Convert to string in the format you prefer
        date_time_str = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
        # using Wandb Core
        wandb.require("service")
        # Initialize wandb
        self.wandb_run = wandb.init(
            # set the wandb project where this run will be logged
            project="imperative-path-planning",
            # Set the run name to current date and time
            name=date_time_str + "adamW",
            mode="offline",
            config={
                "learning_rate": self.args.lr,
                "architecture": "PlannerNet",  # Replace with your actual architecture
                "dataset": self.args.data_root,  # Assuming this holds the dataset name
                "epochs": self.args.epochs,
                "goal_step": self.args.goal_step,
                "max_episode": self.args.max_episode,
                "fear_ahead_dist": self.args.fear_ahead_dist,
            }
        )

    def load_config(self):
        with open(os.path.join(os.path.dirname(self.root_folder), 'config', 'training_config.json')) as json_file:
            self.config = json.load(json_file)

    def prepare_model(self):
        # IMPORTANT: If your PlannerNet now takes scan data as an additional channel,
        # you need to adjust 'in_channel' accordingly.
        # For example, if original 'in_channel' was 3 (RGB-like depth), and scan adds 1 channel, it becomes 4.
        self.net = PlannerNet(self.args.in_channel, self.args.knodes) 
        if self.args.resume == True or not self.args.training:
            self.net, self.best_loss = torch.load(self.args.model_save, map_location=torch.device("cpu"))
            print("Resume training from best loss: {}".format(self.best_loss))
        else:
            self.best_loss = float('Inf')

        if torch.cuda.is_available():
            print("Available GPU list: {}".format(list(range(torch.cuda.device_count()))))
            print("Runnin on GPU: {}".format(self.args.gpu_id))
            self.net = self.net.cuda(self.args.gpu_id)

        self.optimizer = optim.AdamW(self.net.parameters(), lr=self.args.lr, weight_decay=self.args.w_decay)
        self.scheduler = EarlyStopScheduler(self.optimizer, factor=self.args.factor, verbose=True, min_lr=self.args.min_lr, patience=self.args.patience)

    def prepare_data(self):
        ids_path = os.path.join(self.args.data_root, self.args.env_id)
        with open(ids_path) as f:
            self.env_list = [line.rstrip() for line in f.readlines()]

        depth_transform = transforms.Compose([
            transforms.Resize((self.args.crop_size)),
            transforms.ToTensor()])
        
        total_img_data = 0
        track_id = 0
        test_env_id = min(self.args.test_env_id, len(self.env_list)-1)
        
        self.train_loader_list = []
        self.val_loader_list   = []
        self.traj_cost_list    = []
        self.traj_viz_list     = []
        
        for env_name in tqdm.tqdm(self.env_list):
            if not self.args.training and track_id != test_env_id:
                track_id += 1
                continue
            is_anymal_frame = False
            sensorOffsetX = 0.0
            camera_tilt = 0.0
            if 'anymal' in env_name:
                is_anymal_frame = True
                sensorOffsetX = self.args.sensor_offsetX_ANYmal
                camera_tilt = self.args.camera_tilt
            elif 'tilt' in env_name:
                camera_tilt = self.args.camera_tilt
            data_path = os.path.join(*[self.args.data_root, self.args.env_type, env_name])
            import time
            time.time()
            
            # === custom_collate_fn update: added 'scan' ===
            def custom_collate_fn(batch):
                # 解构 batch 为多个列表，每个列表对应某一项
                # Order: image, ground_truth, goal, past_trajic, future_trajic, cost_map, lidar_grid_tensor (scan), now_pos, gt_cost_map, future_odom
                images, ground_truths, goals, pasts, futures, cost_maps, scans, now_poses,  future_odom,scan_map,ped_map,transform_matrics = zip(*batch)
                def to_tensor_batch(x):
                    # 尝试统一转换为 Tensor，兼容 numpy 和 list
                    return torch.stack([torch.as_tensor(item) for item in x])

                # future_odom 是一个长度为 BATCH_SIZE 的 list，每个元素是长度 5 的 list，每个 list 是 (3,) Tensor
                # 我们要转成 (B, 5, 3)
                future_odom_tensor = torch.stack([
                    torch.stack([torch.as_tensor(t) for t in one_seq])  # (5, 3)
                    for one_seq in future_odom
                ])  # → (B, 5, 3)

                batch_dict = (
                    torch.stack(images),               # image (depth/RGB-like)
                    to_tensor_batch(ground_truths),    # ground_truth
                    to_tensor_batch(goals),            # goal
                    to_tensor_batch(pasts),            # past_trajic
                    to_tensor_batch(futures),          # future_trajic
                    list(cost_maps),                   # cost_map (list of LidarCostMap objects)
                    to_tensor_batch(scans),            # lidar scan grid tensor
                    to_tensor_batch(now_poses).squeeze(1),  # now_pos
                    future_odom_tensor,                 # future_odom in (B, 5, 3)
                    to_tensor_batch(scan_map),  # scan_map (tensor)
                    to_tensor_batch(ped_map),           # ped_map (tensor)
                    to_tensor_batch(transform_matrics)
                )

                return batch_dict
            train_data = PlannerData(root=data_path,
                                     train=True, 
                                     transform=depth_transform,
                                     sensorOffsetX=sensorOffsetX,
                                     is_robot=is_anymal_frame,
                                     goal_step=self.args.goal_step,
                                     max_episode=self.args.max_episode, 
                                     max_depth=self.args.max_camera_depth)
            
            total_img_data += len(train_data)
            train_loader = DataLoader(train_data, batch_size=self.args.batch_size, shuffle=True, num_workers=0,collate_fn=custom_collate_fn)
            self.train_loader_list.append(train_loader)

            val_data = PlannerData(root=data_path,
                                     train=False,
                                     transform=depth_transform,
                                     sensorOffsetX=sensorOffsetX,
                                     is_robot=is_anymal_frame,
                                     goal_step=self.args.goal_step,
                                     max_episode=self.args.max_episode,
                                     max_depth=self.args.max_camera_depth)

            val_loader = DataLoader(val_data, batch_size=self.args.batch_size, shuffle=True, num_workers=0,collate_fn=custom_collate_fn)
            self.val_loader_list.append(val_loader)
            track_id += 1
            
        print("Data Loading Completed!")
        print("Number of image: %d | Number of goal-image pairs: %d"%(total_img_data, total_img_data * (int)(self.args.max_episode / self.args.goal_step)))
        
        return None

    def MapObsLoss(self, preds, fear, traj_cost, odom, goal, future_trajic, valid_mask, esdf_maps,step=0.1):
        total_loss1 = 0
        fear_labels_list = []

        lidar_trajicCost = LidarTrajCost()
        loss1, fear_labels = lidar_trajicCost.CostofTraj(
            trajic_future=future_trajic,
            valid_mask=valid_mask,
            waypoints=preds,  # 只取前两个坐标
            odom=odom,
            goal=goal,
            ahead_dist=self.args.fear_ahead_dist,
            esdf_map=esdf_maps
        )
        total_loss1 += loss1
        fear_labels = fear_labels.squeeze(1)
        loss2 = F.binary_cross_entropy(fear, fear_labels)

        return total_loss1 + loss2
    
    def MapObsLoss_lobby(self, preds, fear, cost_map, odom, goal, future_trajic, now_pose,future_odom_list,transform_matrics,step=0.1):
        total_loss1 = 0
        fear_labels_list = []

        lidar_trajicCost = LidarTrajCost()
        loss1, fear_labels,smooth_loss = lidar_trajicCost.CostofTraj_lobby(
            trajic_future=future_trajic,
            lidar_map=cost_map,
            waypoints=preds,  # 只取前两个坐标
            odom=odom,
            goal=goal,
            ahead_dist=self.args.fear_ahead_dist,
            now_pose = now_pose,
            transform_matrics = transform_matrics
        )
        total_loss1 += loss1
        # fear_labels = fear_labels.squeeze(1)
        # fear.squeeze(1)
        loss2 = F.binary_cross_entropy(fear, fear_labels)

        return total_loss1 + loss2,smooth_loss

    def train_epoch(self, epoch):
        loss_sum = 0.0
        env_num = len(self.train_loader_list)

        combined = list(self.train_loader_list)
        random.shuffle(combined)

        total_spl = 0.0
        total_spl_samples = 0
        total_success_count = 0
        total_samples = 0
        
        for env_id, loader in enumerate(combined):
            train_loss, batches = 0, len(loader)
            enumerater = tqdm.tqdm(enumerate(loader))

            for batch_idx, inputs in enumerater:
                if torch.cuda.is_available():
                    # === INPUTS ORDER: image, ground_truth, goal, past_trajic, future_trajic, cost_map, scan, now_pose ===
                    image = inputs[0].cuda(self.args.gpu_id)
                    odom = inputs[1]  # ground_truth (odom sequence)
                    goal = inputs[2].cuda(self.args.gpu_id)
                    past_trajic = inputs[3].cuda(self.args.gpu_id)
                    future_trajic = inputs[4].cuda(self.args.gpu_id)
                    lidar_map = inputs[5] # cost_map
                    scan = inputs[7].cuda(self.args.gpu_id) # NEW: scan data (lidar_grid_tensor)
                    now_pose = inputs[6] # now_pos
                    # esdf_maps = inputs[7] # This was an old index, now it's 'now_pose', and scan is inputs[6]
                    future_odom_list = inputs[8] # NEW: future_odom
                    scan_map = inputs[9].cuda(self.args.gpu_id) # scan_map (tensor)
                    ped_map = inputs[10].cuda(self.args.gpu_id) # ped_map (tensor)
                    transform_matrics = inputs[11].cuda(self.args.gpu_id) # transform_matrics (tensor)
                self.optimizer.zero_grad()
                
                # === Concatenate image and scan for PlannerNet input ===
                # image shape: [B, 3, H, W] (pseudo-RGB depth)
                # scan shape: [B, 1, H, W] (occupancy grid)
                # Concatenate along the channel dimension (dim=1)
               # Ensure scan is [B, H, W] before concatenation

                preds, fear = self.net(image, goal,scan_map,ped_map) # Pass combined_input
                
                loss,smooth_loss = self.MapObsLoss_lobby(preds, fear, lidar_map, odom, goal, future_trajic, now_pose,future_odom_list,transform_matrics)

                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()

                # === SPL & Success Rate (Only on success cases) ===
                B, T = preds.size(0), preds.size(1)
                preds_traj = preds.view(B, T, 3)[:, :, :2]  # [B, T, 2]
                gt_traj = torch.tensor(odom, device=preds.device)[..., :2] # Use odom as GT path

                def compute_trajectory_length(traj):  # traj: [B, T, 2]
                    diffs = traj[:, 1:, :] - traj[:, :-1, :]
                    dists = torch.norm(diffs, dim=-1)
                    return dists.sum(dim=1)

                pred_lengths = compute_trajectory_length(preds_traj)  # [B]
                gt_lengths = compute_trajectory_length(gt_traj)        # [B]
                
                # Goal calculation: last point of predicted trajectory vs. target goal
                # Assuming goal is relative to start, so pred_last should match goal
                pred_last = preds[:, -1, :2] # Take x, y from last predicted waypoint
                goal_target = goal[:, :2]    # Take x, y from the goal
                dist = torch.norm(pred_last - goal_target, dim=1) # [B]
                success_mask = dist < 0.1                            # [B], bool (adjust threshold as needed)

                total_success_count += success_mask.sum().item()
                total_samples += B

                # === SPL only for successful predictions ===
                spl_valid_mask = (gt_lengths > 1e-6) & success_mask  # Only success cases
                if spl_valid_mask.any():
                    spl_batch = torch.where(
                        pred_lengths[spl_valid_mask] < gt_lengths[spl_valid_mask],
                        torch.ones_like(pred_lengths[spl_valid_mask]),
                        gt_lengths[spl_valid_mask] / pred_lengths[spl_valid_mask]
                    )
                    total_spl += spl_batch.sum().item()
                    total_spl_samples += spl_valid_mask.sum().item()

                enumerater.set_description(
                    f"Epoch: {epoch} Env: ({env_id+1}/{env_num}) | Loss: {train_loss/(batch_idx+1):.4f} | SPL: {(total_spl / (total_spl_samples + 1e-6)):.4f} | Success Rate: {(total_success_count / total_samples)*100:.2f}%"
                )

            loss_sum += train_loss / (batch_idx + 1)
            wandb.log({"Running Loss": train_loss / (batch_idx + 1)})

        loss_sum /= env_num
        avg_spl = total_spl / (total_spl_samples + 1e-6)
        avg_success_rate = total_success_count / (total_samples + 1e-6)

        print(f"[Epoch {epoch}] Train Loss: {loss_sum:.4f} | Avg SPL: {avg_spl:.4f} | Avg Success Rate: {avg_success_rate*100:.2f}%")

        wandb.log({
            "Avg SPL": avg_spl,
            "Goal Success Rate": avg_success_rate,
            "Avg Training Loss": loss_sum
        })

        return loss_sum
        
    def train(self):
        # Convert to string in the format you prefer
        date_time_str = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
        
        self.args.log_save += (date_time_str + ".txt")
        open(self.args.log_save, 'w').close()

        for epoch in range(self.args.epochs):
            start_time = time.time()
            train_loss = self.train_epoch(epoch)
            val_loss,spl = self.evaluate(is_visualize=False)
            duration = (time.time() - start_time) / 60 # minutes

            self.log_message("Epoch: %d | Training Loss: %f | Val Loss: %f | Duration: %f | SPL: %f" % (epoch, train_loss, val_loss, duration,spl))
            # Log metrics to wandb
            wandb.log({"Avg Training Loss": train_loss, "Validation Loss": val_loss, "Duration (min)": duration})
            self.log_message("Save model of epoch %d" % epoch)
            torch.save((self.net, val_loss), self.args.model_save+f"{epoch}.pt")
            if val_loss < self.best_loss:
                self.log_message("Save model of epoch %d" % epoch)
                torch.save((self.net, val_loss), self.args.model_save+"best.pt")
                self.best_loss = val_loss
                self.log_message("Current val loss: %.4f" % self.best_loss)
                self.log_message("Epoch: %d model saved | Current Min Val Loss: %f" % (epoch, val_loss))

            self.log_message("------------------------------------------------------------------------")
            if self.scheduler.step(val_loss):
                self.log_message('Early Stopping!')
                break
            
        # Close wandb run at the end of training
        self.wandb_run.finish()
    
    def log_message(self, message):
        with open(self.args.log_save, 'a') as f:
            f.writelines(message)
            f.write('\n')
        print(message)

    def evaluate(self, is_visualize=False):
        def compute_trajectory_length(traj):  # traj: [B, T, 2]
            diffs = traj[:, 1:, :] - traj[:, :-1, :]
            dists = torch.norm(diffs, dim=-1)
            return dists.sum(dim=1)  # [B]

        if self.args.test_model:
            model_path = "/opt/data/private/ros/trajPre/FIREDA/THUD_Robot/iPlanner/iplanner/models/lobby_global_Lidar_Dual_esdf_Extend/plannernet还行.pt"
            if os.path.exists(model_path):
                net, val_loss = torch.load(model_path, map_location=f"cuda:{self.args.gpu_id}" if torch.cuda.is_available() else "cpu")
                self.net = net
                print(f"[INFO] Loaded model from {model_path}")
            else:
                print(f"[WARNING] Model path not found: {model_path}")

        self.net.eval()
        def print_model_weights(model, prefix=''):
            for name, module in model.named_children():
                full_name = prefix + '.' + name if prefix else name
                
                # 打印当前模块的权重信息
                if hasattr(module, 'weight') and module.weight is not None:
                    weight = module.weight.data
                    print(f"{full_name} - Weight: min={weight.min().item():.6f}, max={weight.max().item():.6f}, mean={weight.mean().item():.6f}")
                
                # 打印当前模块的偏置信息
                if hasattr(module, 'bias') and module.bias is not None:
                    bias = module.bias.data
                    print(f"{full_name} - Bias: min={bias.min().item():.6f}, max={bias.max().item():.6f}, mean={bias.mean().item():.6f}")
                
                # 递归处理子模块
                if len(list(module.children())) > 0:
                    print_model_weights(module, full_name)

        # 在需要的地方调用
        # print_model_weights(self.net)
        # import pdb
        # pdb.set_trace()
        test_loss = 0
        total_smooth_loss = 0.0  # 新增：用于累积平滑损失
        total_batches = 0
        total_spl = 0.0
        total_spl_samples = 0
        total_batch_nums = 0
        total_success_rate = 0.0

        # 可视化缓存
        preds_viz, odom_viz, goal_viz, now_pose_viz, lidar_map_viz = [], [], [], [], []

        with torch.no_grad():
            for val_loader in self.val_loader_list:
                for batch_idx, inputs in enumerate(val_loader):
                    total_batches += 1
                    if torch.cuda.is_available():
                        # === INPUTS ORDER: image, ground_truth, goal, past_trajic, future_trajic, cost_map, scan, now_pose ===
                        image = inputs[0].cuda(self.args.gpu_id)
                        odom = inputs[1]
                        goal = inputs[2].cuda(self.args.gpu_id)
                        past_trajic = inputs[3].cuda(self.args.gpu_id)
                        future_trajic = inputs[4].cuda(self.args.gpu_id)
                        lidar_map = inputs[5]
                        scan = inputs[7].cuda(self.args.gpu_id) # NEW: scan data (lidar_grid_tensor)
                        now_pose = inputs[6]
                        future_odom_list = inputs[8] # NEW: future_odom
                        scan_map = inputs[9].cuda(self.args.gpu_id) # scan_map (tensor)
                        ped_map = inputs[10].cuda(self.args.gpu_id) # ped_map (tensor)
                        transform_matrics = inputs[11].cuda(self.args.gpu_id) # transform_matrics (tensor)

                    # === Concatenate image and scan for PlannerNet input ===
                    
                    preds, fear = self.net(image, goal,scan_map,ped_map) # Pass combined_input
                    
                    loss, smooth_loss = self.MapObsLoss_lobby(preds, fear, lidar_map, odom, goal, future_trajic, now_pose, future_odom_list, transform_matrics)
                    test_loss += loss
                    total_smooth_loss += smooth_loss  # 新增：累积平滑损失

                    B = preds.size(0)
                    T = preds.size(1)
                    preds_traj = preds.view(B, T, 3)[:, :, :2]
                    gt_traj = odom[:, :, :2]

                    pred_lengths = compute_trajectory_length(preds_traj)
                    gt_lengths = compute_trajectory_length(gt_traj)

                    pred_last = preds[:, -1, :2]
                    goal_target = goal[:, :2]
                    dist = torch.norm(pred_last - goal_target, dim=1)
                    success_mask = dist < 0.05

                    success_count = success_mask.sum().item()
                    success_rate = success_count / B
                    total_success_rate += success_rate
                    total_batch_nums += 1

                    device = torch.device(f'cuda:{self.args.gpu_id}' if torch.cuda.is_available() else 'cpu')
                    gt_lengths = gt_lengths.to(device)
                    success_mask = success_mask.to(device)
                    spl_valid_mask = (gt_lengths > 1e-6) & success_mask
                    if spl_valid_mask.any():
                        spl_batch = torch.where(
                            pred_lengths[spl_valid_mask] < gt_lengths[spl_valid_mask],
                            torch.ones_like(pred_lengths[spl_valid_mask]),
                            gt_lengths[spl_valid_mask] / pred_lengths[spl_valid_mask]
                        )
                        total_spl += spl_batch.sum().item()
                        total_spl_samples += spl_valid_mask.sum().item()

                    print(f"batch SPL: {spl_batch.mean() if spl_valid_mask.any() else 0:.4f} "
                        f"| Goal success rate: {success_rate:.2%} "
                        f"| Smoothed Loss: {smooth_loss:.4f}")

                    # 可视化缓存采样
                    if is_visualize and len(preds_viz) < self.args.visual_number:
                        remain = self.args.visual_number - len(preds_viz)
                        take_n = min(B, remain)
                        preds_viz.extend(preds[:take_n].cpu())
                        odom_viz.extend(odom[:take_n].cpu())
                        goal_viz.extend(goal[:take_n].cpu())
                        now_pose_viz.extend(now_pose[:take_n].cpu())
                        lidar_map_viz.extend(lidar_map[:take_n])

            # 执行可视化
            if is_visualize and len(preds_viz) > 0:
                visualize_batch(
                    torch.stack(preds_viz),
                    torch.stack(odom_viz),
                    torch.stack(goal_viz),
                    torch.stack(now_pose_viz),
                    lidar_map_viz,
                    save_dir='/opt/data/private/ros/trajPre/FIREDA/THUD_Robot/iPlanner/iplanner/traj_viz',
                    prefix='val'
                )

        avg_loss = test_loss / total_batches
        avg_spl = total_spl / max(total_spl_samples, 1)
        avg_success_rate = total_success_rate / total_batch_nums
        avg_smooth_loss = total_smooth_loss / total_batches  # 新增：计算平均平滑损失
        
        print(f"[Evaluation] Avg Loss: {avg_loss:.4f}, "
            f"Avg SPL: {avg_spl:.4f}, "
            f"Total Success rate: {avg_success_rate:.2%}, "
            f"Avg Smoothed Loss: {avg_smooth_loss:.4f}")  # 新增平均平滑损失输出
        
        return avg_loss, avg_spl, avg_smooth_loss  # 返回全局平均平滑损失


    def parse_args(self):
        parser = argparse.ArgumentParser(description='Training script for PlannerNet')

        # dataConfig
        parser.add_argument("--data-root", type=str, default=os.path.join(self.root_folder, self.config['dataConfig'].get('data-root')), help="dataset root folder")
        parser.add_argument('--env-id', type=str, default=self.config['dataConfig'].get('env-id'), help='environment id list')
        parser.add_argument('--env_type', type=str, default=self.config['dataConfig'].get('env_type'), help='the dataset type')
        parser.add_argument('--crop-size', nargs='+', type=int, default=self.config['dataConfig'].get('crop-size'), help='image crop size')
        parser.add_argument('--max-camera-depth', type=float, default=self.config['dataConfig'].get('max-camera-depth'), help='maximum depth detection of camera, unit: meter')

        # modelConfig
        parser.add_argument("--model-save", type=str, default=os.path.join(self.root_folder, self.config['modelConfig'].get('model-save')), help="model save point")
        parser.add_argument('--resume', type=str, default=self.config['modelConfig'].get('resume'))
        # IMPORTANT: 'in-channel' should now be 4 if you're combining 3-channel depth + 1-channel scan
        parser.add_argument('--in-channel', type=int, default=self.config['modelConfig'].get('in-channel'), help='goal input channel numbers') 
        parser.add_argument("--knodes", type=int, default=self.config['modelConfig'].get('knodes'), help="number of max nodes predicted")
        parser.add_argument("--goal-step", type=int, default=self.config['modelConfig'].get('goal-step'), help="number of frames betwen goals") # 目标与目标之间相差5帧
        parser.add_argument("--max-episode", type=int, default=self.config['modelConfig'].get('max-episode-length'), help="maximum episode frame length")
        parser.add_argument("--test-model", type=bool, default=self.config['modelConfig'].get('test-model'), help="is test")

        # trainingConfig
        parser.add_argument('--training', type=str, default=self.config['trainingConfig'].get('training'))
        parser.add_argument("--lr", type=float, default=self.config['trainingConfig'].get('lr'), help="learning rate")
        parser.add_argument("--factor", type=float, default=self.config['trainingConfig'].get('factor'), help="ReduceLROnPlateau factor")
        parser.add_argument("--min-lr", type=float, default=self.config['trainingConfig'].get('min-lr'), help="minimum lr for ReduceLROnPlateau")
        parser.add_argument("--patience", type=int, default=self.config['trainingConfig'].get('patience'), help="patience of epochs for ReduceLROnPlateau")
        parser.add_argument("--epochs", type=int, default=self.config['trainingConfig'].get('epochs'), help="number of training epochs")
        parser.add_argument("--batch-size", type=int, default=self.config['trainingConfig'].get('batch-size'), help="number of minibatch size")
        parser.add_argument("--w-decay", type=float, default=self.config['trainingConfig'].get('w-decay'), help="weight decay of the optimizer")
        parser.add_argument("--num-workers", type=int, default=self.config['trainingConfig'].get('num-workers'), help="number of workers for dataloader")
        parser.add_argument("--gpu-id", type=int, default=self.config['trainingConfig'].get('gpu-id'), help="GPU id")

        # logConfig
        parser.add_argument("--log-save", type=str, default=os.path.join(self.root_folder, self.config['logConfig'].get('log-save')), help="train log file")
        parser.add_argument('--test-env-id', type=int, default=self.config['logConfig'].get('test-env-id'), help='the test env id in the id list')
        parser.add_argument('--visual-number', type=int, default=self.config['logConfig'].get('visual-number'), help='number of visualized trajectories')

        # sensorConfig
        parser.add_argument('--camera-tilt', type=float, default=self.config['sensorConfig'].get('camera-tilt'), help='camera tilt angle for visualization only')
        parser.add_argument('--sensor-offsetX-ANYmal', type=float, default=self.config['sensorConfig'].get('sensor-offsetX-ANYmal'), help='anymal front camera sensor offset in X axis')
        parser.add_argument("--fear-ahead-dist", type=float, default=self.config['sensorConfig'].get('fear-ahead-dist'), help="fear lookahead distance")

        self.args = parser.parse_args()

def main():
    trainer = PlannerNetTrainer()
    # if trainer.args.training == True:
    #     trainer.train()
    trainer.evaluate(is_visualize=True)

if __name__ == "__main__":
    main()