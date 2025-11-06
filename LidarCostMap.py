import torch
import torch.nn.functional as F
import numpy as np
import os

class LidarCostMap:
    def __init__(self, seq, stamp, frame_id, resolution, width, height, origin, data):
        self.header = {
            'seq': seq,
            'stamp': stamp,
            'frame_id': frame_id
        }
        self.info = {
            'resolution': resolution,
            'width': width,
            'height': height,
            'origin': origin
        }
        self.data = data
        
    @classmethod
    def from_file(cls, file_path):
        
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        # 打印文件前20行内容

        
        # 解析元数据
        try:
            seq = int(lines[0].split(': ')[1].strip())
            stamp = int(lines[1].split(': ')[1].strip())
            frame_id = lines[2].split(': ')[1].strip()
            resolution = float(lines[3].split(': ')[1].strip())
            width = int(lines[4].split(': ')[1].strip())
            height = int(lines[5].split(': ')[1].strip())
            origin = {
                'position': {
                    'x': float(lines[6].split(': ')[1].strip()),
                    'y': float(lines[7].split(': ')[1].strip()),
                    'z': float(lines[8].split(': ')[1].strip())
                },
                'orientation': {
                    'x': float(lines[9].split(': ')[1].strip()),
                    'y': float(lines[10].split(': ')[1].strip()),
                    'z': float(lines[11].split(': ')[1].strip()),
                    'w': float(lines[12].split(': ')[1].strip())
                }
            }
        except Exception as e:
            print(f"Error parsing metadata: {e}")
            return None

        # 查找数据起始行
        data_start_index = None
        for i, line in enumerate(lines):
            if "data: [" in line:
                data_start_index = i + 1
                break
        
        if data_start_index is None:
            print("Error: Could not find 'data: [' marker in cost map file")
            return None
        
        # 解析成本图数据
        data_lines = lines[data_start_index:]
        data = []
        
        
        # 处理每行数据
        for line in data_lines:
            # 移除行首尾空格和逗号
            line = line.strip().rstrip(',')
            
            # 检查是否到达数据结束标记
            if line == ']':
                break
                
            # 分割行中的多个数值
            values = line.split(',')
            for value in values:
                value = value.strip()
                if value:  # 确保不是空字符串
                    try:
                        data.append(int(value))
                    except ValueError:
                        # 如果遇到非数字值，跳过或记录警告
                        print(f"Warning: Skipping invalid value '{value}' in cost map data")
        
        # 打印解析出的数据统计
        # 检查数据长度
        expected_size = width * height
        
        if len(data) < expected_size:
            print(f"Warning: Data length {len(data)} < expected {expected_size}, padding with zeros")
            # 用0补齐不足部分
            data.extend([0] * (expected_size - len(data)))
        elif len(data) > expected_size:
            print(f"Warning: Data length {len(data)} > expected {expected_size}, truncating")
            # 截断多余数据
            data = data[:expected_size]
        
        # 将数据转换为numpy数组并reshape为指定的形状
        data_array = np.array(data, dtype=np.uint8).reshape(height, width)

        # 打印数据数组统计
        return cls(seq, stamp, frame_id, resolution, width, height, origin, data_array)

    def __str__(self):
        return (f"header.seq: {self.header['seq']}\n"
                f"header.stamp: {self.header['stamp']}\n"
                f"header.frame_id: {self.header['frame_id']}\n"
                f"info.resolution: {self.info['resolution']}\n"
                f"info.width: {self.info['width']}\n"
                f"info.height: {self.info['height']}\n"
                f"info.origin.position.x: {self.info['origin']['position']['x']}\n"
                f"info.origin.position.y: {self.info['origin']['position']['y']}\n"
                f"info.origin.position.z: {self.info['origin']['position']['z']}\n"
                f"info.origin.orientation.x: {self.info['origin']['orientation']['x']}\n"
                f"info.origin.orientation.y: {self.info['origin']['orientation']['y']}\n"
                f"info.origin.orientation.z: {self.info['origin']['orientation']['z']}\n"
                f"info.origin.orientation.w: {self.info['origin']['orientation']['w']}\n"
                f"data: {self.data.flatten().tolist()}")

    def global_pos_to_ind(self, positions: torch.Tensor):
        """
        将全局坐标 positions 转换为成本图中归一化坐标。
        positions: [T, 2]，全局坐标系下的轨迹点
        
        返回:
            norm_inds: [T, 2] 在 [-1, 1] 之间的 grid_sample 坐标
            valid_mask: [T] 表示哪些点在成本图中合法
        """
        device = positions.device
        resolution = float(self.info['resolution'])
        width = int(self.info['width'])
        height = int(self.info['height'])
        origin_x = float(self.info['origin']['position']['x'])
        origin_y = float(self.info['origin']['position']['y'])
        
        # 全局坐标 -> Pixel
        pixel_xy = (positions[:, :2] - torch.tensor([origin_x, origin_y], device=device)) / resolution
        
        # 创建有效掩码
        valid_mask = (pixel_xy[:, 0] >= 0) & (pixel_xy[:, 0] < width) & \
                     (pixel_xy[:, 1] >= 0) & (pixel_xy[:, 1] < height)
        
        # Pixel -> [-1, 1] 归一化坐标
        norm_xy = pixel_xy.clone()
        norm_xy[:, 0] = (pixel_xy[:, 0] / (width - 1)) * 2 - 1
        norm_xy[:, 1] = (pixel_xy[:, 1] / (height - 1)) * 2 - 1

        return norm_xy, valid_mask
