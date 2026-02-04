#!/usr/bin/env python3
"""
CoNIC数据集加载器 - 集成到SAM-Free框架
支持两种数据格式：
1. npy格式：images.npy + labels.npy
2. 混合格式：jpg图像 + txt点标注 + labels.npy实例分割
"""
import os
import numpy as np
import cv2
from PIL import Image
from typing import List, Tuple, Dict, Any, Optional
import random

# 全局缓存，避免反复加载巨大 labels.npy
_CONIC_LABELS = None


class CoNICDatasetLoader:
    """
    CoNIC数据集加载器
    支持npy格式和混合格式两种数据源
    优先使用labels.npy实例通道生成紧致边界框，退化时使用点标注生成固定大小边界框
    """
    
    def __init__(self, data_dir: str, scale_mode: str = "none", box_number: int = 3):
        """
        初始化CoNIC数据集加载器
        
        Args:
            data_dir: CoNIC数据集根目录
            scale_mode: 缩放模式 (暂时未使用)
            box_number: 用于prompt的边界框数量
        """
        self.data_dir = data_dir
        self.scale_mode = scale_mode
        self.box_number = box_number
        
        # 检测数据格式并初始化相应的加载逻辑
        self.data_format = self._detect_data_format()
        
        if self.data_format == "npy":
            self._init_npy_format()
        elif self.data_format == "mixed":
            self._init_mixed_format()
        else:
            raise ValueError(f"无法识别的CoNIC数据格式: {data_dir}")
    
    def _detect_data_format(self) -> str:
        """检测数据格式"""
        # 检查npy格式
        images_npy = os.path.join(self.data_dir, "images.npy")
        labels_npy = os.path.join(self.data_dir, "labels.npy")
        
        # 检查混合格式
        test_images_dir = os.path.join(self.data_dir, "test", "images")
        train_images_dir = os.path.join(self.data_dir, "train", "images")
        val_images_dir = os.path.join(self.data_dir, "val", "images")
        
        print(f"[CONIC DEBUG] 数据格式检测:")
        print(f"[CONIC DEBUG]   images.npy存在: {os.path.exists(images_npy)}")
        print(f"[CONIC DEBUG]   labels.npy存在: {os.path.exists(labels_npy)}")
        print(f"[CONIC DEBUG]   test/images/存在: {os.path.exists(test_images_dir)}")
        print(f"[CONIC DEBUG]   train/images/存在: {os.path.exists(train_images_dir)}")
        print(f"[CONIC DEBUG]   val/images/存在: {os.path.exists(val_images_dir)}")
        
        # 优先使用混合格式，因为它提供更细粒度的点标注（142个点 vs 5个大实例）
        if (os.path.exists(test_images_dir) or os.path.exists(train_images_dir) or os.path.exists(val_images_dir)):
            print("检测到CoNIC混合格式数据（jpg+txt+labels.npy）")
            print("[CONIC DEBUG] 优先使用混合格式以获取细粒度点标注")
            return "mixed"
        elif os.path.exists(images_npy) and os.path.exists(labels_npy):
            print("检测到CoNIC npy格式数据")
            print("[CONIC DEBUG] 使用npy格式（粗粒度实例分割）")
            return "npy"
        else:
            raise FileNotFoundError(f"在 {self.data_dir} 中未找到有效的CoNIC数据格式")
    
    def _init_npy_format(self):
        """初始化npy格式数据加载"""
        self.images_file = os.path.join(self.data_dir, "images.npy")
        self.labels_file = os.path.join(self.data_dir, "labels.npy")
        
        # 检查实际可用的样本数量
        img_size = os.path.getsize(self.images_file)
        label_size = os.path.getsize(self.labels_file)
        
        # 计算实际样本数
        actual_img_samples = (img_size - 126) // (256*256*3)
        actual_label_samples = (label_size - 126) // (256*256*2*2)
        self.total_samples = min(actual_img_samples, actual_label_samples)
        
        # 预定义的数据划分 (可以根据需要调整)
        train_end = int(self.total_samples * 0.7)      # 70% 训练
        val_end = int(self.total_samples * 0.85)       # 15% 验证
        
        self.train_indices = list(range(0, train_end))
        self.val_indices = list(range(train_end, val_end))
        self.test_indices = list(range(val_end, self.total_samples))
        
        print(f"CoNIC数据集(npy格式)加载完成:")
        print(f"  训练集: {len(self.train_indices)} 个样本")
        print(f"  验证集: {len(self.val_indices)} 个样本") 
        print(f"  测试集: {len(self.test_indices)} 个样本")
    
    def _init_mixed_format(self):
        """初始化混合格式数据加载"""
        # 获取各split的可用文件
        self.splits_data = {}
        for split in ['train', 'val', 'test']:
            split_images_dir = os.path.join(self.data_dir, split, "images")
            if os.path.exists(split_images_dir):
                files = [f for f in os.listdir(split_images_dir) 
                        if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                filenames = [os.path.splitext(f)[0] for f in files]
                self.splits_data[split] = filenames
            else:
                self.splits_data[split] = []
        
        # 总样本统计
        total_files = sum(len(files) for files in self.splits_data.values())
        
        print(f"CoNIC数据集(混合格式)加载完成:")
        print(f"  训练集: {len(self.splits_data['train'])} 个样本")
        print(f"  验证集: {len(self.splits_data['val'])} 个样本")
        print(f"  测试集: {len(self.splits_data['test'])} 个样本")
        print(f"  总计: {total_files} 个样本")
    
    def get_all_filenames(self) -> List[str]:
        """获取所有可用的文件名列表"""
        if self.data_format == "npy":
            return [str(i) for i in range(self.total_samples)]
        else:  # mixed format
            all_files = []
            for split_files in self.splits_data.values():
                all_files.extend(split_files)
            return all_files
    
    def get_split_filenames(self, split: str) -> List[str]:
        """获取指定split的文件名列表"""
        if self.data_format == "npy":
            if split == "train":
                indices = self.train_indices
            elif split == "val":
                indices = self.val_indices
            elif split == "test":
                indices = self.test_indices
            else:
                raise ValueError(f"未知的split: {split}")
            return [str(i) for i in indices]
        else:  # mixed format
            return self.splits_data.get(split, [])
    
    def _load_labels_npy(self) -> Optional[np.ndarray]:
        """懒加载并缓存 labels.npy（形状约为 [N, 256, 256, 2]）"""
        global _CONIC_LABELS
        if _CONIC_LABELS is not None:
            return _CONIC_LABELS

        labels_path = os.path.join(self.data_dir, "labels.npy")
        if not os.path.exists(labels_path):
            print(f"  警告: 找不到 labels.npy ({labels_path})，将仅使用点生成固定大小框。")
            _CONIC_LABELS = None
            return None

        print(f"加载 labels.npy: {labels_path}")
        try:
            _CONIC_LABELS = np.load(labels_path)
            print(f"  labels.npy 形状: {_CONIC_LABELS.shape}")
        except Exception as e:
            # 如果 labels.npy 文件损坏或shape不匹配，直接放弃使用它，退化为点框
            print(f"  警告: 加载 labels.npy 失败，将仅使用点生成固定大小框。错误信息: {e}")
            _CONIC_LABELS = None

        return _CONIC_LABELS
    
    def _get_instance_map_from_labels(self, filename: str) -> Optional[np.ndarray]:
        """
        根据文件名（如 '4001'）在 labels.npy 中取对应的实例通道
        默认约定：文件名转为 int 即为 npy 中的索引
        """
        labels = self._load_labels_npy()
        if labels is None:
            return None

        try:
            # 对于mixed格式，filename就是索引；对于npy格式，也是索引
            idx = int(filename)
        except ValueError:
            print(f"  警告: 文件名 {filename} 不能转换为整型索引，将退化为点框。")
            return None

        if idx < 0 or idx >= labels.shape[0]:
            print(f"  警告: 索引 {idx} 超出 labels.npy 范围 [0, {labels.shape[0]-1}]，将退化为点框。")
            return None

        # labels 的第 0 通道为实例分割图（根据 README.txt）
        if len(labels.shape) == 4 and labels.shape[3] >= 1:
            instance_map = labels[idx, :, :, 0]
        elif len(labels.shape) == 3:
            # 如果只有3维，假设就是实例分割图
            instance_map = labels[idx, :, :]
        else:
            print(f"  警告: labels.npy 形状异常 {labels.shape}，将退化为点框。")
            return None
        
        return instance_map
    
    def _load_mixed_sample_data(self, filename: str, split: str) -> Tuple[np.ndarray, Optional[np.ndarray], List[Tuple[int, int]]]:
        """
        加载混合格式CoNIC样本的完整数据
        
        Args:
            filename: 文件名（不含扩展名）
            split: 数据划分 (train/test/val)
        
        Returns:
            image: RGB图像
            binary_mask: 二值分割mask（前景/背景），可能为None
            points: 真实点标注
        """
        # 加载原始图像
        img_path = os.path.join(self.data_dir, split, "images", f"{filename}.jpg")
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"找不到图像文件: {img_path}")
        
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 加载二值分割mask（可选）
        mask_path = os.path.join(self.data_dir, split, "oa_gt", f"{filename}.png")
        binary_mask = None
        if os.path.exists(mask_path):
            binary_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # 加载真实点标注
        gt_path = os.path.join(self.data_dir, split, "gt", f"{filename}.txt")
        points = []
        if os.path.exists(gt_path):
            with open(gt_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            x, y = map(int, line.split())
                            points.append((x, y))
                        except ValueError:
                            continue
        
        return image, binary_mask, points
    
    def _load_sample(self, index: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        从npy文件中加载指定索引的样本
        
        Args:
            index: 样本索引
            
        Returns:
            image: RGB图像 (256, 256, 3)
            instance_map: 实例分割图 (256, 256)
            class_map: 分类图 (256, 256)
        """
        # 计算偏移量
        img_offset = 126 + index * 256 * 256 * 3  # npy头部 + 前面的图像
        label_offset = 126 + index * 256 * 256 * 2 * 2  # npy头部 + 前面的标签
        
        # 读取图像
        with open(self.images_file, 'rb') as f:
            f.seek(img_offset)
            img_data = np.frombuffer(f.read(256*256*3), dtype=np.uint8)
            image = img_data.reshape(256, 256, 3)
        
        # 读取标签
        with open(self.labels_file, 'rb') as f:
            f.seek(label_offset)
            label_data = np.frombuffer(f.read(256*256*2*2), dtype=np.uint16)
            labels = label_data.reshape(256, 256, 2)
        
        instance_map = labels[:, :, 0]  # 实例分割图
        class_map = labels[:, :, 1]     # 分类图
        
        return image, instance_map, class_map
    
    def _extract_bboxes_from_instance_map(self, instance_map: np.ndarray, min_area: int = 10) -> List[Tuple[int, int, int, int, int]]:
        """
        从实例分割图中提取紧致边界框
        与 create_corrected_visualization.py / visualize_and_extract_boxes 中逻辑一致：
        对每个实例取上下左右边界。
        
        Args:
            instance_map: 实例分割图，每个实例有唯一ID
            min_area: 最小面积阈值，过滤小目标
        
        Returns:
            List of (x1, y1, x2, y2, instance_id)
        """
        bboxes = []
        unique_instances = np.unique(instance_map)

        for instance_id in unique_instances:
            if instance_id == 0:
                continue  # 跳过背景

            mask = (instance_map == instance_id)
            area = np.sum(mask)
            if area < min_area:
                continue

            y_coords, x_coords = np.where(mask)
            if len(x_coords) == 0 or len(y_coords) == 0:
                continue

            x1 = int(np.min(x_coords))
            x2 = int(np.max(x_coords))
            y1 = int(np.min(y_coords))
            y2 = int(np.max(y_coords))

            bboxes.append((x1, y1, x2, y2, int(instance_id)))

        return bboxes
    
    def _generate_bboxes_from_points(self, points: List[Tuple[int, int]], box_size: int = 16) -> List[Tuple[int, int, int, int, int]]:
        """
        从点标注生成边界框
        每个点周围生成一个固定大小的边界框
        
        Args:
            points: 点列表 [(x, y), ...]
            box_size: 边界框的大小（边长）
        
        Returns:
            bboxes: 边界框列表 [(x1, y1, x2, y2, id), ...]
        """
        bboxes = []
        half_size = box_size // 2
        
        for i, (x, y) in enumerate(points):
            x1 = max(0, x - half_size)
            y1 = max(0, y - half_size)
            x2 = min(255, x + half_size)  # CoNIC图像是256x256
            y2 = min(255, y + half_size)
            
            bboxes.append((x1, y1, x2, y2, i))  # 添加索引作为ID
        
        return bboxes
    
    def _extract_center_points_from_instance_map(self, instance_map: np.ndarray, min_area: int = 10) -> List[Tuple[int, int, int]]:
        """
        从实例分割图中提取中心点
        
        Args:
            instance_map: 实例分割图
            min_area: 最小面积阈值
        
        Returns:
            List of (x, y, instance_id)
        """
        centers = []
        unique_instances = np.unique(instance_map)
        
        for instance_id in unique_instances:
            if instance_id == 0:  # 跳过背景
                continue
            
            # 创建当前实例的mask
            mask = (instance_map == instance_id).astype(np.uint8)
            
            # 计算面积
            area = np.sum(mask)
            if area < min_area:
                continue
            
            # 计算质心
            M = cv2.moments(mask)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                centers.append((cx, cy, instance_id))
        
        return centers
    
    def get_image_path(self, filename: str) -> str:
        """获取图像路径（虚拟路径，因为数据在npy中）"""
        return f"conic_sample_{filename}"
    
    def get_image_and_boxes(self, filename: str, split: str = "test") -> Tuple[np.ndarray, List[List[int]], int]:
        """
        获取图像和边界框信息
        实现双重策略：优先使用labels.npy实例通道生成紧致边界框，退化时使用点标注
        
        Args:
            filename: 样本索引（字符串形式）
            split: 数据划分（仅对mixed格式有效）
            
        Returns:
            image: numpy array格式的图像
            boxes: 边界框列表，每个框格式为[x1, y1, x2, y2]
            count: 目标数量
        """
        print(f"[CONIC DEBUG] get_image_and_boxes调用: filename={filename}, split={split}, data_format={self.data_format}")
        
        if self.data_format == "npy":
            print(f"[CONIC DEBUG] 使用npy格式加载")
            result = self._get_image_and_boxes_npy_format(filename)
        else:
            print(f"[CONIC DEBUG] 使用混合格式加载")
            result = self._get_image_and_boxes_mixed_format(filename, split)
        
        image, boxes, count = result
        print(f"[CONIC DEBUG] get_image_and_boxes结果: boxes数量={len(boxes)}, count={count}")
        return result
    
    def _get_image_and_boxes_npy_format(self, filename: str) -> Tuple[np.ndarray, List[List[int]], int]:
        """npy格式的图像和边界框加载"""
        index = int(filename)
        
        # 加载样本
        image, instance_map, class_map = self._load_sample(index)
        
        # 提取边界框
        bboxes_with_id = self._extract_bboxes_from_instance_map(instance_map, min_area=10)
        
        # 转换为标准格式 [x1, y1, x2, y2]
        boxes = [[x1, y1, x2, y2] for x1, y1, x2, y2, _ in bboxes_with_id]
        count = len(boxes)
        
        return image, boxes, count
    
    def _get_image_and_boxes_mixed_format(self, filename: str, split: str) -> Tuple[np.ndarray, List[List[int]], int]:
        """
        混合格式的图像和边界框加载
        优先使用 labels.npy 的实例通道生成紧致边界框；
        若当前样本在 labels.npy 中没有对应实例通道，则退化为：
        从点标注生成固定大小的边界框，用于 few-shot 学习。
        """
        # 加载图像和点标注
        image, binary_mask, points = self._load_mixed_sample_data(filename, split)
        
        # 1) 尝试从 labels.npy 的实例通道生成紧致边界框
        instance_map = self._get_instance_map_from_labels(filename)
        
        if instance_map is not None:
            # 使用实例通道生成紧致边界框
            bboxes_with_id = self._extract_bboxes_from_instance_map(instance_map, min_area=10)
            boxes = [[x1, y1, x2, y2] for x1, y1, x2, y2, _ in bboxes_with_id]
        else:
            # 退化策略：从点标注生成固定大小边界框
            bboxes_with_id = self._generate_bboxes_from_points(points, box_size=16)
            boxes = [[x1, y1, x2, y2] for x1, y1, x2, y2, _ in bboxes_with_id]
        
        count = len(boxes)
        return image, boxes, count
    
    def get_image_and_points(self, filename: str, split: str = "test") -> Tuple[np.ndarray, List[List[int]], int]:
        """
        获取图像和中心点信息
        支持两种格式，优先使用实例分割图计算质心，退化时使用点标注
        
        Args:
            filename: 样本索引（字符串形式）
            split: 数据划分（仅对mixed格式有效）
            
        Returns:
            image: numpy array格式的图像
            points: 中心点列表，每个点格式为[x, y]
            count: 目标数量
        """
        print(f"[CONIC DEBUG] get_image_and_points调用: filename={filename}, split={split}, data_format={self.data_format}")
        
        if self.data_format == "npy":
            print(f"[CONIC DEBUG] 使用npy格式加载点")
            result = self._get_image_and_points_npy_format(filename)
        else:
            print(f"[CONIC DEBUG] 使用混合格式加载点")
            result = self._get_image_and_points_mixed_format(filename, split)
        
        image, points, count = result
        print(f"[CONIC DEBUG] get_image_and_points结果: points数量={len(points)}, count={count}")
        return result
    
    def _get_image_and_points_npy_format(self, filename: str) -> Tuple[np.ndarray, List[List[int]], int]:
        """npy格式的图像和中心点加载"""
        index = int(filename)
        
        # 加载样本
        image, instance_map, class_map = self._load_sample(index)
        
        # 提取中心点
        centers_with_id = self._extract_center_points_from_instance_map(instance_map, min_area=10)
        
        # 转换为标准格式 [x, y]
        points = [[x, y] for x, y, _ in centers_with_id]
        count = len(points)
        
        return image, points, count
    
    def _get_image_and_points_mixed_format(self, filename: str, split: str) -> Tuple[np.ndarray, List[List[int]], int]:
        """
        混合格式的图像和中心点加载
        优先从实例分割图计算质心，退化时直接使用点标注
        """
        # 加载图像和点标注
        image, binary_mask, points = self._load_mixed_sample_data(filename, split)
        
        # 1) 尝试从 labels.npy 的实例通道计算质心
        instance_map = self._get_instance_map_from_labels(filename)
        
        if instance_map is not None:
            # 使用实例通道计算质心
            centers_with_id = self._extract_center_points_from_instance_map(instance_map, min_area=10)
            points_list = [[x, y] for x, y, _ in centers_with_id]
        else:
            # 退化策略：直接使用点标注
            points_list = [[x, y] for x, y in points]
        
        count = len(points_list)
        return image, points_list, count
    
    def select_few_shot_boxes(self, bboxes: List[Tuple], strategy: str = "spatial", k: int = 3) -> List[int]:
        """
        选择少量边界框作为few-shot样本
        
        Args:
            bboxes: 边界框列表 [(x1, y1, x2, y2, instance_id), ...]
            strategy: 选择策略 ("spatial", "random", "largest", "center")
            k: 选择数量
        
        Returns:
            选中的边界框索引列表
        """
        if len(bboxes) <= k:
            return list(range(len(bboxes)))
        
        H, W = 256, 256  # CoNIC图像尺寸
        
        if strategy == "random":
            return random.sample(range(len(bboxes)), k)
        
        elif strategy == "largest":
            # 按面积排序，选择最大的k个
            areas = [(x2-x1)*(y2-y1) for x1, y1, x2, y2, _ in bboxes]
            sorted_indices = sorted(range(len(areas)), key=lambda i: areas[i], reverse=True)
            return sorted_indices[:k]
        
        elif strategy == "center":
            # 选择最接近图像中心的k个
            center_x, center_y = W // 2, H // 2
            distances = []
            for x1, y1, x2, y2, _ in bboxes:
                bbox_cx, bbox_cy = (x1 + x2) // 2, (y1 + y2) // 2
                dist = ((bbox_cx - center_x) ** 2 + (bbox_cy - center_y) ** 2) ** 0.5
                distances.append(dist)
            sorted_indices = sorted(range(len(distances)), key=lambda i: distances[i])
            return sorted_indices[:k]
        
        elif strategy == "spatial":
            # 空间分布策略：将图像划分为网格，每个网格选择一个
            grid_h = grid_w = int(np.ceil(np.sqrt(k)))
            selected_indices = []
            
            for i in range(grid_h):
                for j in range(grid_w):
                    if len(selected_indices) >= k:
                        break
                    
                    # 计算网格边界
                    y1_grid = i * H // grid_h
                    y2_grid = (i + 1) * H // grid_h
                    x1_grid = j * W // grid_w
                    x2_grid = (j + 1) * W // grid_w
                    
                    # 找到该网格内的边界框
                    grid_boxes = []
                    for idx, (x1, y1, x2, y2, _) in enumerate(bboxes):
                        bbox_cx, bbox_cy = (x1 + x2) // 2, (y1 + y2) // 2
                        if x1_grid <= bbox_cx < x2_grid and y1_grid <= bbox_cy < y2_grid:
                            grid_boxes.append(idx)
                    
                    # 如果该网格有边界框，随机选择一个
                    if grid_boxes:
                        selected_indices.append(random.choice(grid_boxes))
            
            # 如果选择不够，补充随机选择
            if len(selected_indices) < k:
                remaining = set(range(len(bboxes))) - set(selected_indices)
                additional = random.sample(list(remaining), min(k - len(selected_indices), len(remaining)))
                selected_indices.extend(additional)
            
            return selected_indices[:k]
        
        else:
            raise ValueError(f"Unknown strategy: {strategy}")


# 测试函数
def test_conic_loader():
    """测试CoNIC数据加载器"""
    print("=" * 60)
    print("测试CoNIC数据加载器")
    print("=" * 60)
    
    try:
        # 创建加载器
        loader = CoNICDatasetLoader(data_dir="/ZHANGyong/wjj/dataset/CoNIC")
        
        # 测试获取文件名
        all_files = loader.get_all_filenames()
        test_files = loader.get_split_filenames("test")
        print(f"总文件数: {len(all_files)}")
        print(f"测试集文件数: {len(test_files)}")
        
        # 测试加载样本
        test_filename = test_files[0]
        print(f"\\n测试文件: {test_filename}")
        
        # 测试边界框
        image, boxes, count = loader.get_image_and_boxes(test_filename)
        print(f"图像形状: {image.shape}")
        print(f"边界框数量: {len(boxes)}")
        print(f"目标数量: {count}")
        if boxes:
            print(f"第一个边界框: {boxes[0]}")
        
        # 测试中心点
        image, points, count = loader.get_image_and_points(test_filename)
        print(f"中心点数量: {len(points)}")
        if points:
            print(f"第一个中心点: {points[0]}")
        
        print("\\n✓ CoNIC数据加载器测试通过！")
        
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_conic_loader()