from torchvision import transforms
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset
import os
import numpy as np
from typing import List, Tuple, Dict, Any


IM_NORM_MEAN = [0.485, 0.456, 0.406]
IM_NORM_STD = [0.229, 0.224, 0.225]


class CARPK(Dataset):
    def __init__(self, data_dir: str, split: str, subset_scale: float = 1.0, resize_val: bool = True):
        """
        CARPK数据集加载器
        
        Parameters
        ----------
        data_dir : str, path to the CARPK data directory
        split : str, 'train', 'val' or 'test'
        subset_scale : float, scale of the subset of the dataset to use
        resize_val : bool, whether to random crop validation images to 384x384
        """
        assert split in ['train', 'val', 'test']

        # 使用传入的数据目录路径，而不是硬编码
        self.data_dir = data_dir
        self.resize_val = resize_val
        self.im_dir = os.path.join(self.data_dir, 'Images')
        self.anno_path = os.path.join(self.data_dir, "Annotations")
        self.data_split_path = os.path.join(self.data_dir, 'ImageSets')
        self.split = split
        self.split_file = os.path.join(self.data_split_path, split + '.txt')
        
        # 验证路径存在
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"CARPK数据目录不存在: {self.data_dir}")
        if not os.path.exists(self.split_file):
            raise FileNotFoundError(f"Split文件不存在: {self.split_file}")
        
        # 加载split文件
        with open(self.split_file, "r") as s:
            img_names = s.readlines()
        self.idx_running_set = [x.strip() for x in img_names]
        
        # 加载标注信息
        self.gt_cnt = {}
        self.bbox = {}
        self._load_annotations()
        
        # 图像预处理
        self.preprocess = transforms.Compose([
            transforms.Resize(384),
            transforms.ToTensor(),
        ])
    
    def _load_annotations(self):
        """加载所有图像的标注信息"""
        for im_name in self.idx_running_set:
            img_path = os.path.join(self.im_dir, f"{im_name}.png")
            anno_path = os.path.join(self.anno_path, f"{im_name}.txt")
            
            if not os.path.exists(img_path):
                print(f"警告: 图像文件不存在: {img_path}")
                continue
            if not os.path.exists(anno_path):
                print(f"警告: 标注文件不存在: {anno_path}")
                continue
                
            with open(anno_path) as f:
                boxes = f.readlines()
                # 每行格式: x1 y1 x2 y2 count
                boxes = [x.strip().split() for x in boxes if x.strip()]
                boxes = [[int(float(x)) for x in box][:4] for box in boxes if len(box) >= 4]
                
                self.gt_cnt[im_name] = len(boxes)
                self.bbox[im_name] = boxes
            


    def __len__(self):
        return len(self.idx_running_set)

    def __getitem__(self, idx):
        im_name = self.idx_running_set[idx]
        im_path = os.path.join(self.im_dir, f"{im_name}.png")
        img = Image.open(im_path)
        img = self.preprocess(img)
        gt_cnt = self.gt_cnt[im_name]

        return img, gt_cnt
    
    # 添加与框架兼容的接口方法
    def get_all_filenames(self) -> List[str]:
        """获取所有可用的文件名列表"""
        return self.idx_running_set
    
    def get_image_path(self, filename: str) -> str:
        """获取指定文件名的图像路径"""
        return os.path.join(self.im_dir, f"{filename}.png")
    
    def get_image_and_boxes(self, filename: str) -> Tuple[np.ndarray, List[List[int]], int]:
        """
        获取图像和边界框信息
        
        Returns:
            image: numpy array格式的图像
            boxes: 边界框列表，每个框格式为[x1, y1, x2, y2]
            count: 目标数量
        """
        img_path = self.get_image_path(filename)
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"图像文件不存在: {img_path}")
        
        # 加载图像
        image = Image.open(img_path).convert('RGB')
        image_np = np.array(image)
        
        # 获取边界框和数量
        boxes = self.bbox.get(filename, [])
        count = self.gt_cnt.get(filename, 0)
        
        return image_np, boxes, count
    
    def get_count(self, filename: str) -> int:
        """获取指定图像的目标数量"""
        return self.gt_cnt.get(filename, 0)
    
    def get_boxes(self, filename: str) -> List[List[int]]:
        """获取指定图像的边界框列表"""
        return self.bbox.get(filename, [])


class CARPKDatasetLoader:
    """
    CARPK数据集加载器 - 与现有框架兼容的接口
    """
    
    def __init__(self, data_dir: str, scale_mode: str = "none", box_number: int = 3):
        """
        Parameters:
            data_dir: CARPK数据集根目录
            scale_mode: 缩放模式 (暂时未使用)
            box_number: 用于prompt的边界框数量
        """
        self.data_dir = data_dir
        self.scale_mode = scale_mode
        self.box_number = box_number
        
        # 验证数据目录
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"CARPK数据目录不存在: {data_dir}")
        
        self.im_dir = os.path.join(data_dir, 'Images')
        self.anno_dir = os.path.join(data_dir, 'Annotations')
        self.split_dir = os.path.join(data_dir, 'ImageSets')
        
        # 加载所有可用的文件名
        self._load_all_filenames()
    
    def _load_all_filenames(self):
        """加载所有可用的文件名"""
        self.all_filenames = []
        
        # 从训练和测试split文件中加载所有文件名
        for split in ['train', 'test']:
            split_file = os.path.join(self.split_dir, f'{split}.txt')
            if os.path.exists(split_file):
                with open(split_file, 'r') as f:
                    filenames = [line.strip() for line in f.readlines() if line.strip()]
                    self.all_filenames.extend(filenames)
        
        # 去重
        self.all_filenames = list(set(self.all_filenames))
        
        # 验证文件存在性
        valid_filenames = []
        for filename in self.all_filenames:
            img_path = os.path.join(self.im_dir, f"{filename}.png")
            anno_path = os.path.join(self.anno_dir, f"{filename}.txt")
            if os.path.exists(img_path) and os.path.exists(anno_path):
                valid_filenames.append(filename)
        
        self.all_filenames = valid_filenames
        print(f"CARPK数据集加载完成，共 {len(self.all_filenames)} 个有效样本")
    
    def get_all_filenames(self) -> List[str]:
        """获取所有可用的文件名列表"""
        return self.all_filenames
    
    def get_image_path(self, filename: str) -> str:
        """获取指定文件名的图像路径"""
        return os.path.join(self.im_dir, f"{filename}.png")
    
    def get_image_and_boxes(self, filename: str) -> Tuple[np.ndarray, List[List[int]], int]:
        """
        获取图像和边界框信息
        
        Returns:
            image: numpy array格式的图像
            boxes: 边界框列表，每个框格式为[x1, y1, x2, y2]
            count: 目标数量
        """
        img_path = self.get_image_path(filename)
        anno_path = os.path.join(self.anno_dir, f"{filename}.txt")
        
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"图像文件不存在: {img_path}")
        if not os.path.exists(anno_path):
            raise FileNotFoundError(f"标注文件不存在: {anno_path}")
        
        # 加载图像
        image = Image.open(img_path).convert('RGB')
        image_np = np.array(image)
        
        # 加载边界框
        boxes = []
        with open(anno_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split()
                    if len(parts) >= 4:
                        # 格式: x1 y1 x2 y2 count
                        box = [int(float(parts[i])) for i in range(4)]
                        boxes.append(box)
        
        count = len(boxes)
        return image_np, boxes, count
    
