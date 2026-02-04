"""
统一的数据集管理模块
支持多种数据集格式：Sheep OBB, FSC147等
"""
import os
import json
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from PIL import Image
import cv2

# 导入数据集加载器
from myutil.SheepOBB import SheepOBBDatasetLoader
from myutil.FSC_147 import FSC147DatasetLoader
from myutil.CARPK import CARPKDatasetLoader
from myutil.CoNIC import CoNICDatasetLoader
from myutil.ShanghaiTech_loader import ShanghaiTechDatasetLoader


class DatasetManager:
    """统一的数据集管理器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.dataset_config = config['dataset']
        self.loader = None
        # evaluator.run_evaluation(split) 会先调用 get_image_ids(split)，再循环调用 get_sample_data(image_id)
        # 为避免 split 不一致（尤其是 ShanghaiTech 需要 split 来解析路径），这里缓存 active_split。
        self.active_split: str = "test"
        self._init_loader()
    
    def _init_loader(self):
        """根据配置初始化对应的数据集加载器"""
        loader_class = self.dataset_config['loader_class']
        
        if loader_class == "SheepOBBDatasetLoader":
            self.loader = SheepOBBDatasetLoader(
                annotation_dir=self.dataset_config['annotation_dir'],
                image_dir=self.dataset_config['image_dir'],
                scale_mode="none",
                box_number=max(1, self.config['training']['prompt_box_num']),
            )
        elif loader_class == "FSC147DatasetLoader":
            self.loader = FSC147DatasetLoader(
                annotation_file=self.dataset_config['annotation_file'],
                image_root=self.dataset_config['image_root'],
                scale_mode='none'
            )
        elif loader_class == "CARPKDatasetLoader":
            self.loader = CARPKDatasetLoader(
                data_dir=self.dataset_config['data_dir'],
                scale_mode="none",
                box_number=max(1, self.config['training']['prompt_box_num']),
            )
        elif loader_class == "CoNICDatasetLoader":
            self.loader = CoNICDatasetLoader(
                data_dir=self.dataset_config['data_dir'],
                scale_mode="none",
                box_number=max(1, self.config['training']['prompt_box_num']),
            )
        elif loader_class == "ShanghaiTechDatasetLoader":
            self.loader = ShanghaiTechDatasetLoader(
                data_dir=self.dataset_config["data_dir"],
                part=self.dataset_config.get("part", "A"),
                scale_mode="none",
                box_number=max(1, self.config["training"]["prompt_box_num"]),
                box_size=self.dataset_config.get("box_size", None),
                adaptive_box_min=self.dataset_config.get("adaptive_box_min", 16),
                adaptive_box_max=self.dataset_config.get("adaptive_box_max", 64),
                adaptive_box_div=self.dataset_config.get("adaptive_box_div", 18),
            )
        else:
            raise ValueError(f"不支持的数据集加载器: {loader_class}")
    
    def get_image_ids(self, split: str = "test") -> List[str]:
        """获取指定split的图像ID列表"""
        self.active_split = split
        if self.dataset_config['name'] == "Sheep_OBB":
            return self._get_sheep_image_ids()
        elif self.dataset_config['name'] == "FSC147":
            return self._get_fsc147_image_ids(split)
        elif self.dataset_config['name'] == "CARPK":
            return self._get_carpk_image_ids(split)
        elif self.dataset_config['name'] == "CoNIC":
            return self._get_conic_image_ids(split)
        elif self.dataset_config["name"] == "ShanghaiTech":
            return self._get_shanghaitech_image_ids(split)
        else:
            raise ValueError(f"不支持的数据集: {self.dataset_config['name']}")
    
    def _get_sheep_image_ids(self) -> List[str]:
        """获取Sheep数据集的图像ID"""
        all_filenames = self.loader.get_all_filenames()
        
        # 如果指定了split文件，则使用split文件
        split_file = self.dataset_config.get('split_file', '')
        if split_file and os.path.exists(split_file):
            return self._load_split_list(split_file, all_filenames)
        else:
            return all_filenames
    
    def _get_fsc147_image_ids(self, split: str) -> List[str]:
        """获取FSC147数据集的图像ID"""
        with open(self.dataset_config['split_file'], 'r') as f:
            data_split = json.load(f)
        
        im_ids = data_split[split]
        all_filenames = self.loader.get_all_filenames()
        
        # 验证文件存在性
        valid_im_ids = [fname for fname in im_ids if fname in all_filenames]
        if len(valid_im_ids) != len(im_ids):
            print(f"警告：{len(im_ids) - len(valid_im_ids)} 个文件在数据集中未找到")
        
        return valid_im_ids
    
    def _get_carpk_image_ids(self, split: str) -> List[str]:
        """获取CARPK数据集的图像ID"""
        split_file = os.path.join(self.dataset_config['data_dir'], 'ImageSets', f'{split}.txt')
        
        if not os.path.exists(split_file):
            raise FileNotFoundError(f"CARPK split文件不存在: {split_file}")
        
        with open(split_file, 'r') as f:
            im_ids = [line.strip() for line in f.readlines() if line.strip()]
        
        all_filenames = self.loader.get_all_filenames()
        
        # 验证文件存在性
        valid_im_ids = [fname for fname in im_ids if fname in all_filenames]
        if len(valid_im_ids) != len(im_ids):
            print(f"警告：{len(im_ids) - len(valid_im_ids)} 个文件在CARPK数据集中未找到")
        
        return valid_im_ids
    
    def _get_conic_image_ids(self, split: str) -> List[str]:
        """获取CoNIC数据集的图像ID"""
        # CoNIC使用内置的split划分
        filenames = self.loader.get_split_filenames(split)
        return filenames

    def _get_shanghaitech_image_ids(self, split: str) -> List[str]:
        """获取ShanghaiTech数据集的图像ID（文件名，含扩展名）"""
        return self.loader.get_split_filenames(split)
    
    def _load_split_list(self, split_file: str, available_filenames: List[str]) -> List[str]:
        """从split文件加载文件名列表"""
        with open(split_file, "r", encoding="utf-8") as f:
            raw = [x.strip() for x in f.readlines()]
        raw = [x for x in raw if x]

        avail_set = set(available_filenames)
        resolved: List[str] = []
        exts = [".jpg", ".png", ".jpeg"]
        
        for item in raw:
            if item in avail_set:
                resolved.append(item)
                continue
            stem, ext = os.path.splitext(item)
            if ext:  # 有扩展名但没匹配到
                continue
            hit: Optional[str] = None
            for e in exts:
                cand = stem + e
                if cand in avail_set:
                    hit = cand
                    break
            if hit is not None:
                resolved.append(hit)
        return resolved
    
    def get_sample_data(self, image_id: str) -> Dict[str, Any]:
        """获取单个样本的数据"""
        if self.dataset_config['name'] == "Sheep_OBB":
            return self._get_sheep_sample(image_id)
        elif self.dataset_config['name'] == "FSC147":
            return self._get_fsc147_sample(image_id)
        elif self.dataset_config['name'] == "CARPK":
            return self._get_carpk_sample(image_id)
        elif self.dataset_config['name'] == "CoNIC":
            return self._get_conic_sample(image_id)
        elif self.dataset_config["name"] == "ShanghaiTech":
            # 与 get_image_ids(split) 保持一致，避免 train/test 混读
            return self._get_shanghaitech_sample(image_id, self.active_split)
        else:
            raise ValueError(f"不支持的数据集: {self.dataset_config['name']}")

    def _get_shanghaitech_sample(self, image_id: str, split: str) -> Dict[str, Any]:
        """获取ShanghaiTech数据集的单个样本"""
        # 加载图像（RGB numpy）
        image = self.loader.get_image(image_id, split=split)
        h, w = image.shape[0], image.shape[1]

        # 读取 GT 点（xy）
        points = self.loader.get_points(image_id, split=split)
        gt_cnt = int(points.shape[0])

        # box prompt：从点生成自适应大小框（xyxy）
        bboxes = self.loader.get_boxes_from_points(points, img_w=w, img_h=h)

        # centers 就是 points（用于 point prompt / 备用）
        centers = points.astype(np.float32) if gt_cnt > 0 else np.empty((0, 2), dtype=np.float32)

        return {
            "image_id": image_id,
            "image": image,
            "bboxes": bboxes,
            "centers": centers,
            "gt_count": gt_cnt,
            "gt_points": centers,  # 定位评估使用 GT 点
            "split": split,
        }
    
    def _get_sheep_sample(self, image_id: str) -> Dict[str, Any]:
        """获取Sheep数据集的单个样本"""
        # 获取标注信息
        ann = self.loader.get_annotations(image_id, return_scaled=False)
        bboxes = np.asarray(ann["bboxes"], dtype=np.float32)  # (N,4) xyxy
        centers = np.asarray(ann["centers"], dtype=np.float32)  # (N,2) xy
        gt_cnt = int(centers.shape[0])
        
        # 加载图像
        img_path = os.path.join(self.dataset_config['image_dir'], image_id)
        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f"无法加载图像: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        return {
            'image_id': image_id,
            'image': image,
            'bboxes': bboxes,
            'centers': centers,
            'gt_count': gt_cnt,
            'gt_points': centers,  # 用于定位评估
        }
    
    def _get_fsc147_sample(self, image_id: str) -> Dict[str, Any]:
        """获取FSC147数据集的单个样本"""
        # 获取标注信息
        ann = self.loader.get_annotations(image_id, return_scaled=False, return_numpy=True)
        points = ann["points"]  # (N,2) xy - 真实点位置
        boxes = ann["boxes"]    # (M,4) xyxy - 示例框
        gt_cnt = int(points.shape[0])
        
        # 加载图像
        image = self.loader.get_image(image_id)
        image = np.array(image)  # PIL -> numpy
        
        return {
            'image_id': image_id,
            'image': image,
            'bboxes': boxes,  # 示例框
            'centers': points,  # 真实点位置
            'gt_count': gt_cnt,
            'gt_points': points,  # 用于定位评估
        }
    
    def _get_carpk_sample(self, image_id: str) -> Dict[str, Any]:
        """获取CARPK数据集的单个样本"""
        # 获取图像和边界框信息
        image, boxes, gt_cnt = self.loader.get_image_and_boxes(image_id)
        
        # 转换边界框格式为numpy数组
        bboxes = np.asarray(boxes, dtype=np.float32)  # (N,4) xyxy格式
        
        # 计算边界框中心点
        if len(bboxes) > 0:
            centers = np.column_stack([
                (bboxes[:, 0] + bboxes[:, 2]) / 2,  # x中心
                (bboxes[:, 1] + bboxes[:, 3]) / 2   # y中心
            ])
        else:
            centers = np.empty((0, 2), dtype=np.float32)
        
        sample_data = {
            'image': image,
            'bboxes': bboxes,
            'centers': centers,
            'gt_count': gt_cnt,
            'image_id': image_id,
            'gt_points': centers,  # 用于定位评估
        }
        
        return sample_data
    
    def _get_conic_sample(self, image_id: str) -> Dict[str, Any]:
        """获取CoNIC数据集的单个样本"""
        # 调试日志：显示数据格式和加载器信息
        print(f"[CONIC DEBUG] 加载样本 {image_id}")
        print(f"[CONIC DEBUG] 加载器数据格式: {self.loader.data_format}")
        
        # 需要确定当前处理的是哪个split
        # 通过检查image_id是否在各split中来确定
        split = "test"  # 默认值
        for split_name in ['train', 'val', 'test']:
            split_files = self.loader.get_split_filenames(split_name)
            if image_id in split_files:
                split = split_name
                break
        
        print(f"[CONIC DEBUG] 确定的split: {split}")
        
        # 获取图像和边界框信息（传递split参数）
        # 新的CoNIC加载器会自动处理混合格式，优先使用labels.npy生成紧致边界框
        print(f"[CONIC DEBUG] 调用 get_image_and_boxes({image_id}, {split})")
        image, boxes, gt_cnt = self.loader.get_image_and_boxes(image_id, split)
        print(f"[CONIC DEBUG] get_image_and_boxes 返回: boxes数量={len(boxes)}, gt_cnt={gt_cnt}")
        
        # 转换边界框格式为numpy数组
        bboxes = np.asarray(boxes, dtype=np.float32)  # (N,4) xyxy格式
        
        # 获取真实点位置（同样支持混合格式，优先从实例分割图计算质心）
        print(f"[CONIC DEBUG] 调用 get_image_and_points({image_id}, {split})")
        image_points, points_list, _ = self.loader.get_image_and_points(image_id, split)
        print(f"[CONIC DEBUG] get_image_and_points 返回: points数量={len(points_list)}")
        gt_points = np.asarray(points_list, dtype=np.float32) if points_list else np.empty((0, 2), dtype=np.float32)
        
        # 计算边界框中心点作为备用
        if len(bboxes) > 0:
            centers = np.column_stack([
                (bboxes[:, 0] + bboxes[:, 2]) / 2,  # x中心
                (bboxes[:, 1] + bboxes[:, 3]) / 2   # y中心
            ])
        else:
            centers = np.empty((0, 2), dtype=np.float32)
        
        sample_data = {
            'image': image,
            'bboxes': bboxes,
            'centers': centers,  # 边界框中心点，用于prompt生成
            'gt_count': gt_cnt,
            'image_id': image_id,
            'gt_points': gt_points,  # 真实点位置，用于定位评估（优先从实例分割图计算）
            'split': split,  # 添加split信息，用于混合格式数据加载
        }
        
        return sample_data
    
    def generate_prompts(self, sample_data: Dict[str, Any], rng: np.random.Generator) -> Tuple[List, int]:
        """生成prompts"""
        prompt_type = self.config['training']['prompt_type']
        prompt_box_num = self.config['training']['prompt_box_num']
        prompt_select = self.config['training']['prompt_select']
        
        if self.dataset_config['name'] == "Sheep_OBB":
            return self._generate_sheep_prompts(sample_data, prompt_type, prompt_box_num, prompt_select, rng)
        elif self.dataset_config['name'] == "FSC147":
            return self._generate_fsc147_prompts(sample_data, prompt_type, prompt_box_num, prompt_select, rng)
        elif self.dataset_config['name'] == "CARPK":
            return self._generate_carpk_prompts(sample_data, prompt_type, prompt_box_num, prompt_select, rng)
        elif self.dataset_config['name'] == "CoNIC":
            return self._generate_conic_prompts(sample_data, prompt_type, prompt_box_num, prompt_select, rng)
        elif self.dataset_config["name"] == "ShanghaiTech":
            return self._generate_shanghaitech_prompts(sample_data, prompt_type, prompt_box_num, prompt_select, rng)
        else:
            raise ValueError(f"不支持的数据集: {self.dataset_config['name']}")

    def _generate_shanghaitech_prompts(
        self,
        sample_data: Dict[str, Any],
        prompt_type: str,
        prompt_box_num: int,
        prompt_select: str,
        rng: np.random.Generator,
    ) -> Tuple[List, int]:
        """生成ShanghaiTech数据集的prompts（box为主，点为备选）"""
        bboxes = sample_data["bboxes"]
        centers = sample_data["centers"]
        gt_cnt = sample_data["gt_count"]

        # 优先用 bboxes 的数量做选择基准（避免因 points->boxes 过滤导致长度不一致）
        total = len(bboxes) if bboxes is not None else 0
        if total <= 0:
            return [], 0

        prompt_idx = self._select_prompt_indices(total, prompt_box_num, prompt_select, rng)
        prompt_k = len(prompt_idx)

        if prompt_type == "box":
            prompts = bboxes[prompt_idx].tolist() if prompt_k > 0 else []
        elif prompt_type == "point":
            # 点 prompt：使用 centers，与 idx 对齐（如果 centers 少于 boxes，回退截断）
            if centers is None or len(centers) == 0:
                prompts = []
                prompt_k = 0
            else:
                safe_idx = prompt_idx[prompt_idx < len(centers)]
                prompt_k = len(safe_idx)
                prompts = centers[safe_idx].tolist() if prompt_k > 0 else []
        else:
            raise ValueError(f"不支持的prompt类型: {prompt_type}")

        return prompts, prompt_k
    
    def _generate_sheep_prompts(self, sample_data: Dict[str, Any], prompt_type: str, 
                               prompt_box_num: int, prompt_select: str, rng: np.random.Generator) -> Tuple[List, int]:
        """生成Sheep数据集的prompts"""
        bboxes = sample_data['bboxes']
        centers = sample_data['centers']
        gt_cnt = sample_data['gt_count']
        
        # 选择用于prompt的示例索引
        prompt_idx = self._select_prompt_indices(gt_cnt, prompt_box_num, prompt_select, rng)
        prompt_k = len(prompt_idx)
        
        if prompt_type == "box":
            ref_prompt = bboxes[prompt_idx].tolist() if gt_cnt > 0 else []
        else:  # point
            ref_prompt = centers[prompt_idx].tolist() if gt_cnt > 0 else []
        
        return ref_prompt, prompt_k
    
    def _generate_fsc147_prompts(self, sample_data: Dict[str, Any], prompt_type: str,
                                prompt_box_num: int, prompt_select: str, rng: np.random.Generator) -> Tuple[List, int]:
        """生成FSC147数据集的prompts"""
        boxes = sample_data['bboxes']  # 示例框
        
        # FSC147 使用示例框作为 prompt（通常3个框）
        prompt_k = min(len(boxes), prompt_box_num) if len(boxes) > 0 else 0
        if prompt_k > 0:
            prompt_idx = self._select_prompt_indices(len(boxes), prompt_k, prompt_select, rng)
            ref_prompt = boxes[prompt_idx].tolist()
        else:
            ref_prompt = []
        
        return ref_prompt, prompt_k
    
    def _select_prompt_indices(self, total: int, k: int, mode: str, rng: np.random.Generator) -> np.ndarray:
        """选择prompt索引"""
        if total <= 0:
            return np.array([], dtype=np.int64)
        k = max(1, int(k))
        k = min(k, total)
        
        if mode == "head":
            return np.arange(k, dtype=np.int64)
        elif mode == "random":
            return rng.choice(total, size=k, replace=False).astype(np.int64)
        elif mode == "spatial":
            # 对于不支持空间策略的数据集，回退到随机选择
            # CoNIC数据集会在自己的方法中实现空间策略
            return rng.choice(total, size=k, replace=False).astype(np.int64)
        elif mode in ["largest", "center"]:
            # 对于不支持这些策略的数据集，回退到随机选择
            # CoNIC数据集会在自己的方法中实现这些策略
            return rng.choice(total, size=k, replace=False).astype(np.int64)
        else:
            raise ValueError(f"Unknown prompt_select: {mode}")
    
    def get_bmnet_data(self, image_id: str):
        """获取BMNet所需的数据（仅对Sheep数据集）"""
        if self.dataset_config['name'] != "Sheep_OBB":
            return None
        
        all_fnames = self.loader.get_all_filenames()
        idx = all_fnames.index(image_id) if image_id in all_fnames else -1
        if idx >= 0:
            data = self.loader[idx]
            return {
                'patches': data["patches"],
                'scales': data["scales"]
            }
        return None
    
    def _generate_carpk_prompts(self, sample_data: Dict[str, Any], prompt_type: str, 
                               prompt_box_num: int, prompt_select: str, rng: np.random.Generator) -> Tuple[List, int]:
        """生成CARPK数据集的prompts"""
        bboxes = sample_data['bboxes']
        centers = sample_data['centers']
        gt_cnt = sample_data['gt_count']
        
        # 选择用于prompt的示例索引
        prompt_idx = self._select_prompt_indices(gt_cnt, prompt_box_num, prompt_select, rng)
        prompt_k = len(prompt_idx)
        
        if prompt_type == "box":
            # 使用边界框作为prompt
            prompt_boxes = bboxes[prompt_idx] if len(prompt_idx) > 0 else np.empty((0, 4))
            prompts = prompt_boxes.tolist()
        elif prompt_type == "point":
            # 使用中心点作为prompt
            prompt_points = centers[prompt_idx] if len(prompt_idx) > 0 else np.empty((0, 2))
            prompts = prompt_points.tolist()
        else:
            raise ValueError(f"不支持的prompt类型: {prompt_type}")
        
        return prompts, prompt_k
    
    def _generate_conic_prompts(self, sample_data: Dict[str, Any], prompt_type: str, 
                               prompt_box_num: int, prompt_select: str, rng: np.random.Generator) -> Tuple[List, int]:
        """生成CoNIC数据集的prompts - 支持多种小样本选择策略"""
        bboxes = sample_data['bboxes']
        centers = sample_data['centers']
        gt_cnt = sample_data['gt_count']
        
        # CoNIC数据集使用自己的选择策略
        if prompt_select in ["spatial", "largest", "center"] and len(bboxes) > 0:
            # 使用CoNIC加载器的选择策略
            bboxes_with_id = [(bboxes[i][0], bboxes[i][1], bboxes[i][2], bboxes[i][3], i) 
                             for i in range(len(bboxes))]
            prompt_idx = self.loader.select_few_shot_boxes(bboxes_with_id, strategy=prompt_select, k=prompt_box_num)
            prompt_idx = np.array(prompt_idx, dtype=np.int64)
        else:
            # 使用标准选择策略
            prompt_idx = self._select_prompt_indices(gt_cnt, prompt_box_num, prompt_select, rng)
        
        prompt_k = len(prompt_idx)
        
        if prompt_type == "box":
            # 使用边界框作为prompt
            prompt_boxes = bboxes[prompt_idx] if len(prompt_idx) > 0 else np.empty((0, 4))
            prompts = prompt_boxes.tolist()
        elif prompt_type == "point":
            # 使用中心点作为prompt
            prompt_points = centers[prompt_idx] if len(prompt_idx) > 0 else np.empty((0, 2))
            prompts = prompt_points.tolist()
        else:
            raise ValueError(f"不支持的prompt类型: {prompt_type}")
        
        return prompts, prompt_k