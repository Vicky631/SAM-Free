"""
ShanghaiTech 数据集加载器（适配 SAM-Free 框架的 DatasetManager 接口）

目录结构（与你当前数据一致）：
  {data_dir}/part_A/{train|test}/
    - images/IMG_1.jpg ...
    - ground_truth/GT_IMG_1.mat ...

约定：
- image_id 使用带扩展名的文件名，例如 "IMG_1.jpg"
- 评估与可视化使用原图坐标系：不做强制 resize / rotate
- box prompt：从 GT 点生成自适应大小的方形框（xyxy）
- 定位指标：直接使用 GT 点（xy）
"""

from __future__ import annotations

import os
import glob
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image
from scipy import io


class ShanghaiTechDatasetLoader:
    """
    ShanghaiTech 数据集加载器

    只提供当前框架需要的最小接口：
    - get_split_filenames(split) / get_all_filenames()
    - get_image(image_id) -> np.ndarray RGB
    - get_points(image_id) -> np.ndarray (N,2) xy
    - get_boxes_from_points(points, W, H) -> np.ndarray (N,4) xyxy
    """

    def __init__(
        self,
        data_dir: str,
        part: str = "A",
        scale_mode: str = "none",
        box_number: int = 3,
        box_size: Optional[int] = None,
        adaptive_box_min: int = 16,
        adaptive_box_max: int = 64,
        adaptive_box_div: int = 18,
    ):
        self.data_dir = os.path.abspath(data_dir)
        self.part = str(part).upper()
        self.scale_mode = scale_mode  # 预留，当前不缩放
        self.box_number = int(box_number)
        self.box_size = int(box_size) if box_size is not None else None
        self.adaptive_box_min = int(adaptive_box_min)
        self.adaptive_box_max = int(adaptive_box_max)
        self.adaptive_box_div = int(adaptive_box_div)

        if self.part not in {"A", "B"}:
            raise ValueError(f"ShanghaiTech part 必须是 'A' 或 'B'，当前: {part}")

        part_dir = os.path.join(self.data_dir, f"part_{self.part}")
        if not os.path.isdir(part_dir):
            raise FileNotFoundError(f"未找到 ShanghaiTech part 目录: {part_dir}")

        self._cache_points: dict[str, np.ndarray] = {}

    def _get_split_root(self, split: str) -> str:
        split = str(split).lower()
        if split not in {"train", "test", "val"}:
            # ShanghaiTech 原始仅 train/test；为了与框架兼容，val 可自行准备同结构目录
            raise ValueError(f"ShanghaiTech split 必须是 train/test/val，当前: {split}")

        split_root = os.path.join(self.data_dir, f"part_{self.part}", split)
        if not os.path.isdir(split_root):
            raise FileNotFoundError(f"未找到 ShanghaiTech split 目录: {split_root}")
        return split_root

    def get_split_filenames(self, split: str) -> List[str]:
        split_root = self._get_split_root(split)
        images_dir = os.path.join(split_root, "images")
        if not os.path.isdir(images_dir):
            raise FileNotFoundError(f"未找到 images 目录: {images_dir}")

        files = sorted(glob.glob(os.path.join(images_dir, "*.jpg")))
        # 返回带扩展名的文件名（便于与现有 DatasetManager 的 split list 逻辑一致）
        return [os.path.basename(p) for p in files]

    def get_all_filenames(self) -> List[str]:
        # 合并 train/test（有些场景会用这个接口）
        out: List[str] = []
        for split in ["train", "test"]:
            try:
                out.extend(self.get_split_filenames(split))
            except Exception:
                continue
        # 去重并排序
        return sorted(list(set(out)))

    def _resolve_paths(self, image_id: str, split: str) -> Tuple[str, str]:
        split_root = self._get_split_root(split)
        images_dir = os.path.join(split_root, "images")
        gt_dir = os.path.join(split_root, "ground_truth")

        img_path = os.path.join(images_dir, image_id)
        stem = os.path.splitext(os.path.basename(image_id))[0]
        gt_path = os.path.join(gt_dir, f"GT_{stem}.mat")

        if not os.path.exists(img_path):
            raise FileNotFoundError(f"图像不存在: {img_path}")
        if not os.path.exists(gt_path):
            raise FileNotFoundError(f"标注不存在: {gt_path}")
        return img_path, gt_path

    def get_image(self, image_id: str, split: str = "test") -> np.ndarray:
        img_path, _ = self._resolve_paths(image_id, split)
        img = Image.open(img_path).convert("RGB")
        return np.asarray(img)

    def _parse_mat_points(self, gt_path: str) -> np.ndarray:
        """
        解析 ShanghaiTech GT mat，尽量兼容不同结构。
        常见结构：mat["image_info"][0][0][0][0][0] -> N x 2
        """
        try:
            mat = io.loadmat(gt_path)
        except Exception:
            return np.empty((0, 2), dtype=np.float32)

        pts = None

        # 主路径：image_info
        if "image_info" in mat:
            try:
                data = mat["image_info"][0][0][0][0][0]
                pts = np.asarray(data, dtype=np.float32)
            except Exception:
                pts = None

        # 备选：annPoints（一些实现会这么命名）
        if pts is None and "annPoints" in mat:
            try:
                pts = np.asarray(mat["annPoints"], dtype=np.float32)
            except Exception:
                pts = None

        if pts is None:
            return np.empty((0, 2), dtype=np.float32)

        # 规范到 (N,2)
        if pts.ndim == 1 and pts.size >= 2:
            pts = pts.reshape(1, 2)
        if pts.ndim != 2 or pts.shape[1] < 2:
            return np.empty((0, 2), dtype=np.float32)

        pts = pts[:, :2].astype(np.float32)
        # 过滤非法点
        mask = np.isfinite(pts).all(axis=1)
        pts = pts[mask]
        pts = pts[(pts[:, 0] >= 0) & (pts[:, 1] >= 0)]
        return pts

    def get_points(self, image_id: str, split: str = "test") -> np.ndarray:
        cache_key = f"{split}:{image_id}"
        if cache_key in self._cache_points:
            return self._cache_points[cache_key].copy()

        _, gt_path = self._resolve_paths(image_id, split)
        pts = self._parse_mat_points(gt_path)
        self._cache_points[cache_key] = pts
        return pts.copy()

    def get_boxes_from_points(self, points: np.ndarray, img_w: int, img_h: int) -> np.ndarray:
        if points is None or len(points) == 0:
            return np.empty((0, 4), dtype=np.float32)

        if self.box_size is None:
            # 自适应 box：与 ShanghaiTech_box_generation_plan.md 的推荐策略一致（16~64）
            box_size = max(
                self.adaptive_box_min,
                min(self.adaptive_box_max, min(img_w, img_h) // max(1, self.adaptive_box_div)),
            )
        else:
            box_size = int(self.box_size)

        half = box_size / 2.0
        x = points[:, 0]
        y = points[:, 1]

        x1 = np.clip(x - half, 0, img_w - 1)
        y1 = np.clip(y - half, 0, img_h - 1)
        x2 = np.clip(x + half, 0, img_w - 1)
        y2 = np.clip(y + half, 0, img_h - 1)

        boxes = np.stack([x1, y1, x2, y2], axis=1).astype(np.float32)
        # 去掉退化框
        ok = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
        return boxes[ok]

