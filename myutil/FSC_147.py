import os
import json
from typing import Optional, Union, Tuple

import cv2
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm


class FSC147DatasetLoader:
    """
    FSC147数据集加载类（支持：无缩放 / 等比例缩放 / 拉伸）
    兼容两种点标注格式：
    1. 平铺式：{"points": [[x1,y1], [x2,y2], ...]}
    2. 嵌套式：{"annotations": [{"points": [x1,y1]}, ...]}
    """

    def __init__(self,
                 annotation_file: str,
                 image_root: str,
                 max_size: int = 1024,
                 fixed_size: Optional[Tuple[int, int]] = None,
                 scale_mode: str = 'ratio'):
        self.annotation_file = annotation_file
        self.image_root = image_root
        self.max_size = max_size
        self.fixed_size = fixed_size
        self.scale_mode = scale_mode

        if self.scale_mode == 'fixed_stretch' and self.fixed_size is None:
            raise ValueError("fixed_stretch 必须指定 fixed_size")
        if self.scale_mode not in ['ratio', 'fixed_stretch', 'none']:
            raise ValueError("scale_mode 必须是 ratio/fixed_stretch/none")

        with open(annotation_file, 'r', encoding='utf-8') as f:
            self.annotations = json.load(f)

        self._preprocess_annotations()

    def _calculate_scale(self, orig_w: int, orig_h: int) -> Tuple[float, float]:
        if self.scale_mode == 'ratio':
            scale = self.max_size / max(orig_w, orig_h)
            return scale, scale
        elif self.scale_mode == 'fixed_stretch':
            target_w, target_h = self.fixed_size
            return target_w / orig_w, target_h / orig_h
        else:  # none
            return 1.0, 1.0

    def _parse_points(self, target: dict) -> torch.Tensor:
        """
        兼容两种点标注格式的解析函数
        :param target: 单张图片的标注字典
        :return: 点坐标张量 [N, 2]
        """
        # 格式1：平铺式 points 直接存在
        if 'points' in target and isinstance(target['points'], list):
            points = torch.tensor(target['points'], dtype=torch.float32)
        # 格式2：嵌套式 annotations 下的 points
        elif 'annotations' in target and isinstance(target['annotations'], (dict, list)):
            # 处理 annotations 是字典的情况（如 {0: {...}, 1: {...}}）
            if isinstance(target['annotations'], dict):
                annotations = list(target['annotations'].values())
            else:  # list 情况
                annotations = target['annotations']

            # 提取每个annotation的points并展平
            points_list = []
            for ann in annotations:
                if 'points' in ann and isinstance(ann['points'], list):
                    # 兼容 points 是 [[x,y]] 或 [x,y] 两种子格式
                    if len(ann['points']) > 0 and isinstance(ann['points'][0], list):
                        points_list.extend(ann['points'])
                    else:
                        points_list.append(ann['points'])

            points = torch.tensor(points_list, dtype=torch.float32)
        else:
            raise ValueError(f"不支持的点标注格式！目标字典键：{list(target.keys())}")

        # 确保是二维张量 [N, 2]
        if points.ndim == 1:
            points = points.unsqueeze(0)

        return points

    def _preprocess_annotations(self):
        self.annotation_cache = {}
        for fname in tqdm(self.annotations.keys(), desc="Preprocessing annotations"):
            target = self.annotations[fname]
            orig_w, orig_h = target['width'], target['height']
            scale_w, scale_h = self._calculate_scale(orig_w, orig_h)

            # 兼容两种格式解析点坐标
            orig_points = self._parse_points(target)

            # 解析示例框（两种格式的box字段通常一致）
            if 'box_examples_coordinates' in target:
                orig_boxes = torch.tensor(target['box_examples_coordinates'], dtype=torch.float32)
            else:
                raise KeyError(f"未找到示例框标注 'box_examples_coordinates' - {fname}")

            # 缩放坐标
            scaled_points = orig_points.clone()
            scaled_points[:, 0] *= scale_w
            scaled_points[:, 1] *= scale_h

            scaled_boxes = orig_boxes.clone()
            scaled_boxes[:, [0, 2]] *= scale_w
            scaled_boxes[:, [1, 3]] *= scale_h

            self.annotation_cache[fname] = {
                'orig_size': (orig_w, orig_h),
                'scale_w': scale_w,
                'scale_h': scale_h,
                'orig_points': orig_points,
                'orig_boxes': orig_boxes,
                'scaled_points': scaled_points,
                'scaled_boxes': scaled_boxes
            }

    def get_image(self, fname: str, return_scaled: bool = False) -> Union[Image.Image, Tuple[Image.Image, dict]]:
        img_path = os.path.join(self.image_root, fname)
        img = Image.open(img_path).convert("RGB")

        if not return_scaled:
            return img

        orig_w, orig_h = img.size
        scale_w, scale_h = self._calculate_scale(orig_w, orig_h)

        if self.scale_mode == 'ratio':
            new_w = int(orig_w * scale_w)
            new_h = int(orig_h * scale_h)
            img_scaled = img.resize((new_w, new_h), Image.BILINEAR)
            img_padded = Image.new("RGB", (self.max_size, self.max_size), (255, 255, 255))
            img_padded.paste(img_scaled, (0, 0))
            img_scaled = img_padded
        elif self.scale_mode == 'fixed_stretch':
            img_scaled = img.resize(self.fixed_size, Image.BILINEAR)
        else:
            img_scaled = img.copy()

        scale_info = {
            'scale_w': scale_w,
            'scale_h': scale_h,
            'orig_size': (orig_w, orig_h),
            'scaled_size': img_scaled.size
        }
        return img_scaled, scale_info

    def get_annotations(self, fname: str, return_scaled: bool = False, return_numpy: bool = False):
        if fname not in self.annotation_cache:
            raise KeyError(f"未找到 {fname}")

        cache = self.annotation_cache[fname]
        result = {
            'orig_size': cache['orig_size'],
            'scale_w': cache['scale_w'],
            'scale_h': cache['scale_h']
        }

        if return_scaled:
            points = cache['scaled_points']
            boxes = cache['scaled_boxes']
        else:
            points = cache['orig_points']
            boxes = cache['orig_boxes']

        if return_numpy:
            points = points.numpy()
            boxes = boxes.numpy()

        result['points'] = points
        result['boxes'] = boxes
        return result

    def get_all_filenames(self):
        return list(self.annotations.keys())

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx: int):
        fname = self.get_all_filenames()[idx]
        img = self.get_image(fname)
        anns = self.get_annotations(fname)
        return {
            'filename': fname,
            'image': img,
            'orig_size': anns['orig_size'],
            'scale_w': anns['scale_w'],
            'scale_h': anns['scale_h'],
            'points': anns['points'],
            'boxes': anns['boxes']
        }


# -------------------------- 可视化函数 --------------------------
def visualize_annotations(img, points, boxes, save_path):
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    if isinstance(points, torch.Tensor):
        points = points.numpy()
    if isinstance(boxes, torch.Tensor):
        boxes = boxes.numpy()

    # 兼容可能的维度问题（确保points是[N,2]）
    if points.ndim == 1:
        points = points.reshape(-1, 2)

    # 绘制点标注
    for (x, y) in points:
        cv2.circle(img_cv, (int(x), int(y)), radius=5, color=(0, 0, 255), thickness=-1)

    # 绘制示例框（兼容boxes可能的维度）
    if boxes.ndim == 2 and boxes.shape[1] == 4:
        for (x1, y1, x2, y2) in boxes:
            cv2.rectangle(img_cv, (int(x1), int(y1)), (int(x2), int(y2)), color=(0, 255, 0), thickness=2)
    # 兼容box是嵌套格式 [[x1,y1],[x2,y2]] 的情况
    elif boxes.ndim == 3 and boxes.shape[1:] == (4, 2):
        for box in boxes:
            x1, y1 = box[0][0], box[0][1]
            x2, y2 = box[2][0], box[2][1]
            cv2.rectangle(img_cv, (int(x1), int(y1)), (int(x2), int(y2)), color=(0, 255, 0), thickness=2)

    cv2.imwrite(save_path, img_cv)
    print(f"可视化已保存: {save_path}")


# -------------------------- 完整测试用例 --------------------------
if __name__ == "__main__":
    annotation_file = '/data/wjj/dataset/FSC_147/annotation_FSC147_384_with_gt.json'
    image_root = '/data/wjj/dataset/FSC_147/images_384_VarV2'

    # 测试图片列表（选几张不同长宽比的图）
    test_filenames = [
        '2.jpg'
    ]

    # --------------------------------------------------
    # 测试 1：无缩放模式（原图尺寸）
    # --------------------------------------------------
    print("\n=== 测试 1：无缩放模式 ===")
    loader_none = FSC147DatasetLoader(
        annotation_file=annotation_file,
        image_root=image_root,
        scale_mode='none'
    )

    for fname in test_filenames:
        img, scale_info = loader_none.get_image(fname, return_scaled=True)
        anns = loader_none.get_annotations(fname, return_scaled=True)

        assert scale_info['scale_w'] == 1.0
        assert scale_info['scale_h'] == 1.0
        assert img.size == anns['orig_size']

        print(f"无缩放模式 - {fname}: 尺寸={img.size}, scale={scale_info['scale_w']:.4f}")

        visualize_annotations(
            img=img,
            points=anns['points'],
            boxes=anns['boxes'],
            save_path=f'./test_none_{fname}'
        )

    # --------------------------------------------------
    # 测试 2：等比例缩放模式（带白边）
    # --------------------------------------------------
    print("\n=== 测试 2：等比例缩放模式 ===")
    loader_ratio = FSC147DatasetLoader(
        annotation_file=annotation_file,
        image_root=image_root,
        max_size=512,
        scale_mode='ratio'
    )

    for fname in test_filenames:
        img, scale_info = loader_ratio.get_image(fname, return_scaled=True)
        anns = loader_ratio.get_annotations(fname, return_scaled=True)

        print(f"等比例缩放 - {fname}: 原图尺寸={anns['orig_size']}, 缩放后={img.size}, scale={scale_info['scale_w']:.4f}")

        visualize_annotations(
            img=img,
            points=anns['points'],
            boxes=anns['boxes'],
            save_path=f'./test_ratio_{fname}'
        )

    # --------------------------------------------------
    # 测试 3：拉伸模式（无白边）
    # --------------------------------------------------
    print("\n=== 测试 3：拉伸模式 ===")
    loader_stretch = FSC147DatasetLoader(
        annotation_file=annotation_file,
        image_root=image_root,
        fixed_size=(512, 512),
        scale_mode='fixed_stretch'
    )

    for fname in test_filenames:
        img, scale_info = loader_stretch.get_image(fname, return_scaled=True)
        anns = loader_stretch.get_annotations(fname, return_scaled=True)

        assert img.size == (512, 512)

        print(f"拉伸模式 - {fname}: 目标尺寸=(512,512), 实际尺寸={img.size}")

        visualize_annotations(
            img=img,
            points=anns['points'],
            boxes=anns['boxes'],
            save_path=f'./test_stretch_{fname}'
        )

    # --------------------------------------------------
    # 测试 4：批量随机测试（验证 100 张图不报错）
    # --------------------------------------------------
    print("\n=== 测试 4：批量随机测试 100 张图 ===")
    loader = FSC147DatasetLoader(
        annotation_file=annotation_file,
        image_root=image_root,
        scale_mode='none'
    )

    all_filenames = loader.get_all_filenames()
    import random
    random.shuffle(all_filenames)

    for fname in tqdm(all_filenames[:100], desc="批量测试"):
        img, scale_info = loader.get_image(fname, return_scaled=True)
        anns = loader.get_annotations(fname, return_scaled=True)

        assert img.size == anns['orig_size']
        assert scale_info['scale_w'] == 1.0
        assert scale_info['scale_h'] == 1.0

    print("批量测试通过！")

    # --------------------------------------------------
    # 测试 5：验证坐标是否正确（缩放后坐标应在图像范围内）
    # --------------------------------------------------
    print("\n=== 测试 5：验证坐标有效性 ===")
    loader = FSC147DatasetLoader(
        annotation_file=annotation_file,
        image_root=image_root,
        fixed_size=(600, 400),
        scale_mode='fixed_stretch'
    )

    for fname in test_filenames:
        img, scale_info = loader.get_image(fname, return_scaled=True)
        anns = loader.get_annotations(fname, return_scaled=True)
        points = anns['points'].numpy()

        w, h = img.size
        assert (points[:, 0] >= 0).all()
        assert (points[:, 0] <= w).all()
        assert (points[:, 1] >= 0).all()
        assert (points[:, 1] <= h).all()

        print(f"坐标验证通过 - {fname}")

    print("\n所有测试通过！🎉")