import os
import numpy as np
from typing import Optional, Union, Tuple
from PIL import Image, ImageFilter  # 修复BILINEAR引用问题
import cv2
from tqdm import tqdm
import torch
from torchvision.transforms import transforms  # 新增：用于patch预处理


class SheepOBBDatasetLoader:
    """
    羊目标OBB（旋转边界框）数据集加载类（支持：无缩放 / 等比例缩放 / 拉伸）
    适配DOTA格式标注：每行 = x1 y1 x2 y2 x3 y3 x4 y4 class difficult
    新增：1.自动提取每个目标的中心点并返回  2.从OBB生成BBox（轴对齐边界框）并返回
    新增patch/scale：3.从BBox裁剪patch样本块 4.计算patch相对图片的尺度scale
    """

    def __init__(self,
                 annotation_dir: str,
                 image_dir: str,
                 max_size: int = 1024,
                 fixed_size: Optional[Tuple[int, int]] = None,
                 scale_mode: str = 'ratio',
                 box_number: int = 3,  # 新增：每个图片取前N个BBox生成patch
                 scale_number: int = 20,  # 新增：scale量化的区间数
                 exemplar_size: Tuple[int, int] = (128, 128)):  # 新增：patch的固定尺寸
        self.annotation_dir = annotation_dir  # 标注文件目录（DOTA格式txt）
        self.image_dir = image_dir  # 图像文件目录
        self.max_size = max_size  # 等比例缩放的最大尺寸
        self.fixed_size = fixed_size  # 固定拉伸尺寸
        self.scale_mode = scale_mode  # 缩放模式

        # 新增：patch/scale相关参数
        self.box_number = box_number  # 每个图取前N个BBox生成patch
        self.scale_number = scale_number  # scale量化区间数
        self.exemplar_size = exemplar_size  # patchresize后的尺寸
        # 新增：patch预处理transform（参考FSC147）
        self.query_transform = transforms.Compose([
            transforms.Resize(self.exemplar_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        # 参数校验
        if self.scale_mode == 'fixed_stretch' and self.fixed_size is None:
            raise ValueError("fixed_stretch模式必须指定fixed_size（宽, 高）")
        if self.scale_mode not in ['ratio', 'fixed_stretch', 'none']:
            raise ValueError("scale_mode仅支持 'ratio'（等比例）/'fixed_stretch'（拉伸）/'none'（无缩放）")

        # 加载所有标注文件（假设图像和标注文件同名，后缀分别为.jpg和.txt）
        self.image_filenames = self._get_valid_image_filenames()
        self.annotation_cache = self._preprocess_annotations()

    def _get_valid_image_filenames(self) -> list:
        """获取所有存在对应标注文件的图像文件名"""
        valid_fnames = []
        for fname in os.listdir(self.image_dir):
            img_ext = os.path.splitext(fname)[1].lower()
            if img_ext not in ['.jpg', '.png', '.jpeg']:
                continue

            # 检查是否存在对应标注文件
            ann_fname = os.path.splitext(fname)[0] + '.txt'
            ann_path = os.path.join(self.annotation_dir, ann_fname)
            if os.path.exists(ann_path):
                valid_fnames.append(fname)
        return valid_fnames

    def _calculate_scale(self, orig_w: int, orig_h: int) -> Tuple[float, float]:
        """计算宽高缩放比例"""
        if self.scale_mode == 'ratio':
            # 等比例缩放，最大边不超过max_size
            scale = self.max_size / max(orig_w, orig_h)
            return scale, scale
        elif self.scale_mode == 'fixed_stretch':
            # 拉伸到固定尺寸
            target_w, target_h = self.fixed_size
            return target_w / orig_w, target_h / orig_h
        else:  # none：不缩放
            return 1.0, 1.0

    def _parse_obb_annotations(self, ann_path: str) -> Tuple[np.ndarray, list]:
        """
        解析DOTA格式OBB标注
        :param ann_path: 标注文件路径
        :return: obbs（N, 4, 2）：N个目标，每个目标4个顶点（x,y）；classes：类别列表
        """
        obbs = []
        classes = []
        with open(ann_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # 解析字段（x1 y1 x2 y2 x3 y3 x4 y4 class difficult）
            fields = line.split()
            if len(fields) != 10:
                print(f"警告：标注行格式错误（跳过）：{line}")
                continue

            # 提取4个顶点坐标（8个数值 -> 4x2矩阵）
            coords = np.array(fields[:8], dtype=np.float32).reshape(4, 2)
            obbs.append(coords)
            classes.append(fields[8])

        return np.array(obbs), classes

    def _obb_to_bbox(self, obbs: np.ndarray) -> np.ndarray:
        """
        核心新增：将OBB（N,4,2）转换为BBox（N,4）
        BBox格式：x1, y1, x2, y2（x1=最小x, y1=最小y, x2=最大x, y2=最大y）
        :param obbs: (N,4,2) OBB坐标
        :return: bboxes (N,4) BBox坐标
        """
        if len(obbs) == 0:
            return np.array([])

        # 提取所有x/y坐标
        x_coords = obbs[:, :, 0]  # (N,4)
        y_coords = obbs[:, :, 1]  # (N,4)

        # 计算每个OBB对应的BBox
        x1 = x_coords.min(axis=1)  # 最小x
        y1 = y_coords.min(axis=1)  # 最小y
        x2 = x_coords.max(axis=1)  # 最大x
        y2 = y_coords.max(axis=1)  # 最大y

        # 拼接为(N,4)的BBox数组
        bboxes = np.stack([x1, y1, x2, y2], axis=1).astype(np.float32)
        return bboxes

    def _calculate_centers(self, obbs: np.ndarray) -> np.ndarray:
        """
        计算OBB框的中心点坐标（核心新增方法）
        :param obbs: (N, 4, 2) 边界框坐标
        :return: centers (N, 2)：每个目标的中心点（x,y），x为4个顶点x均值，y为4个顶点y均值
        """
        centers_x = obbs[:, :, 0].mean(axis=1)  # 每个OBB的x坐标均值
        centers_y = obbs[:, :, 1].mean(axis=1)  # 每个OBB的y坐标均值
        return np.stack([centers_x, centers_y], axis=1).astype(np.float32)

    # 新增：计算patch的scale值（参考FSC147逻辑）
    def _calculate_patch_scale(self, bbox: np.ndarray, img_w: int, img_h: int) -> int:
        """
        计算单个BBox对应的patch尺度值
        :param bbox: (4,) BBox坐标 [x1,y1,x2,y2]
        :param img_w: 图片宽度
        :param img_h: 图片高度
        :return: 量化后的scale值（0 ~ scale_number-1）
        """
        x1, y1, x2, y2 = bbox
        # 计算patch宽高占图片的比例
        w_ratio = (x2 - x1) / img_w
        h_ratio = (y2 - y1) / img_h
        # 综合比例（参考FSC147的0.5系数）
        scale = (w_ratio + h_ratio) * 0.5
        # 量化到指定区间
        scale_step = 0.5 / self.scale_number
        scale_quantized = scale // scale_step
        # 限制最大值，防止越界
        scale_quantized = scale_quantized if scale_quantized < self.scale_number - 1 else self.scale_number - 1
        return int(scale_quantized)

    def _preprocess_annotations(self) -> dict:
        """预处理所有标注，缓存原始/缩放后的OBB、BBox、中心点、patch/scale"""
        cache = {}
        for fname in tqdm(self.image_filenames, desc="预处理标注文件"):
            # 读取图像获取原始尺寸
            img_path = os.path.join(self.image_dir, fname)
            img = Image.open(img_path)
            orig_w, orig_h = img.size

            # 计算缩放比例
            scale_w, scale_h = self._calculate_scale(orig_w, orig_h)

            # 解析标注
            ann_fname = os.path.splitext(fname)[0] + '.txt'
            ann_path = os.path.join(self.annotation_dir, ann_fname)
            orig_obbs, classes = self._parse_obb_annotations(ann_path)

            # 缩放OBB坐标
            scaled_obbs = orig_obbs.copy()
            scaled_obbs[..., 0] *= scale_w  # x坐标缩放
            scaled_obbs[..., 1] *= scale_h  # y坐标缩放

            # 生成原始BBox和缩放后BBox
            orig_bboxes = self._obb_to_bbox(orig_obbs)
            scaled_bboxes = self._obb_to_bbox(scaled_obbs)

            # 计算原始中心点和缩放后中心点
            orig_centers = self._calculate_centers(orig_obbs)
            scaled_centers = self._calculate_centers(scaled_obbs)

            # 新增：预处理patch/scale（只缓存BBox坐标和scale，patch在getitem时裁剪）
            # 取前N个BBox（box_number控制）
            orig_bboxes_selected = orig_bboxes[:self.box_number] if len(orig_bboxes) > 0 else np.array([])
            scaled_bboxes_selected = scaled_bboxes[:self.box_number] if len(scaled_bboxes) > 0 else np.array([])

            # 计算原始/缩放后的scale值
            orig_scales = []
            scaled_scales = []
            if len(orig_bboxes_selected) > 0:
                # 原始scale（基于原图尺寸）
                orig_scales = [self._calculate_patch_scale(bbox, orig_w, orig_h) for bbox in orig_bboxes_selected]
                # 缩放后scale（基于缩放后图片尺寸）
                scaled_w = orig_w * scale_w
                scaled_h = orig_h * scale_h
                scaled_scales = [self._calculate_patch_scale(bbox, scaled_w, scaled_h) for bbox in
                                 scaled_bboxes_selected]

            # 缓存数据（新增patch相关的BBox和scale）
            cache[fname] = {
                'orig_size': (orig_w, orig_h),
                'scale_w': scale_w,
                'scale_h': scale_h,
                'orig_obbs': orig_obbs,
                'scaled_obbs': scaled_obbs,
                'orig_bboxes': orig_bboxes,
                'scaled_bboxes': scaled_bboxes,
                'orig_centers': orig_centers,
                'scaled_centers': scaled_centers,
                'classes': classes,
                'img_path': img_path,
                # 新增：patch相关缓存
                'orig_bboxes_selected': orig_bboxes_selected,  # 用于生成patch的BBox
                'scaled_bboxes_selected': scaled_bboxes_selected,
                'orig_scales': orig_scales,  # patch对应的scale值
                'scaled_scales': scaled_scales
            }
        return cache

    def get_image(self, fname: str, return_scaled: bool = False) -> Union[Image.Image, Tuple[Image.Image, dict]]:
        """
        获取图像（支持缩放）
        :param fname: 图像文件名
        :param return_scaled: 是否返回缩放后的图像和缩放信息
        :return: 原图 或 (缩放图, 缩放信息)
        """
        if fname not in self.annotation_cache:
            raise KeyError(f"未找到图像 {fname} 的标注文件")

        img_path = os.path.join(self.image_dir, fname)
        img = Image.open(img_path).convert("RGB")

        if not return_scaled:
            return img

        # 计算缩放参数
        orig_w, orig_h = img.size
        scale_w, scale_h = self._calculate_scale(orig_w, orig_h)

        # 执行缩放
        if self.scale_mode == 'ratio':
            # 等比例缩放 + 补白到max_size（保持图像比例）
            new_w = int(orig_w * scale_w)
            new_h = int(orig_h * scale_h)
            img_scaled = img.resize((new_w, new_h), Image.Resampling.BILINEAR)
            # 创建白色背景画布
            img_padded = Image.new("RGB", (self.max_size, self.max_size), (255, 255, 255))
            # 居中粘贴（可选：这里左对齐粘贴）
            img_padded.paste(img_scaled, (0, 0))
            img_scaled = img_padded
        elif self.scale_mode == 'fixed_stretch':
            # 直接拉伸到固定尺寸
            img_scaled = img.resize(self.fixed_size, Image.Resampling.BILINEAR)
        else:
            # 无缩放
            img_scaled = img.copy()

        # 缩放信息
        scale_info = {
            'scale_w': scale_w,
            'scale_h': scale_h,
            'orig_size': (orig_w, orig_h),
            'scaled_size': img_scaled.size
        }
        return img_scaled, scale_info

    # 新增：从图片裁剪patch并预处理
    def _extract_patches(self, img: Image.Image, bboxes: np.ndarray) -> Tuple[list, list]:
        """
        从图像中裁剪BBox对应的patch，并返回预处理后的patch和scale
        :param img: PIL图像
        :param bboxes: (N,4) BBox坐标 [x1,y1,x2,y2]
        :return: patches（预处理后的Tensor列表）, scales（scale值列表）
        """
        patches = []
        scales = []
        img_w, img_h = img.size

        if len(bboxes) == 0:
            return patches, scales

        # 遍历选中的BBox，裁剪patch
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox
            # 坐标取整，防止裁剪出错
            x1, y1, x2, y2 = int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))
            # 边界裁剪（防止坐标越界）
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(img_w - 1, x2)
            y2 = min(img_h - 1, y2)
            # 跳过无效框（宽/高为0）
            if x2 - x1 <= 0 or y2 - y1 <= 0:
                continue
            # 裁剪patch
            patch = img.crop((x1, y1, x2, y2))
            # 预处理patch（resize+归一化）
            patch_tensor = self.query_transform(patch)
            patches.append(patch_tensor)
            # 计算scale值
            scale = self._calculate_patch_scale(bbox, img_w, img_h)
            scales.append(scale)

        return patches, scales

    def get_annotations(self, fname: str, return_scaled: bool = False) -> dict:
        """
        获取标注信息（新增patch/scale相关字段）
        :param fname: 图像文件名
        :param return_scaled: 是否返回缩放后的坐标/scale
        :return: 包含patch/scale的标注字典
        """
        if fname not in self.annotation_cache:
            raise KeyError(f"未找到图像 {fname} 的标注文件")

        cache = self.annotation_cache[fname]
        return {
            'orig_size': cache['orig_size'],
            'scale_w': cache['scale_w'],
            'scale_h': cache['scale_h'],
            'obbs': cache['scaled_obbs'] if return_scaled else cache['orig_obbs'],
            'bboxes': cache['scaled_bboxes'] if return_scaled else cache['orig_bboxes'],
            'centers': cache['scaled_centers'] if return_scaled else cache['orig_centers'],
            'classes': cache['classes'],
            'image_path': cache['img_path'],
            # 新增：patch相关字段
            'bboxes_selected': cache['scaled_bboxes_selected'] if return_scaled else cache['orig_bboxes_selected'],
            'scales': cache['scaled_scales'] if return_scaled else cache['orig_scales']
        }

    def get_all_filenames(self) -> list:
        """获取所有有效图像文件名"""
        return self.image_filenames

    def __len__(self) -> int:
        """数据集大小（有效图像数量）"""
        return len(self.image_filenames)

    def __getitem__(self, idx: int) -> dict:
        """按索引获取数据（新增patches和scales字段）"""
        fname = self.image_filenames[idx]
        img = self.get_image(fname)
        anns = self.get_annotations(fname)

        # 提取patch和scale
        patches, scales = self._extract_patches(img, anns['bboxes_selected'])
        # 将patches堆叠为Tensor（空则返回空Tensor）
        patches_tensor = torch.stack(patches) if len(patches) > 0 else torch.tensor([])
        # scales转为Tensor
        scales_tensor = torch.tensor(scales, dtype=torch.long) if len(scales) > 0 else torch.tensor([],
                                                                                                    dtype=torch.long)

        return {
            'filename': fname,
            'image': img,
            'orig_size': anns['orig_size'],
            'obbs': anns['obbs'],
            'bboxes': anns['bboxes'],
            'centers': anns['centers'],
            'classes': anns['classes'],
            # 新增：patch和scale
            'patches': patches_tensor,  # 预处理后的patch Tensor (N,3,H,W)
            'scales': scales_tensor  # patch对应的scale值 Tensor (N,)
        }


# -------------------------- 可视化函数（修改后：同时显示OBB/BBox/中心点） --------------------------
def visualize_obb_bbox_annotations(img: Image.Image, obbs: np.ndarray, bboxes: np.ndarray, centers: np.ndarray,
                                   classes: list, save_path: str):
    """
    可视化标注：OBB（绿色旋转框）+ BBox（蓝色轴对齐框）+ 中心点（红色）
    :param img: PIL图像
    :param obbs: (N, 4, 2) OBB坐标
    :param bboxes: (N, 4) BBox坐标（x1,y1,x2,y2）
    :param centers: (N, 2) 中心点坐标
    :param classes: 类别列表
    :param save_path: 保存路径
    """
    # 转换为OpenCV格式（BGR通道）
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    h, w = img_cv.shape[:2]

    # 定义颜色（BGR格式）
    color_map = {'sheep': (0, 255, 0)}  # OBB绿色
    bbox_color = (255, 0, 0)  # BBox蓝色
    center_color = (0, 0, 255)  # 中心点红色

    # 1. 绘制OBB旋转矩形框
    for i, (obb, cls) in enumerate(zip(obbs, classes)):
        pts = obb.astype(np.int32)  # 坐标转为整数
        pts = pts.reshape((-1, 1, 2))  # OpenCV需要的形状
        cv2.polylines(img_cv, [pts], isClosed=True, color=color_map.get(cls, (0, 0, 255)), thickness=2)

    # 2. 绘制BBox轴对齐框（新增）
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox.astype(np.int32)
        cv2.rectangle(img_cv, (x1, y1), (x2, y2), bbox_color, thickness=2)  # 蓝色矩形框

    # 3. 绘制中心点
    for (cx, cy) in centers:
        cv2.circle(img_cv, (int(round(cx)), int(round(cy))), radius=4, color=center_color, thickness=-1)

    # 4. 添加类别标签（在BBox左上角）
    for i, (bbox, cls) in enumerate(zip(bboxes, classes)):
        x1, y1 = bbox[:2].astype(np.int32)
        cv2.putText(
            img_cv, cls, (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_map.get(cls, (0, 0, 255)),
            thickness=1
        )

    # 保存可视化结果,这里不改了但是cv的保存路径不可以有中文
    cv2.imwrite(save_path, img_cv)
    print(f"可视化结果已保存：{save_path}")


import numpy as np


def process_boxes(original_boxes, order=[1, 0, 3, 2], return_num=None):
    """
    处理原始boxes：先将浮点坐标取整，再调整坐标顺序，最后返回指定数量的box
    :param original_boxes: 原始bbox数组/列表 (形状：(4,) 或 (N,4))
    :param order: 坐标顺序调整规则，默认 [1,0,3,2] 表示：
                  原始顺序[xmin, ymin, xmax, ymax] → 新顺序[ymin, xmin, ymax, xmax]
    :param return_num: 要返回的box数量，None表示返回全部；
                       若为整数：正数取前N个，负数取后N个（仅批量box有效）
    :return: 处理后的整数型boxes数组（指定数量）
    """
    # 1. 转换为numpy数组，统一处理格式
    boxes = np.array(original_boxes, dtype=np.float32)

    # 输入校验：确保是1维(单box)或2维(批量box)
    if boxes.ndim not in [1, 2]:
        raise ValueError("原始boxes必须是(4,) 或 (N,4) 形状的数组/列表")
    if boxes.shape[-1] != 4:
        raise ValueError("每个box必须包含4个坐标值（xmin, ymin, xmax, ymax）")

    # 2. 浮点坐标取整（截断小数，替代原四舍五入）
    boxes_rounded = np.trunc(boxes).astype(np.int32)

    # 3. 调整坐标顺序
    boxes_processed = boxes_rounded[..., order]  # ... 适配单/批量box

    # 4. 按指定数量返回box
    if return_num is None:
        # 不限制数量，返回全部
        return boxes_processed
    else:
        # 校验return_num为整数
        if not isinstance(return_num, int):
            raise TypeError("return_num必须为整数（None/正/负）")

        # 单box场景：若指定数量≠1，需特殊处理
        if boxes_processed.ndim == 1:
            if return_num == 1:
                return boxes_processed
            else:
                raise ValueError("单box场景下，return_num只能为1或None")

        # 批量box场景：按数量截取
        total_num = boxes_processed.shape[0]
        # 处理超出总数的情况（若指定数量大于总数，返回全部）
        if abs(return_num) >= total_num:
            return boxes_processed
        # 正数：取前N个；负数：取后N个
        if return_num > 0:
            return boxes_processed[:return_num]
        else:
            return boxes_processed[return_num:]


# -------------------------- 测试用例（新增：测试patch/scale功能） --------------------------
if __name__ == "__main__":
    # 配置路径（请根据实际目录修改！）
    ANNOTATION_DIR = r"/mnt/mydisk/wjj/dataset/Sheep_obb/DOTA"  # 标注文件目录
    IMAGE_DIR = r"/mnt/mydisk/wjj/dataset/Sheep_obb/img"  # 图像文件目录
    SAVE_DIR = r"/mnt/mydisk/wjj/dataset/Sheep_obb/see_output"  # 可视化结果保存目录

    # 创建保存目录
    os.makedirs(SAVE_DIR, exist_ok=True)

    # 测试加载器（新增patch/scale参数）
    loader = SheepOBBDatasetLoader(
        annotation_dir=ANNOTATION_DIR,
        image_dir=IMAGE_DIR,
        scale_mode='none',  # 无缩放模式
        box_number=3,  # 每个图取前3个BBox生成patch
        scale_number=20,  # scale量化为0~19
        exemplar_size=(128, 128)  # patch resize到128x128
    )
    test_filenames = loader.get_all_filenames()[:5]  # 测试前5张图
    if not test_filenames:
        raise FileNotFoundError("未找到有效图像-标注对，请检查路径是否正确")

    # 测试patch/scale生成
    print("\n=== 测试patch/scale生成 ===")
    for idx, fname in enumerate(test_filenames):
        data = loader[idx]
        print(f"\n{fname} 数据信息：")
        print(f"  图片尺寸：{data['orig_size']}")
        print(f"  BBox数量：{len(data['bboxes'])}")
        print(f"  选中的BBox数量：{len(data['bboxes_selected'])}" if 'bboxes_selected' in data else "  无选中BBox")
        print(f"  Patch数量：{data['patches'].shape[0] if len(data['patches']) > 0 else 0}")
        print(f"  Patch形状：{data['patches'].shape if len(data['patches']) > 0 else '空'}")
        print(f"  Scale值：{data['scales'].tolist() if len(data['scales']) > 0 else '空'}")

    # 原有可视化测试保持不变
    print("\n=== 测试可视化（OBB+BBox+中心点）===")
    for fname in test_filenames:
        data = loader[loader.get_all_filenames().index(fname)]
        img = data['image']
        obbs = data['obbs']
        bboxes = data['bboxes']
        centers = data['centers']
        classes = data['classes']

        save_path = os.path.join(SAVE_DIR, f"sheep_patch_test_{os.path.basename(fname)}")
        visualize_obb_bbox_annotations(
            img=img,
            obbs=obbs,
            bboxes=bboxes,
            centers=centers,
            classes=classes,
            save_path=save_path
        )

    print("\n所有测试完成！")