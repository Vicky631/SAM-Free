import os
import sys
import math
import json
import argparse
from datetime import datetime
from os.path import exists, join, abspath, dirname
from typing import List, Optional

import numpy as np
import cv2
import torch
from scipy.ndimage import gaussian_filter
from tqdm import tqdm
from PIL import Image
from torchvision.transforms import transforms as T

from shi_segment_anything import sam_model_registry
from shi_segment_anything.automatic_mask_generator import SamAutomaticMaskGenerator

from myutil.SheepOBB import SheepOBBDatasetLoader
from myutil.FSC_147 import FSC147DatasetLoader
from myutil.localization import evaluate_detection_metrics, visualize_masks, get_pred_points_from_density


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def points_to_density_map(pred_points, image, sigma=4):
    """
    将参考点坐标转换为密度图（从 main-fsc147.py 借鉴）
    Args:
        pred_points: 参考点列表，格式为[[x1,y1], [x2,y2], ...]
        image: 输入图像（ndarray），用于获取尺寸 (H, W, C)
        sigma: 高斯核的标准差，控制点的扩散范围（默认4，可根据图像尺寸调整）
    Returns:
        pred_density_map: 预测密度图 (torch.Tensor, 形状 (H, W))
    """
    # 1. 获取图像的高度H和宽度W
    H, W = image.shape[0], image.shape[1]  # image是ndarray，形状是(H, W, 3)

    # 2. 初始化密度图（全零，numpy数组）
    density_map = np.zeros((H, W), dtype=np.float32)

    # 3. 遍历每个参考点，在对应位置添加1（离散点）
    for (x, y) in pred_points:
        # 确保坐标在图像范围内（防止越界）
        x = np.clip(x, 0, W - 1)
        y = np.clip(y, 0, H - 1)
        # 转换为整数坐标（像素是离散的）
        x_int, y_int = int(np.round(x)), int(np.round(y))
        density_map[y_int, x_int] += 1  # 注意：图像的y是行，x是列

    # 4. 对离散点进行高斯滤波，生成连续的密度图
    density_map = gaussian_filter(density_map, sigma=sigma)

    # 5. 转换为torch.Tensor（形状 (H, W)）
    pred_density_map = torch.from_numpy(density_map)

    return pred_density_map


def _load_split_list(
    split_file: str,
    available_filenames: List[str],
) -> List[str]:
    """
    split_file: 每行一个文件名（可带扩展名，也可不带）
    返回：在 available_filenames 中可匹配到的文件名列表（保持split顺序）
    """
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


def _select_prompt_indices(
    total: int,
    k: int,
    mode: str,
    rng: np.random.Generator,
) -> np.ndarray:
    if total <= 0:
        return np.array([], dtype=np.int64)
    k = max(1, int(k))
    k = min(k, total)
    if mode == "head":
        return np.arange(k, dtype=np.int64)
    if mode == "random":
        return rng.choice(total, size=k, replace=False).astype(np.int64)
    raise ValueError(f"Unknown prompt_select: {mode}")


def points_to_density_map(pred_points, image, sigma=4):
    """
    将参考点坐标转换为密度图（参考 main-fsc147.py）
    Args:
        pred_points: 参考点列表，格式为[[x1,y1], [x2,y2], ...]
        image: 输入图像（ndarray），用于获取尺寸 (H, W, C)
        sigma: 高斯核的标准差，控制点的扩散范围（默认4，可根据图像尺寸调整）
    Returns:
        pred_density_map: 预测密度图 (torch.Tensor, 形状 (H, W))
    """
    # 1. 获取图像的高度H和宽度W
    H, W = image.shape[0], image.shape[1]  # image是ndarray，形状是(H, W, 3)

    # 2. 初始化密度图（全零，numpy数组）
    density_map = np.zeros((H, W), dtype=np.float32)

    # 3. 遍历每个参考点，在对应位置添加1（离散点）
    for (x, y) in pred_points:
        # 确保坐标在图像范围内（防止越界）
        x = np.clip(x, 0, W - 1)
        y = np.clip(y, 0, H - 1)
        # 转换为整数坐标（像素是离散的）
        x_int, y_int = int(np.round(x)), int(np.round(y))
        density_map[y_int, x_int] += 1  # 注意：图像的y是行，x是列

    # 4. 对离散点进行高斯滤波，生成连续的密度图
    density_map = gaussian_filter(density_map, sigma=sigma)

    # 5. 转换为torch.Tensor（形状 (H, W)）
    pred_density_map = torch.from_numpy(density_map)

    return pred_density_map


def parse_args():
    parser = argparse.ArgumentParser(description="Multi-dataset counting/localization with SAM (training-free)")
    
    # 数据集选择
    parser.add_argument("--dataset", type=str, default="sheep", choices=["sheep", "fsc147"], help="数据集类型：sheep 或 fsc147")
    
    # Sheep 数据集参数
    parser.add_argument("--image_dir", type=str, default="/ZHANGyong/wjj/dataset/Sheep_obb/img/", help="图像目录（Sheep数据集）")
    parser.add_argument("--annotation_dir", type=str, default="/ZHANGyong/wjj/dataset/Sheep_obb/DOTA/", help="DOTA格式OBB标注txt目录（Sheep数据集）")
    parser.add_argument("--split_file", type=str, default="", help="可选：split txt，每行一个文件名（Sheep数据集）")
    parser.add_argument("--split_tag", type=str, default="test", help="仅用于输出目录/日志标识")

    # FSC147 数据集参数
    parser.add_argument("--fsc147_annotation", type=str, default="/ZHANGyong/wjj/dataset/FSC_147/annotation_FSC147_384_with_gt.json", help="FSC147标注文件路径")
    parser.add_argument("--fsc147_images", type=str, default="/ZHANGyong/wjj/dataset/FSC_147/images_384_VarV2", help="FSC147图像目录")
    parser.add_argument("--fsc147_split", type=str, default="/ZHANGyong/wjj/dataset/FSC_147/Train_Test_Val_FSC_147.json", help="FSC147数据集划分文件")
    parser.add_argument("--test_split", type=str, default="test", choices=["train", "test", "val"], help="测试集划分（FSC147使用）")

    parser.add_argument("--output_dir", type=str, default="./logsSave", help="输出根目录")
    # 主要参数
    parser.add_argument("--device", type=str, default="cuda:7", help="设备，如 cuda:0 / cpu")

    parser.add_argument("--sam_checkpoint", type=str, default="/ZHANGyong/wjj/online_models/sam/sam_vit_b_01ec64.pth", help="SAM权重路径")
    parser.add_argument("--model_type", type=str, default="vit_b", help="sam_model_registry key，如 vit_b")

    parser.add_argument("--prompt_type", type=str, default="box", choices=["box", "point"], help="prompt类型")
    parser.add_argument("--prompt_box_num", type=int, default=3, help="每张图用于prompt的示例框/点数量（不要用全量）")
    parser.add_argument("--prompt_select", type=str, default="head", choices=["head", "random"], help="示例框选择策略")
    parser.add_argument("--seed", type=int, default=0, help="随机种子（prompt_select=random时生效）")

    parser.add_argument("--distance_thresh", type=float, default=10.0, help="定位匹配距离阈值（像素）")
    parser.add_argument("--save_vis", action="store_true", help="是否保存可视化图像（预测点与GT点）")
    parser.add_argument("--vis_sigma", type=float, default=4.0, help="密度图高斯滤波的sigma值")
    parser.add_argument("--points_per_side", type=int, default=64, help="网格每边点数基准，越大越密，默认64")
    parser.add_argument("--points_per_batch", type=int, default=64, help="SAM 每次处理的候选点数，显存富裕时可增大如 128/256")
    # BMNet 候选点（方式 A：可选外部点）
    parser.add_argument("--use_bmnet", action="store_true", help="使用 BMNet 生成候选点替代均匀网格")
    parser.add_argument("--bmnet_cfg", type=str, default="/ZHANGyong/wjj/BMNet/config/test_bmnet+.yaml", help="BMNet 配置 yaml 路径，默认使用 BMNet/config/test_bmnet+.yaml")
    parser.add_argument("--bmnet_checkpoint", type=str, default="/ZHANGyong/wjj/BMNet/checkpoints/bmnet+_pretrained/model_best.pth", help="BMNet 权重路径，默认由 cfg 推导 snapshot/exp/VAL.resume")
    parser.add_argument("--bmnet_device", type=str, default="", help="BMNet 运行设备，默认与 --device 相同")
    parser.add_argument("--verbose_point_stats", action="store_true", help="打印候选点各阶段丢弃数量（point_mask/相似度/IoU/稳定性/NMS）")
    parser.add_argument("--pred_iou_thresh", type=float, default=0.88, help="SAM mask 质量阈值，仅保留 predicted_iou > 该值；用 BMNet 时可适当降低如 0.7")
    parser.add_argument("--stability_score_thresh", type=float, default=0.85, help="SAM 稳定性阈值，仅保留 stability >= 该值；用 BMNet 时可适当降低如 0.7")
    parser.add_argument("--use_ref_sim_filter", action="store_true", help="用「与 ref 的相似度」替代 pred_iou 做质量过滤（留更像 ref 的 mask）")
    parser.add_argument("--ref_sim_thresh", type=float, default=0.3, help="与 ref 的余弦相似度阈值，仅当 --use_ref_sim_filter 时生效；范围约 [-1,1]，默认 0.3")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # -------------------- 入参打印 --------------------
    print("=" * 60)
    print("项目入参 (Project Arguments)")
    print("=" * 60)
    for k, v in vars(args).items():
        print(f"  {k}: {v}")
    print("=" * 60)

    # -------------------- 输出目录 --------------------
    # 根据数据集类型调整输出目录
    if args.dataset == "sheep":
        final_output_dir = join(args.output_dir, "Sheep")
        split_tag = args.split_tag
    else:  # fsc147
        final_output_dir = join(args.output_dir, "FSC147")
        split_tag = args.test_split
    
    _ensure_dir(final_output_dir)
    _ensure_dir(join(final_output_dir, "logs"))
    prompt_tag = (args.prompt_type if args.dataset == "sheep" else "box") + ("_bmnet" if args.use_bmnet else "")
    _ensure_dir(join(final_output_dir, split_tag))
    _ensure_dir(join(final_output_dir, split_tag, prompt_tag))
    if args.save_vis:
        now = datetime.now()
        vis_suffix = f"vis_{now.month:02d}_{now.day:02d}_{now.hour:02d}_{now.minute:02d}"
        vis_dir = join(final_output_dir, split_tag, prompt_tag, vis_suffix)
        _ensure_dir(vis_dir)

    log_path = join(final_output_dir, "logs", f"log-{split_tag}-{prompt_tag}.csv")
    log_f = open(log_path, "w", encoding="utf-8")
    log_f.write("image_id,pred_count,gt_count,count_abs_err,count_rel_err,f1,precision,recall,prompt_k\n")
    log_f.flush()

    # -------------------- 数据集加载 --------------------
    if args.dataset == "sheep":
        loader = SheepOBBDatasetLoader(
            annotation_dir=args.annotation_dir,
            image_dir=args.image_dir,
            scale_mode="none",
            box_number=max(1, args.prompt_box_num),
        )
        all_filenames = loader.get_all_filenames()

        if args.split_file and exists(args.split_file):
            im_ids = _load_split_list(args.split_file, all_filenames)
        else:
            im_ids = all_filenames
    else:  # fsc147
        import json
        
        # 加载FSC147数据集
        loader = FSC147DatasetLoader(
            annotation_file=args.fsc147_annotation,
            image_root=args.fsc147_images,
            scale_mode='none'
        )
        
        # 加载数据集划分
        with open(args.fsc147_split, 'r') as f:
            data_split = json.load(f)
        
        im_ids = data_split[args.test_split]
        all_filenames = loader.get_all_filenames()
        
        # 验证文件存在性
        valid_im_ids = [fname for fname in im_ids if fname in all_filenames]
        if len(valid_im_ids) != len(im_ids):
            print(f"警告：{len(im_ids) - len(valid_im_ids)} 个文件在数据集中未找到")
        im_ids = valid_im_ids

    # -------------------- BMNet（可选：生成候选点） --------------------
    bmnet_model = None
    bmnet_device = args.bmnet_device or args.device
    bmnet_img_transform = None
    if args.use_bmnet:
        _script_dir = dirname(abspath(__file__))
        _wjj_root = dirname(_script_dir)
        _bmnet_path = join(_wjj_root, "BMNet")
        if _bmnet_path not in sys.path:
            sys.path.insert(0, _bmnet_path)
        try:
            from config import cfg as bmnet_cfg
            cfg_path = args.bmnet_cfg or join(_bmnet_path, "config", "test_bmnet+.yaml")
            if not exists(cfg_path):
                raise FileNotFoundError(f"BMNet 配置不存在: {cfg_path}")
            bmnet_cfg.merge_from_file(cfg_path)
            # snapshot 相对 BMNet 目录解析，与当前工作目录无关
            bmnet_cfg.DIR.snapshot = join(_bmnet_path, bmnet_cfg.DIR.snapshot) if not os.path.isabs(bmnet_cfg.DIR.snapshot) else bmnet_cfg.DIR.snapshot
            bmnet_cfg.DIR.output_dir = join(bmnet_cfg.DIR.snapshot, bmnet_cfg.DIR.exp)
            bmnet_cfg.VAL.resume = join(bmnet_cfg.DIR.output_dir, bmnet_cfg.VAL.resume)
            ckpt_path = args.bmnet_checkpoint or bmnet_cfg.VAL.resume
            if not exists(ckpt_path):
                raise FileNotFoundError(f"BMNet 权重不存在: {ckpt_path}")
            from models import build_model as build_bmnet
            bmnet_model = build_bmnet(bmnet_cfg)
            ckpt = torch.load(ckpt_path, map_location="cpu")
            bmnet_model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
            bmnet_model.to(bmnet_device)
            bmnet_model.eval()
            bmnet_img_transform = T.Compose([
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
            print(f"  BMNet 已加载: {ckpt_path}")
        except Exception as e:
            print(f"  BMNet 加载失败，将使用网格候选点: {e}")
            args.use_bmnet = False

    # -------------------- SAM + 自动mask生成器 --------------------
    sam = sam_model_registry[args.model_type](checkpoint=args.sam_checkpoint)
    sam.to(device=args.device)
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=args.points_per_side,
        points_per_batch=args.points_per_batch,
        pred_iou_thresh=args.pred_iou_thresh,
        stability_score_thresh=args.stability_score_thresh,
        use_ref_sim_for_filter=args.use_ref_sim_filter,
        ref_sim_thresh=args.ref_sim_thresh,
    )

    # -------------------- 指标累计 --------------------
    MAE = 0.0
    RMSE = 0.0
    NAE = 0.0
    SRE = 0.0
    total_f1 = 0.0
    total_precision = 0.0
    total_recall = 0.0

    rng = np.random.default_rng(args.seed)

    dataset_name = "Sheep" if args.dataset == "sheep" else "FSC147"
    for i, fname in tqdm(enumerate(im_ids), total=len(im_ids), desc=f"处理{dataset_name}图片"):
        if args.dataset == "sheep":
            # Sheep 数据集处理
            ann = loader.get_annotations(fname, return_scaled=False)
            bboxes = np.asarray(ann["bboxes"], dtype=np.float32)  # (N,4) xyxy
            centers = np.asarray(ann["centers"], dtype=np.float32)  # (N,2) xy
            gt_cnt = int(centers.shape[0])

            img_path = join(args.image_dir, fname)
            image = cv2.imread(img_path)
            if image is None:
                continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # 选择用于prompt的少量示例（不要用全量标注）
            prompt_idx = _select_prompt_indices(
                total=gt_cnt,
                k=args.prompt_box_num,
                mode=args.prompt_select,
                rng=rng,
            )
            prompt_k = len(prompt_idx)  # 记录实际使用的prompt数量

            if args.prompt_type == "box":
                ref_prompt = bboxes[prompt_idx].tolist() if gt_cnt > 0 else []
            else:  # point
                ref_prompt = centers[prompt_idx].tolist() if gt_cnt > 0 else []
                
        else:  # fsc147
            # FSC147 数据集处理
            ann = loader.get_annotations(fname, return_scaled=False, return_numpy=True)
            points = ann["points"]  # (N,2) xy - 真实点位置
            boxes = ann["boxes"]    # (M,4) xyxy - 示例框
            gt_cnt = int(points.shape[0])

            image = loader.get_image(fname)
            image = np.array(image)  # PIL -> numpy

            # FSC147 使用示例框作为 prompt（通常3个框）
            prompt_k = min(len(boxes), args.prompt_box_num) if len(boxes) > 0 else 0
            if prompt_k > 0:
                prompt_idx = _select_prompt_indices(
                    total=len(boxes),
                    k=prompt_k,
                    mode=args.prompt_select,
                    rng=rng,
                )
                ref_prompt = boxes[prompt_idx].tolist()
            else:
                ref_prompt = []
                
            # FSC147 的真实标注是点，用于定位评估
            centers = points

        # 可选：用 BMNet 密度图提取候选点，替代均匀网格
        external_point_coords = None
        if args.use_bmnet and bmnet_model is not None and bmnet_img_transform is not None:
            try:
                if args.dataset == "sheep":
                    # Sheep 数据集：使用 patches 和 scales
                    all_fnames = loader.get_all_filenames()
                    idx = all_fnames.index(fname) if fname in all_fnames else -1
                    if idx >= 0:
                        data = loader[idx]
                        pt = data["patches"]
                        sc = data["scales"]
                        if pt.numel() == 0:
                            pt = torch.zeros(0, 3, 128, 128)
                            sc = torch.zeros(0, dtype=torch.long)
                        img_t = bmnet_img_transform(Image.fromarray(image)).unsqueeze(0).to(bmnet_device)
                        patches_dict = {
                            "patches": pt.unsqueeze(0).to(bmnet_device),
                            "scale_embedding": sc.unsqueeze(0).to(bmnet_device),
                        }
                        with torch.no_grad():
                            out = bmnet_model(img_t, patches_dict, is_train=False)
                        density_map = out.squeeze(0).squeeze(0)
                else:  # fsc147
                    # FSC147 数据集：从示例框生成 patches
                    if len(ref_prompt) > 0:
                        # 从示例框提取 patches
                        patches_list = []
                        scales_list = []
                        
                        for box in ref_prompt:
                            x1, y1, x2, y2 = map(int, box)
                            # 确保坐标在图像范围内
                            h, w = image.shape[:2]
                            x1, y1 = max(0, x1), max(0, y1)
                            x2, y2 = min(w, x2), min(h, y2)
                            
                            if x2 > x1 and y2 > y1:
                                patch = image[y1:y2, x1:x2]
                                patch_pil = Image.fromarray(patch).resize((128, 128))
                                patch_tensor = bmnet_img_transform(patch_pil)
                                patches_list.append(patch_tensor)
                                # 简单的尺度编码：根据patch大小
                                patch_area = (x2 - x1) * (y2 - y1)
                                scale = min(3, max(0, int(np.log2(patch_area / 1000))))  # 简单的尺度分级
                                scales_list.append(scale)
                        
                        if patches_list:
                            pt = torch.stack(patches_list)
                            sc = torch.tensor(scales_list, dtype=torch.long)
                        else:
                            pt = torch.zeros(0, 3, 128, 128)
                            sc = torch.zeros(0, dtype=torch.long)
                            
                        img_t = bmnet_img_transform(Image.fromarray(image)).unsqueeze(0).to(bmnet_device)
                        patches_dict = {
                            "patches": pt.unsqueeze(0).to(bmnet_device),
                            "scale_embedding": sc.unsqueeze(0).to(bmnet_device),
                        }
                        with torch.no_grad():
                            out = bmnet_model(img_t, patches_dict, is_train=False)
                        density_map = out.squeeze(0).squeeze(0)
                    else:
                        density_map = None
                
                # 上采样密度图到原图尺寸
                if density_map is not None:
                    h_orig, w_orig = image.shape[0], image.shape[1]
                    if density_map.shape[0] != h_orig or density_map.shape[1] != w_orig:
                        density_map = torch.nn.functional.interpolate(
                            density_map.unsqueeze(0).unsqueeze(0),
                            size=(h_orig, w_orig),
                            mode="bilinear",
                        ).squeeze(0).squeeze(0)
                    external_point_coords = get_pred_points_from_density(density_map)
                    if isinstance(external_point_coords, np.ndarray) and len(external_point_coords) == 0:
                        external_point_coords = None
            except Exception as e:
                print(f"BMNet 处理失败 ({fname}): {e}")
                external_point_coords = None

        # 推理：生成所有实例mask（计数=mask数量；定位=mask对应的point_coords）
        # 注意：automatic_mask_generator 需要至少一个 box/point 作为 exemplar 来提取 target embedding；
        # 若该图没有目标（gt_cnt=0）或 ref_prompt 为空，则直接返回空预测。
        if len(ref_prompt) == 0:
            masks = []
        else:
            masks = mask_generator.generate(
                image, ref_prompt,
                point_coords=external_point_coords,
                verbose_point_stats=args.verbose_point_stats,
            )
        pred_cnt = int(len(masks))

        # 计数指标
        err = abs(gt_cnt - pred_cnt)
        MAE += err
        RMSE += err ** 2
        rel_err = (err / gt_cnt) if gt_cnt > 0 else 0.0
        NAE += rel_err
        SRE += (err ** 2) / gt_cnt if gt_cnt > 0 else 0.0

        # 定位指标：提取预测点并生成密度图（用于可视化与评估）
        pred_points = []
        for m in masks:
            pc = m.get("point_coords", None)
            if pc is None or len(pc) == 0:
                continue
            # 兼容不同格式：可能是 [x,y] 或 [[x,y]]
            if isinstance(pc[0], (list, tuple, np.ndarray)):
                pred_points.append(pc[0])
            else:
                pred_points.append(pc)
        pred_points = np.asarray(pred_points, dtype=np.float32) if len(pred_points) > 0 else np.array([], dtype=np.float32).reshape(0, 2)
        
        # 生成密度图（用于可视化与评估）
        pred_density_map = points_to_density_map(pred_points, image, sigma=args.vis_sigma)
        
        # 计算定位指标（使用密度图方式，与 FSC147 保持一致）
        # 如果启用可视化，evaluate_detection_metrics 会自动保存可视化图像
        f1, precision, recall = evaluate_detection_metrics(
            pred_density_map=pred_density_map,
            gt_points=centers,
            distance_thresh=args.distance_thresh,
            vis_dir=vis_dir if args.save_vis else None,
            image_id=fname
        )

        # 可视化 masks（与 gt_points、pred_points 保存到同一目录）
        if args.save_vis:
            fname_base = os.path.splitext(os.path.basename(str(fname)))[0]
            fname_safe = fname_base.replace("/", "_").replace("\\", "_")
            mask_vis_path = join(vis_dir, f"{fname_safe}_masks.png")
            visualize_masks(image, masks, mask_vis_path, title="Masks")
        print(f"  f1: {f1}, precision: {precision}, recall: {recall}")
        total_f1 += f1
        total_precision += precision
        total_recall += recall

        # 写日志
        log_f.write(
            f"{fname},{pred_cnt},{gt_cnt},{err},{rel_err:.6f},{f1:.6f},{precision:.6f},{recall:.6f},{prompt_k}\n"
        )
        log_f.flush()

    n = max(1, len(im_ids))
    avg_MAE = MAE / n
    avg_RMSE = math.sqrt(RMSE / n)
    avg_NAE = NAE / n
    avg_SRE = math.sqrt(SRE / n) if n > 0 else 0.0
    avg_f1 = total_f1 / n
    avg_precision = total_precision / n
    avg_recall = total_recall / n

    summary = (
        f"[{split_tag} | {prompt_tag}] "
        f"MAE={avg_MAE:.4f}, RMSE={avg_RMSE:.4f}, NAE={avg_NAE:.4f}, SRE={avg_SRE:.4f}, "
        f"F1={avg_f1:.4f}, P={avg_precision:.4f}, R={avg_recall:.4f}\n"
    )
    print(summary)
    log_f.write("\n" + summary)
    log_f.close()

