import os
import math
import json
import argparse
from os.path import exists, join
from typing import List, Optional

import numpy as np
import cv2
from tqdm import tqdm

from shi_segment_anything import sam_model_registry
from shi_segment_anything.automatic_mask_generator import SamAutomaticMaskGenerator

from myutil.SheepOBB import SheepOBBDatasetLoader
from myutil.localization import match_points, calculate_precision_recall_f1


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


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


def parse_args():
    parser = argparse.ArgumentParser(description="Sheep OBB counting/localization with SAM (training-free)")
    parser.add_argument("--image_dir", type=str, default="./dataset/Sheep_obb/img", help="图像目录")
    parser.add_argument("--annotation_dir", type=str, default="./dataset/Sheep_obb/DOTA", help="DOTA格式OBB标注txt目录")
    parser.add_argument("--split_file", type=str, default="", help="可选：split txt，每行一个文件名")
    parser.add_argument("--split_tag", type=str, default="test", help="仅用于输出目录/日志标识")

    parser.add_argument("--output_dir", type=str, default="./logsSave/Sheep", help="输出目录")
    parser.add_argument("--device", type=str, default="cuda:0", help="设备，如 cuda:0 / cpu")

    parser.add_argument("--sam_checkpoint", type=str, default="./pretrain/sam_vit_b_01ec64.pth", help="SAM权重路径")
    parser.add_argument("--model_type", type=str, default="vit_b", help="sam_model_registry key，如 vit_b")

    parser.add_argument("--prompt_type", type=str, default="box", choices=["box", "point"], help="prompt类型")
    parser.add_argument("--prompt_box_num", type=int, default=3, help="每张图用于prompt的示例框/点数量（不要用全量）")
    parser.add_argument("--prompt_select", type=str, default="head", choices=["head", "random"], help="示例框选择策略")
    parser.add_argument("--seed", type=int, default=0, help="随机种子（prompt_select=random时生效）")

    parser.add_argument("--distance_thresh", type=float, default=10.0, help="定位匹配距离阈值（像素）")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # -------------------- 输出目录 --------------------
    _ensure_dir(args.output_dir)
    _ensure_dir(join(args.output_dir, "logs"))
    _ensure_dir(join(args.output_dir, args.split_tag))
    _ensure_dir(join(args.output_dir, args.split_tag, args.prompt_type))

    log_path = join(args.output_dir, "logs", f"log-{args.split_tag}-{args.prompt_type}.csv")
    log_f = open(log_path, "w", encoding="utf-8")
    log_f.write("image_id,pred_count,gt_count,count_abs_err,count_rel_err,f1,precision,recall,prompt_k\n")
    log_f.flush()

    # -------------------- 数据集（解析OBB->BBox/center） --------------------
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

    # -------------------- SAM + 自动mask生成器 --------------------
    sam = sam_model_registry[args.model_type](checkpoint=args.sam_checkpoint)
    sam.to(device=args.device)
    mask_generator = SamAutomaticMaskGenerator(model=sam)

    # -------------------- 指标累计 --------------------
    MAE = 0.0
    RMSE = 0.0
    NAE = 0.0
    SRE = 0.0
    total_f1 = 0.0
    total_precision = 0.0
    total_recall = 0.0

    rng = np.random.default_rng(args.seed)

    for i, fname in tqdm(enumerate(im_ids), total=len(im_ids), desc="处理Sheep图片"):
        ann = loader.get_annotations(fname, return_scaled=False)
        bboxes = np.asarray(ann["bboxes"], dtype=np.float32)  # (N,4) xyxy
        centers = np.asarray(ann["centers"], dtype=np.float32)  # (N,2) xy

        gt_cnt = int(centers.shape[0])

        img_path = join(args.image_dir, fname)
        image = cv2.imread(img_path)
        if image is None:
            # 跳过坏图
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 选择用于prompt的少量示例（不要用全量标注）
        prompt_idx = _select_prompt_indices(
            total=gt_cnt,
            k=args.prompt_box_num,
            mode=args.prompt_select,
            rng=rng,
        )

        if args.prompt_type == "box":
            ref_prompt = bboxes[prompt_idx].tolist() if gt_cnt > 0 else []
        else:  # point
            ref_prompt = centers[prompt_idx].tolist() if gt_cnt > 0 else []

        # 推理：生成所有实例mask（计数=mask数量；定位=mask对应的point_coords）
        # 注意：automatic_mask_generator 需要至少一个 box/point 作为 exemplar 来提取 target embedding；
        # 若该图没有目标（gt_cnt=0）或 ref_prompt 为空，则直接返回空预测。
        if len(ref_prompt) == 0:
            masks = []
        else:
            masks = mask_generator.generate(image, ref_prompt)
        pred_cnt = int(len(masks))

        # 计数指标
        err = abs(gt_cnt - pred_cnt)
        MAE += err
        RMSE += err ** 2
        rel_err = (err / gt_cnt) if gt_cnt > 0 else 0.0
        NAE += rel_err
        SRE += (err ** 2) / gt_cnt if gt_cnt > 0 else 0.0

        # 定位指标：直接对 SAM 输出的 point_coords 与 GT centers 做匹配
        pred_points = []
        for m in masks:
            pc = m.get("point_coords", None)
            if pc is None or len(pc) == 0:
                continue
            pred_points.append(pc[0])  # [x,y]
        pred_points = np.asarray(pred_points, dtype=np.float32)
        tp, fp, fn = match_points(pred_points, centers, distance_thresh=float(args.distance_thresh))
        precision, recall, f1 = calculate_precision_recall_f1(tp, fp, fn)
        total_f1 += f1
        total_precision += precision
        total_recall += recall

        # 写日志
        log_f.write(
            f"{fname},{pred_cnt},{gt_cnt},{err},{rel_err:.6f},{f1:.6f},{precision:.6f},{recall:.6f},{len(prompt_idx)}\n"
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
        f"[{args.split_tag} | {args.prompt_type}] "
        f"MAE={avg_MAE:.4f}, RMSE={avg_RMSE:.4f}, NAE={avg_NAE:.4f}, SRE={avg_SRE:.4f}, "
        f"F1={avg_f1:.4f}, P={avg_precision:.4f}, R={avg_recall:.4f}\n"
    )
    print(summary)
    log_f.write("\n" + summary)
    log_f.close()

