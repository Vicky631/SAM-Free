import os
import torch
import torchvision
import argparse
import json
import numpy as np
import os
import copy
import time
import cv2
import math
from tqdm import tqdm
from os.path import exists, join
from PIL import Image
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import clip
from scipy.ndimage import gaussian_filter  # 补充缺失的导入
from shi_segment_anything import sam_model_registry, SamPredictor
from shi_segment_anything.automatic_mask_generator import SamAutomaticMaskGenerator
# 注意：确保 utils.py 和 myutil 目录存在并能导入
from utils import *
from myutil.FSC_147 import FSC147DatasetLoader
from myutil.localization import evaluate_detection_metrics

parser = argparse.ArgumentParser(description="Counting with SAM")
parser.add_argument("-dp", "--data_path", type=str, default='/data/wjj/dataset/FSC_147/',
                    help="Path to the FSC147 dataset")
parser.add_argument("-o", "--output_dir", type=str, default="./logsSave/FSC147", help="/Path/to/output/logs/")
parser.add_argument("-ts", "--test-split", type=str, default='test', choices=["train", "test", "val"],
                    help="what data split to evaluate on on")
parser.add_argument("-pt", "--prompt-type", type=str, default='box', choices=["box", "point", "text"],
                    help="what type of information to prompt")
parser.add_argument("-d", "--device", type=str, default='cuda:0', help="device")
args = parser.parse_args()


def points_to_density_map(pred_points, image, sigma=4):
    """
    将参考点坐标转换为密度图
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


if __name__ == "__main__":

    data_path = args.data_path
    anno_file = data_path + 'annotation_FSC_147_384.json'
    # anno_file = data_path + 'annotation_FSC147_384.json'
    data_split_file = data_path + 'Train_Test_Val_FSC_147.json'
    im_dir = data_path + 'images_384_VarV2'

    if not exists(args.output_dir):
        os.mkdir(args.output_dir)
        os.mkdir(args.output_dir + '/logs')

    if not exists(args.output_dir + '/%s' % args.test_split):
        os.mkdir(args.output_dir + '/%s' % args.test_split)

    if not exists(args.output_dir + '/%s/%s' % (args.test_split, args.prompt_type)):
        os.mkdir(args.output_dir + '/%s/%s' % (args.test_split, args.prompt_type))

    log_file = open(args.output_dir + '/logs/log-%s-%s.txt' % (args.test_split, args.prompt_type), "w")
    # 写入表头
    log_file.write("图片ID,预测数量,真实数量,计数误差,F1分数,精确率,召回率,相对误差\n")
    log_file.flush()

    with open(anno_file) as f:
        annotations = json.load(f)

    with open(data_split_file) as f:
        data_split = json.load(f)

    if args.prompt_type == 'text':
        from shi_segment_anything.automatic_mask_generator_text import SamAutomaticMaskGenerator

        with open(data_path + 'ImageClasses_FSC_147.txt') as f:
            class_lines = f.readlines()

        class_dict = {}
        for cline in class_lines:
            strings = cline.strip().split('\t')
            class_dict[strings[0]] = strings[1]

        clip_model, _ = clip.load("/data/wjj/online_model/clip/ViT-B-16.pt", device=args.device)
        # clip_model, _ = clip.load("CS-ViT-B/16", device=args.device)
        clip_model.eval()

    sam_checkpoint = "/data/wjj/online_model/sam/sam_vit_b_01ec64.pth"
    model_type = "vit_b"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=args.device)

    mask_generator = SamAutomaticMaskGenerator(model=sam)

    # ====================== 初始化指标变量 ======================
    # 计数指标
    MAE = 0
    RMSE = 0
    NAE = 0
    SRE = 0
    # 定位指标（新增）
    total_f1 = 0.0
    total_precision = 0.0
    total_recall = 0.0
    # 存储单张图片的指标（用于后续分析）
    single_metrics = []

    im_ids = data_split[args.test_split]

    EVAL_CFG = {
        "distance_thresh": 10.0,  # 定位匹配距离阈值
        "dsizensity_map_size": (512, 512),  # 密度图固定尺寸
        "annotation_path": '/data/wjj/dataset/FSC_147/annotation_FSC147_384_with_gt.json',
        "image_root_path": '/data/wjj/dataset/FSC_147/images_384_VarV2',
        "csv_save_dir": "/data/wjj/BMNet/experiments/FSC147/eval_csv"  # CSV保存目录
    }

    FSC_loader = FSC147DatasetLoader(
        annotation_file=EVAL_CFG["annotation_path"],
        image_root=EVAL_CFG["image_root_path"],
        scale_mode='none'
    )

    # ====================== 遍历测试集 ======================
    for i, im_id in tqdm(enumerate(im_ids), desc="处理图片"):
        anno = annotations[im_id]
        bboxes = anno['box_examples_coordinates']
        dots = np.array(anno['points'])

        image = cv2.imread('{}/{}'.format(im_dir, im_id))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if args.prompt_type == 'text':
            cls_name = class_dict[im_id]
            input_prompt = get_clip_bboxs(clip_model, image, cls_name, args.device)
        else:
            input_prompt = list()
            for bbox in bboxes:
                x1 = bbox[0][0]
                y1 = bbox[0][1]
                x2 = bbox[2][0]
                y2 = bbox[2][1]
                if args.prompt_type == 'box':
                    input_prompt.append([x1, y1, x2, y2])
                elif args.prompt_type == 'point':
                    input_prompt.append([(x1 + x2) // 2, (y1 + y2) // 2])

        masks = mask_generator.generate(image, input_prompt)

        # ============================== 定位指标计算 ==============================
        # 提取预测点坐标
        pred_points = [m['point_coords'][0] if len(m['point_coords']) == 1 else m['point_coords'] for m in masks]
        # 生成密度图
        pred_density_map = points_to_density_map(pred_points, image, sigma=4)
        # 获取真实点坐标
        gt_points = dots
        # 计算定位指标
        f1, precision, recall = evaluate_detection_metrics(
            pred_density_map=pred_density_map,
            gt_points=gt_points,
            distance_thresh=EVAL_CFG["distance_thresh"]
        )

        # ============================== 计数指标计算 ==============================
        gt_cnt = dots.shape[0]
        pred_cnt = len(masks)
        err = abs(gt_cnt - pred_cnt)
        relative_err = err / gt_cnt if gt_cnt != 0 else 0  # 避免除零错误

        # ============================== 累加全局指标 ==============================
        # 计数指标
        MAE += err
        RMSE += err ** 2
        NAE += relative_err
        SRE += (err ** 2) / gt_cnt if gt_cnt != 0 else 0
        # 定位指标
        total_f1 += f1
        total_precision += precision
        total_recall += recall

        # ============================== 打印单张图片指标 ==============================
        single_metric_str = (
            f"【图片 {i + 1}/{len(im_ids)} - {im_id}】\n"
            f"  计数指标：预测={pred_cnt}, 真实={gt_cnt}, 误差={err}, 相对误差={relative_err:.4f}\n"
            f"  定位指标：F1={f1:.4f}, 精确率={precision:.4f}, 召回率={recall:.4f}\n"
        )
        print(single_metric_str)

        # ============================== 写入日志文件 ==============================
        log_file.write(f"{im_id},{pred_cnt},{gt_cnt},{err},{f1:.4f},{precision:.4f},{recall:.4f},{relative_err:.4f}\n")
        log_file.flush()

        # ============================== 保存单张图片指标 ==============================
        single_metrics.append({
            'image_id': im_id,
            'pred_count': pred_cnt,
            'gt_count': gt_cnt,
            'count_error': err,
            'relative_error': relative_err,
            'f1': f1,
            'precision': precision,
            'recall': recall
        })

        # Mask visualization（可选，注释保留）
        """
        fig = plt.figure()
        plt.axis('off')
        ax = plt.gca()
        ax.xaxis.set_major_locator(matplotlib.ticker.NullLocator())
        ax.yaxis.set_major_locator(matplotlib.ticker.NullLocator())
        plt.imshow(image)
        show_anns(masks, plt.gca())
        plt.savefig('%s/%s/%03d_mask.png'%(args.output_dir,args.test_split,i), bbox_inches='tight', pad_inches=0)
        plt.close()
        """

    # ====================== 计算数据集汇总指标 ======================
    num_images = len(im_ids)
    # 计数指标汇总
    avg_MAE = MAE / num_images
    avg_RMSE = math.sqrt(RMSE / num_images)
    avg_NAE = NAE / num_images
    avg_SRE = math.sqrt(SRE / num_images) if num_images > 0 else 0
    # 定位指标汇总
    avg_f1 = total_f1 / num_images
    avg_precision = total_precision / num_images
    avg_recall = total_recall / num_images

    # ====================== 打印数据集汇总指标 ======================
    summary_str = (
            "\n" + "=" * 60 + "\n"
                              f"【{args.test_split}集 - {args.prompt_type}提示】汇总指标\n"
                              "=" * 60 + "\n"
                                         f"计数指标：\n"
                                         f"  MAE (平均绝对误差)：{avg_MAE:.4f}\n"
                                         f"  RMSE (均方根误差)：{avg_RMSE:.4f}\n"
                                         f"  NAE (归一化绝对误差)：{avg_NAE:.4f}\n"
                                         f"  SRE (归一化均方根误差)：{avg_SRE:.4f}\n"
                                         "=" * 60 + "\n"
                                                    f"定位指标：\n"
                                                    f"  平均F1分数：{avg_f1:.4f}\n"
                                                    f"  平均精确率：{avg_precision:.4f}\n"
                                                    f"  平均召回率：{avg_recall:.4f}\n"
                                                    "=" * 60 + "\n"
    )
    print(summary_str)

    # ====================== 写入汇总指标到日志 ======================
    log_file.write("\n" + "=" * 60 + "\n")
    log_file.write(f"【{args.test_split}集 - {args.prompt_type}提示】汇总指标\n")
    log_file.write("=" * 60 + "\n")
    log_file.write(f"MAE (平均绝对误差)：{avg_MAE:.4f}\n")
    log_file.write(f"RMSE (均方根误差)：{avg_RMSE:.4f}\n")
    log_file.write(f"NAE (归一化绝对误差)：{avg_NAE:.4f}\n")
    log_file.write(f"SRE (归一化均方根误差)：{avg_SRE:.4f}\n")
    log_file.write("=" * 60 + "\n")
    log_file.write(f"平均F1分数：{avg_f1:.4f}\n")
    log_file.write(f"平均精确率：{avg_precision:.4f}\n")
    log_file.write(f"平均召回率：{avg_recall:.4f}\n")
    log_file.write("=" * 60 + "\n")
    log_file.close()

    # 可选：将所有单张图片指标保存为JSON文件，方便后续分析
    with open(args.output_dir + '/logs/single_metrics-%s-%s.json' % (args.test_split, args.prompt_type), 'w') as f:
        json.dump(single_metrics, f, indent=4)