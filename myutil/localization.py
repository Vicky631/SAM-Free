"""
PseCo项目中的定位模块，实现基于局部最大值检测的计数算法。
该模块用于密度图中的目标检测和计数任务，如人群计数、物体检测等计算机视觉应用。

文件作用：
- 实现局部最大值检测算法(LMDS)用于计数
- 提供预测和真实值两种版本的计数功能
- 生成可视化点图以展示检测到的目标位置
- 将计数结果和位置信息写入指定文件
- 新增：计算F1、AP（平均精度）、AR（平均召回率）评估指标

输入：
- 密度图张量（torch.Tensor）：包含目标分布信息的热力图
- 文件名（str）：当前处理的图像文件名
- 文件句柄（file object）：用于写入结果的文件流
- 真实密度图（torch.Tensor）：真实的密度图标注数据
- 真实点坐标（torch.Tensor/np.ndarray）：GT的点信息

输出：
- 计数值（int）：检测到的目标总数
- 关键点坐标数组（numpy.ndarray）：包含每个检测点位置的数组
- 可视化点图（numpy.ndarray）：带有标记的图像
- 更新的位置信息文件：包含文件名和坐标信息的文本文件
- F1、AP、AR（float）：定位精度评估指标
"""

import warnings

warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import math
import torch
import torch.nn.functional as F
import cv2
from scipy.spatial import distance
import os


def LMDS_counting(input, w_fname, f_loc):
    """
    对输入密度图执行局部最大值检测计数，识别并统计目标数量

    参数:
        input (torch.Tensor): 输入的密度图张量
        w_fname (str): 当前处理的文件名
        f_loc (file object): 输出文件句柄，用于写入计数结果

    返回:
        tuple: (计数值, 关键点坐标数组, 更新后的文件句柄)
    """
    input = input.unsqueeze(0)  # 在第0维增加一个维度，通常用于批次处理
    input_max = torch.max(input).item()  # 获取输入张量的最大值并转为Python标量

    ''' find local maxima'''  # 寻找局部最大值区域
    keep = F.max_pool2d(input, (3, 3), stride=1, padding=1)  # 3x3窗口最大池化，保留邻域最大值
    keep = (keep == input).float()  # 创建掩码：当前位置如果是邻域最大值则为1，否则为0
    input = keep * input  # 应用掩码，只保留局部最大值，其他位置设为0

    '''set the pixel value of local maxima as 1 for counting'''  # 将局部最大值像素设为1进行计数
    input[input < 60.0 / 255.0 * input_max] = 0  # 阈值过滤：低于阈值的局部最大值设为0
    input[input > 0] = 1  # 将剩余的正数设为1，准备计数

    # 负样本处理
    if input_max < 0.06:  # 如果输入最大值小于阈值0.06
        input = input * 0  # 则认为是负样本，清零所有值

    count = int(torch.sum(input).item())  # 统计非零元素个数作为计数值

    kpoint = input.data.squeeze(0).squeeze(0).cpu().numpy()  # 移除额外维度并转移到CPU内存转为numpy数组

    f_loc.write('{} {} '.format(w_fname, count))  # 将文件名和计数值写入输出文件
    return count, kpoint, f_loc  # 返回计数值、关键点数组和文件句柄


def generate_point_map(kpoint, f_loc, gt_density, rate=1):
    """
    根据关键点坐标生成可视化点图，并将坐标信息写入文件

    参数:
        kpoint (numpy.ndarray): 关键点坐标数组
        f_loc (file object): 输出文件句柄
        gt_density (torch.Tensor): 真实密度图（当前未完全使用）
        rate (float): 缩放比例，默认为1

    返回:
        numpy.ndarray: 生成的可视化点图
    """
    pred_coor = np.nonzero(kpoint)  # 获取预测的关键点坐标

    # 创建白色背景图像，尺寸根据kpoint和缩放率调整
    point_map = np.zeros((int(kpoint.shape[0] * rate), int(kpoint.shape[1] * rate), 3), dtype="uint8") + 255
    coord_list = []  # 存储坐标列表
    for i in range(0, len(pred_coor[0])):  # 遍历所有检测到的点
        h = int(pred_coor[0][i] * rate)  # 计算高度坐标并按比例缩放
        w = int(pred_coor[1][i] * rate)  # 计算宽度坐标并按比例缩放
        coord_list.append([w, h])  # 将坐标加入列表
        cv2.circle(point_map, (w, h), 2, (0, 0, 0), -1)  # 在图像上画黑色圆点标记位置

    for data in coord_list:  # 将所有坐标写入文件
        f_loc.write('{} {} '.format(math.floor(data[0]), math.floor(data[1])))  # 写入向下取整的坐标
    f_loc.write('\n')  # 写入换行符

    return point_map  # 返回生成的点图


def LMDS_gt_counting(input, w_fname, gt_loc):
    """
    对真实密度图执行局部最大值检测计数，用于生成训练标签

    参数:
        input (torch.Tensor): 输入的真实密度图张量
        w_fname (str): 当前处理的文件名
        gt_loc (file object): 输出真实位置信息的文件句柄

    返回:
        tuple: (计数值, 关键点坐标数组, 更新后的文件句柄)
    """
    input = input.unsqueeze(0)  # 在第0维增加一个维度，通常用于批次处理
    input_max = torch.max(input).item()  # 获取输入张量的最大值并转为Python标量

    ''' find local maxima'''  # 寻找局部最大值区域
    keep = F.max_pool2d(input, (3, 3), stride=1, padding=1)  # 3x3窗口最大池化，保留邻域最大值
    keep = (keep == input).float()  # 创建掩码：当前位置如果是邻域最大值则为1，否则为0
    input = keep * input  # 应用掩码，只保留局部最大值，其他位置设为0

    '''set the pixel valur of local maxima as 1 for counting'''  # 将局部最大值像素设为1进行计数
    input[input < 85 / 255.0 * input_max] = 0  # 阈值过滤：低于阈值的局部最大值设为0
    input[input > 0] = 1  # 将剩余的正数设为1，准备计数

    # 负样本处理
    if input_max < 0.06:  # 如果输入最大值小于阈值0.06
        input = input * 0  # 则认为是负样本，清零所有值

    count = int(torch.sum(input).item())  # 统计非零元素个数作为计数值

    kpoint = input.data.squeeze(0).squeeze(0).cpu().numpy()  # 移除额外维度并转移到CPU内存转为numpy数组

    gt_loc.write('{} {} '.format(w_fname, count))  # 将文件名和计数值写入输出文件
    return count, kpoint, gt_loc  # 返回计数值、关键点数组和文件句柄


def generate_gt_point_map(kpoint, gt_loc, gt_density, rate=1):
    """
    为真实数据生成可视化点图并写入位置信息

    参数:
        kpoint (numpy.ndarray): 关键点坐标数组
        gt_loc (file object): 输出真实位置信息的文件句柄
        gt_density (torch.Tensor): 真实密度图（当前未完全使用）
        rate (float): 缩放比例，默认为1

    返回:
        numpy.ndarray: 生成的可视化点图
    """
    pred_coor = np.nonzero(kpoint)  # 获取预测的关键点坐标

    # 创建白色背景图像，尺寸根据kpoint和缩放率调整
    point_map = np.zeros((int(kpoint.shape[0] * rate), int(kpoint.shape[1] * rate), 3), dtype="uint8") + 255
    coord_list = []  # 存储坐标列表
    for i in range(0, len(pred_coor[0])):  # 遍历所有检测到的点
        h = int(pred_coor[0][i] * rate)  # 计算高度坐标并按比例缩放
        w = int(pred_coor[1][i] * rate)  # 计算宽度坐标并按比例缩放
        coord_list.append([w, h])  # 将坐标加入列表
        cv2.circle(point_map, (w, h), 2, (0, 0, 0), -1)  # 在图像上画黑色圆点标记位置

    for data in coord_list:  # 将所有坐标写入文件
        gt_loc.write('{} {} '.format(math.floor(data[0]), math.floor(data[1])))  # 写入向下取整的坐标
    gt_loc.write('\n')  # 写入换行符

    return point_map  # 返回生成的点图


import torch
import torch.nn.functional as F
import numpy as np


def get_pred_points_from_density(density_map):
    """
    从预测密度图中提取预测点坐标（复用LMDS逻辑）
    新增：全流程调试日志，定位检测不到点的原因
    """
    # ===================== 第一步：基础校验 + 初始化调试日志 =====================
    debug_info = {
        "密度图尺寸": None,
        "原始密度图极值": None,
        "归一化后极值": None,
        "局部极大值数量": 0,
        "阈值过滤后数量": 0,
        "最终预测点数量": 0,
        "失败原因": None
    }

    # 空张量校验
    if density_map.numel() == 0:
        debug_info["失败原因"] = "密度图为空张量"
        print(f"\n【调试日志】{debug_info}")
        return np.array([])

    # 保存密度图尺寸
    H, W = density_map.shape[0], density_map.shape[1]
    debug_info["密度图尺寸"] = f"H={H}, W={W}"

    # ===================== 第二步：密度图归一化（修复重复定义问题） =====================
    # 原始密度图极值（修复：用原始density_map计算，而非重复定义的input）
    input_ori = density_map.unsqueeze(0)  # (1, H, W)
    ori_min = torch.min(input_ori).item()
    ori_max = torch.max(input_ori).item()
    debug_info["原始密度图极值"] = f"min={ori_min:.6f}, max={ori_max:.6f}"

    # 避免除零错误
    if ori_max - ori_min < 1e-8:
        debug_info["失败原因"] = "密度图全为相同值（无波动）"
        print(f"\n【调试日志】{debug_info}")
        return np.array([])

    # 归一化到[0,1]区间（核心步骤）
    input_norm = (input_ori - ori_min) / (ori_max - ori_min)
    norm_min = torch.min(input_norm).item()
    norm_max = torch.max(input_norm).item()
    debug_info["归一化后极值"] = f"min={norm_min:.6f}, max={norm_max:.6f}"

    # ===================== 第三步：局部极大值检测（3x3池化） =====================
    # 池化找局部极大值（修复：input_norm是(1,H,W)，需扩展为(1,1,H,W)适配池化）
    input_pool = input_norm.unsqueeze(0) # (1,1,H,W)
    keep = F.max_pool2d(input_pool, (7, 7), stride=1, padding=3)
    keep = (keep == input_pool).float()  # 仅保留局部极大值位置
    local_max_count = torch.sum(keep).item()
    debug_info["局部极大值数量"] = int(local_max_count)

    # 过滤后只保留局部极大值
    input_filtered = keep * input_pool

    # ===================== 第四步：阈值过滤（修复阈值逻辑） =====================
    # 修复：阈值应基于归一化后的最大值（1.0），而非原始input_max
    threshold = 50 / 255.0  # ≈0.235（归一化后固定阈值）
    # 原错误：input < 60/255 * input_max → 改为input < threshold
    input_filtered[input_filtered < threshold] = 0
    input_filtered[input_filtered > 0] = 1  # 二值化

    # 低置信度过滤（修复：用归一化后的最大值判断）
    if norm_max < 0.06:
        input_filtered = input_filtered * 0
        debug_info["失败原因"] = "归一化后最大值<0.06，强制置零"

    # 统计阈值过滤后数量
    threshold_count = torch.sum(input_filtered > 0).item()
    debug_info["阈值过滤后数量"] = int(threshold_count)

    # ===================== 第五步：提取坐标 + 最终日志 =====================
    # 提取预测点坐标
    pred_coor = np.nonzero(input_filtered.squeeze(0).squeeze(0).cpu().numpy())
    pred_points = np.stack([pred_coor[1], pred_coor[0]], axis=1) if len(pred_coor[0]) > 0 else np.array([])
    debug_info["最终预测点数量"] = len(pred_points)

    # 定位失败原因（如果无点）
    if len(pred_points) == 0 and not debug_info["失败原因"]:
        if local_max_count == 0:
            debug_info["失败原因"] = "3x3池化未检测到任何局部极大值"
        elif threshold_count == 0:
            debug_info["失败原因"] = f"阈值{threshold:.4f}过高，过滤掉所有点"
        else:
            debug_info["失败原因"] = "坐标提取异常（空坐标）"

    # 打印完整调试日志
    print(f"\n【get_pred_points_from_density 调试日志】")
    for k, v in debug_info.items():
        print(f"  {k}: {v}")

    return pred_points

def match_points(pred_points, gt_points, distance_thresh=5.0):
    """
    匹配预测点和真实点（基于距离阈值）

    参数:
        pred_points (np.ndarray): 预测点坐标 (N, 2) [x, y]
        gt_points (np.ndarray): 真实点坐标 (M, 2) [x, y]
        distance_thresh (float): 匹配距离阈值，单位为像素

    返回:
        tuple:
            tp (int): 真正例数量（预测正确的点）
            fp (int): 假正例数量（预测错误的点）
            fn (int): 假负例数量（漏检的点）
    """
    if len(pred_points) == 0 and len(gt_points) == 0:
        return 0, 0, 0
    if len(pred_points) == 0:
        return 0, 0, len(gt_points)
    if len(gt_points) == 0:
        return 0, len(pred_points), 0

    # 计算所有预测点和真实点的距离矩阵
    dist_matrix = distance.cdist(pred_points, gt_points, metric='euclidean')

    # 贪心匹配：为每个真实点匹配最近的预测点（且距离小于阈值）
    gt_matched = np.zeros(len(gt_points), dtype=bool)
    pred_matched = np.zeros(len(pred_points), dtype=bool)
    tp = 0

    # 按距离从小到大排序
    dist_flat = dist_matrix.flatten()
    indices = np.argsort(dist_flat)
    pred_indices = indices // len(gt_points)
    gt_indices = indices % len(gt_points)

    for p_idx, g_idx in zip(pred_indices, gt_indices):
        if dist_matrix[p_idx, g_idx] > distance_thresh:
            break  # 距离超过阈值，后续都不匹配
        if not gt_matched[g_idx] and not pred_matched[p_idx]:
            gt_matched[g_idx] = True
            pred_matched[p_idx] = True
            tp += 1

    fp = len(pred_points) - tp  # 预测点总数 - 正确匹配数 = 假正例
    fn = len(gt_points) - tp  # 真实点总数 - 正确匹配数 = 假负例

    return tp, fp, fn


def calculate_precision_recall_f1(tp, fp, fn):
    """
    计算精确率、召回率、F1分数

    参数:
        tp (int): 真正例
        fp (int): 假正例
        fn (int): 假负例

    返回:
        tuple: (precision, recall, f1)
    """
    if tp + fp == 0:
        precision = 0.0
    else:
        precision = tp / (tp + fp)

    if tp + fn == 0:
        recall = 0.0
    else:
        recall = tp / (tp + fn)

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return precision, recall, f1


def calculate_ap_ar(pred_points_list, gt_points_list, distance_thresh=5.0, num_recall_points=11):
    """
    计算AP（平均精度）和AR（平均召回率）
    采用VOC 2007的11点插值法

    参数:
        pred_points_list (list): 多个样本的预测点列表，每个元素是 (N,2) 的np.ndarray
        gt_points_list (list): 多个样本的真实点列表，每个元素是 (M,2) 的np.ndarray
        distance_thresh (float): 匹配距离阈值
        num_recall_points (int): 插值的召回率点数（默认11点：0,0.1,...,1.0）

    返回:
        tuple: (ap, ar)
    """
    # 收集所有样本的tp/fp/fn
    all_tp = []
    all_fp = []
    all_gt_count = []

    for pred_points, gt_points in zip(pred_points_list, gt_points_list):
        tp, fp, fn = match_points(pred_points, gt_points, distance_thresh)
        all_tp.append(tp)
        all_fp.append(fp)
        all_gt_count.append(len(gt_points))

    # 计算累计tp和fp（按置信度排序，这里所有预测点置信度相同，直接求和）
    total_tp = sum(all_tp)
    total_fp = sum(all_fp)
    total_gt = sum(all_gt_count)

    # 计算11点插值的AP
    recall_levels = np.linspace(0, 1.0, num_recall_points)
    precisions = []

    # 对于每个召回率阈值，计算对应的最大精确率
    for r_thresh in recall_levels:
        # 找出召回率 >= r_thresh 的所有情况的最大精确率
        # 由于这里是单阈值匹配，直接计算整体精度
        if total_tp + total_fp == 0:
            prec = 0.0
        else:
            prec = total_tp / (total_tp + total_fp)

        if total_tp + total_gt == 0:
            recall = 0.0
        else:
            recall = total_tp / total_gt

        if recall >= r_thresh:
            precisions.append(prec)
        else:
            precisions.append(0.0)

    ap = np.mean(precisions)

    # 计算AR（平均召回率）
    if total_tp + total_gt == 0:
        ar = 0.0
    else:
        ar = total_tp / (total_tp + fn) if (total_tp + fn) > 0 else 0.0

    return ap, ar


def evaluate_detection_metrics(pred_density_map, gt_points, distance_thresh=10.0):
    """
    主函数：输入预测密度图和GT点信息，计算F1、AP、AR

    参数:
        pred_density_map (torch.Tensor): 预测的密度图 (H, W)
        gt_points (torch.Tensor/np.ndarray): 真实点坐标 (M, 2) [x, y]
        distance_thresh (float): 匹配距离阈值

    返回:
        tuple: (f1, ap, ar, precision, recall)
    """
    # ==================== 新增：可视化GT点 ====================
    gt_vis_path = f"/ZHANGyong/wjj/SAM-Free/tmp/gt_points.png"
    visualize_points_on_density(
        pred_density_map,
        gt_points,
        gt_vis_path,
        title=f"GT Points (Count: {len(gt_points)})",
        color=(0, 0, 255),  # GT点用红色
        radius=2
    )

    # 1. 从密度图提取预测点
    pred_points = get_pred_points_from_density(pred_density_map)

    # ==================== 新增：可视化预测点 ====================
    pred_vis_path = f"/ZHANGyong/wjj/SAM-Free/tmp/pred_points.png"
    visualize_points_on_density(
        pred_density_map,
        pred_points,
        pred_vis_path,
        title=f"Pred Points (Count: {len(pred_points)})",
        color=(0, 255, 0),  # 预测点用绿色
        radius=2
    )

    # 2. 转换GT点格式为numpy数组
    if isinstance(gt_points, torch.Tensor):
        gt_points = gt_points.cpu().numpy()

    # 3. 单样本匹配（计算AP/AR需要多个样本，这里先返回单样本指标，批量调用需外层封装）
    tp, fp, fn = match_points(pred_points, gt_points, distance_thresh)
    precision, recall, f1 = calculate_precision_recall_f1(tp, fp, fn)



    return f1, precision, recall


def visualize_points_on_density(density_map, points, save_path, title, color=(0, 255, 0), radius=2):
    """
    将点绘制在密度图上并保存

    参数:
        density_map (torch.Tensor): 预测密度图 (H, W)
        points (np.ndarray): 点坐标数组 (N, 2) [x, y]
        save_path (str): 保存路径
        title (str): 图像标题
        color (tuple): 绘制点的颜色 (B, G, R)
        radius (int): 绘制点的半径
    """
    # 创建保存目录
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # 将密度图转换为可显示的图像格式
    if isinstance(density_map, torch.Tensor):
        density_np = density_map.cpu().numpy()
    else:
        density_np = density_map

    # 归一化到0-255范围用于显示
    density_normalized = (density_np - density_np.min()) / (density_np.max() - density_np.min() + 1e-8)
    density_normalized = (density_normalized * 255).astype(np.uint8)

    # 转换为彩色图像
    density_rgb = cv2.cvtColor(density_normalized, cv2.COLOR_GRAY2BGR)

    # 绘制点
    if points is not None and len(points) > 0:
        for (x, y) in points:
            # ========== 核心修复：处理Tensor类型的坐标 ==========
            # 如果是Tensor，先转标量；否则直接用原始值
            if isinstance(x, torch.Tensor):
                x_scalar = x.item()
            else:
                x_scalar = x

            if isinstance(y, torch.Tensor):
                y_scalar = y.item()
            else:
                y_scalar = y

            # 取整并转换为整数
            x_int = int(round(x_scalar))
            y_int = int(round(y_scalar))
            # ==================================================

            # 确保坐标在图像范围内
            if 0 <= x_int < density_rgb.shape[1] and 0 <= y_int < density_rgb.shape[0]:
                cv2.circle(density_rgb, (x_int, y_int), radius, color, -1)

    # 添加标题文本
    cv2.putText(density_rgb, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                1, (255, 255, 255), 2, cv2.LINE_AA)

    # 保存图像
    cv2.imwrite(save_path, density_rgb)
    print(f"可视化图像已保存至: {save_path}")