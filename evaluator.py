"""
统一的评估器模块
支持计数和定位指标评估
"""
import os
import sys
import json
import math
import time
import torch
import numpy as np
from contextlib import redirect_stdout
from datetime import datetime
from typing import Dict, Any, List, Tuple
from tqdm import tqdm
from PIL import Image
from scipy.ndimage import gaussian_filter
from torchvision.transforms import transforms as T

from shi_segment_anything import sam_model_registry
from shi_segment_anything.automatic_mask_generator import SamAutomaticMaskGenerator
from myutil.localization import evaluate_detection_metrics, visualize_masks, get_pred_points_from_density
from dataset_manager import DatasetManager


class CountingEvaluator:
    """统一的计数评估器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = config.get('device', 'cuda:0')
        self.sam_model = None
        self.mask_generator = None
        self.bmnet_model = None
        self.bmnet_transform = None
        self.dataset_manager = None
        
        self._init_models()
    
    def _init_models(self):
        """初始化模型"""
        # 初始化SAM模型
        sam_checkpoint = self.config['model']['sam_checkpoint']
        model_type = self.config['model']['model_type']
        self.sam_model = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        self.sam_model.to(device=self.device)
        
        # 初始化SAM mask生成器
        sam_params = self.config['sam_params']
        self.mask_generator = SamAutomaticMaskGenerator(
            model=self.sam_model,
            points_per_side=sam_params['points_per_side'],
            points_per_batch=sam_params['points_per_batch'],
            pred_iou_thresh=sam_params['pred_iou_thresh'],
            stability_score_thresh=sam_params['stability_score_thresh'],
            use_ref_sim_for_filter=sam_params['use_ref_sim_filter'],
            ref_sim_thresh=sam_params['ref_sim_thresh'],
        )
        
        # 初始化数据集管理器
        self.dataset_manager = DatasetManager(self.config)
        
        # 可选：初始化BMNet模型
        if self.config['bmnet']['use_bmnet']:
            self._init_bmnet()
    
    def _init_bmnet(self):
        """初始化BMNet模型"""
        try:
            from pathlib import Path
            script_dir = Path(__file__).parent
            wjj_root = script_dir.parent
            bmnet_path = wjj_root / "BMNet"
            
            if str(bmnet_path) not in sys.path:
                sys.path.insert(0, str(bmnet_path))
            
            from config import cfg as bmnet_cfg
            cfg_path = self.config['bmnet']['bmnet_cfg']
            
            if not os.path.exists(cfg_path):
                raise FileNotFoundError(f"BMNet 配置不存在: {cfg_path}")
            
            bmnet_cfg.merge_from_file(cfg_path)
            
            # 处理路径
            if not os.path.isabs(bmnet_cfg.DIR.snapshot):
                bmnet_cfg.DIR.snapshot = str(bmnet_path / bmnet_cfg.DIR.snapshot)
            bmnet_cfg.DIR.output_dir = os.path.join(bmnet_cfg.DIR.snapshot, bmnet_cfg.DIR.exp)
            bmnet_cfg.VAL.resume = os.path.join(bmnet_cfg.DIR.output_dir, bmnet_cfg.VAL.resume)
            
            ckpt_path = self.config['bmnet']['bmnet_checkpoint'] or bmnet_cfg.VAL.resume
            if not os.path.exists(ckpt_path):
                raise FileNotFoundError(f"BMNet 权重不存在: {ckpt_path}")
            
            from models import build_model as build_bmnet
            self.bmnet_model = build_bmnet(bmnet_cfg)
            ckpt = torch.load(ckpt_path, map_location="cpu")
            self.bmnet_model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
            
            bmnet_device = self.config['bmnet']['bmnet_device'] or self.device
            self.bmnet_model.to(bmnet_device)
            self.bmnet_model.eval()
            
            self.bmnet_transform = T.Compose([
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
            
            print(f"  BMNet 已加载: {ckpt_path}")
            
        except Exception as e:
            print(f"  BMNet 加载失败，将使用网格候选点: {e}")
            self.config['bmnet']['use_bmnet'] = False
            self.bmnet_model = None
    
    def run_evaluation(self, split: str = "test"):
        """运行评估"""
        print("=" * 60)
        print("开始评估")
        print("=" * 60)
        
        # 数据集管理器已在_init_models中初始化
        
        # 获取图像ID列表
        image_ids = self.dataset_manager.get_image_ids(split)
        # 可选：限制评估图片数量（用于冒烟测试/小样本调试）
        max_images = int(self.config.get("evaluation", {}).get("max_images", 0) or 0)
        if max_images > 0:
            image_ids = image_ids[:max_images]
        print(f"数据集: {self.config['dataset']['name']}")
        print(f"图像数量: {len(image_ids)}")
        
        # 设置输出目录
        output_dirs = self._setup_output_dirs(split)
        
        # 打开日志文件
        log_file = self._setup_logging(output_dirs['log_path'])
        # 打开debug文件（捕获详细stdout）
        debug_file = open(output_dirs['debug_log_path'], "w", encoding="utf-8")
        
        # 初始化指标
        metrics = self._init_metrics()
        
        # 随机数生成器
        rng = np.random.default_rng(self.config['training']['seed'])

        # 每多少张打印一次摘要（默认每张）
        print_every = int(self.config.get("output", {}).get("print_every", 1) or 1)
        
        # 处理每张图像
        dataset_name = self.config['dataset']['name']
        for i, image_id in tqdm(enumerate(image_ids), total=len(image_ids), desc=f"处理{dataset_name}图片"):
            try:
                t0 = time.time()

                # 捕获详细stdout到 debug 文件，终端只保留摘要
                with redirect_stdout(debug_file):
                    # 获取样本数据
                    sample_data = self.dataset_manager.get_sample_data(image_id)
                    print(f"[EVALUATOR DEBUG] 样本 {image_id}: GT计数={sample_data['gt_count']}, GT点数={len(sample_data['gt_points'])}")
                    
                    # 生成prompts
                    ref_prompt, prompt_k = self.dataset_manager.generate_prompts(sample_data, rng)
                    
                    # 可选：使用BMNet生成候选点
                    external_point_coords = None
                    if self.config['bmnet']['use_bmnet'] and self.bmnet_model is not None:
                        external_point_coords = self._get_bmnet_candidates(sample_data, ref_prompt)
                    
                    # 生成masks
                    if len(ref_prompt) == 0:
                        masks = []
                    else:
                        masks = self.mask_generator.generate(
                            sample_data['image'], ref_prompt,
                            point_coords=external_point_coords,
                            verbose_point_stats=self.config['sam_params']['verbose_point_stats'],
                        )
                    
                    # 计算指标
                    sample_metrics = self._compute_sample_metrics(sample_data, masks)
                    sample_metrics['prompt_k'] = prompt_k
                    
                    # 可视化（可选）
                    if self.config['output']['save_visualization']:
                        self._save_visualization(sample_data, masks, output_dirs['vis_dir'])

                # 累加全局指标
                self._accumulate_metrics(metrics, sample_metrics)
                
                # 写入日志（CSV）
                self._write_sample_log(log_file, image_id, sample_metrics)

                # 摘要行（终端）：每 print_every 张打印一次
                if ((i + 1) % print_every == 0) or (i == 0) or (i + 1 == len(image_ids)):
                    dt = time.time() - t0
                    tqdm.write(
                        f"[{i+1}/{len(image_ids)}] {image_id} | "
                        f"pred={sample_metrics['pred_count']} gt={sample_metrics['gt_count']} "
                        f"abs_err={sample_metrics['count_abs_err']} "
                        f"F1={sample_metrics['f1']:.3f} P={sample_metrics['precision']:.3f} R={sample_metrics['recall']:.3f} "
                        f"time={dt:.2f}s"
                    )
                
            except Exception as e:
                tqdm.write(f"处理图片 {image_id} 时出错: {e}")
                continue
        
        # 计算最终指标
        final_metrics = self._compute_final_metrics(metrics, len(image_ids))
        
        # 打印和保存结果
        self._print_and_save_results(final_metrics, log_file, split)
        
        log_file.close()
        debug_file.close()
        print("评估完成！")
        
        return final_metrics
    
    def _setup_output_dirs(self, split: str) -> Dict[str, str]:
        """设置输出目录"""
        base_dir = self.config['output']['base_dir']
        dataset_name = self.config['dataset']['name']
        
        # 根据数据集类型调整输出目录
        if dataset_name == "Sheep_OBB":
            final_output_dir = os.path.join(base_dir, "Sheep")
            split_tag = self.config['output']['split_tag']
        elif dataset_name == "CARPK":
            final_output_dir = os.path.join(base_dir, "CARPK")
            split_tag = self.config['output']['split_tag']
        elif dataset_name == "CoNIC":
            final_output_dir = os.path.join(base_dir, "CoNIC")
            split_tag = self.config['output']['split_tag']
        elif dataset_name == "ShanghaiTech":
            final_output_dir = os.path.join(base_dir, "ShanghaiTech")
            split_tag = self.config['output']['split_tag']
        else:  # FSC147
            final_output_dir = os.path.join(base_dir, "FSC147")
            split_tag = split
        
        os.makedirs(final_output_dir, exist_ok=True)
        os.makedirs(os.path.join(final_output_dir, "logs"), exist_ok=True)
        
        prompt_tag = self.config['training']['prompt_type']
        if self.config['bmnet']['use_bmnet']:
            prompt_tag += "_bmnet"
        
        os.makedirs(os.path.join(final_output_dir, split_tag), exist_ok=True)
        os.makedirs(os.path.join(final_output_dir, split_tag, prompt_tag), exist_ok=True)
        
        # 可视化目录
        vis_dir = None
        if self.config['output']['save_visualization']:
            now = datetime.now()
            vis_suffix = f"vis_{now.month:02d}_{now.day:02d}_{now.hour:02d}_{now.minute:02d}"
            vis_dir = os.path.join(final_output_dir, split_tag, prompt_tag, vis_suffix)
            os.makedirs(vis_dir, exist_ok=True)
        
        log_path = os.path.join(final_output_dir, "logs", f"log-{split_tag}-{prompt_tag}.csv")
        debug_log_path = os.path.join(final_output_dir, "logs", f"debug-{split_tag}-{prompt_tag}.log")
        
        return {
            'final_output_dir': final_output_dir,
            'vis_dir': vis_dir,
            'log_path': log_path,
            'debug_log_path': debug_log_path,
            'split_tag': split_tag,
            'prompt_tag': prompt_tag
        }
    
    def _setup_logging(self, log_path: str):
        """设置日志文件"""
        log_file = open(log_path, "w", encoding="utf-8")
        log_file.write("image_id,pred_count,gt_count,count_abs_err,count_rel_err,f1,precision,recall,prompt_k\n")
        log_file.flush()
        return log_file
    
    def _init_metrics(self) -> Dict[str, float]:
        """初始化指标"""
        return {
            'MAE': 0.0,
            'RMSE': 0.0,
            'NAE': 0.0,
            'SRE': 0.0,
            'total_f1': 0.0,
            'total_precision': 0.0,
            'total_recall': 0.0,
        }
    
    def _get_bmnet_candidates(self, sample_data: Dict[str, Any], ref_prompt: List) -> np.ndarray:
        """使用BMNet生成候选点"""
        try:
            bmnet_device = self.config['bmnet']['bmnet_device'] or self.device
            
            if self.config['dataset']['name'] == "Sheep_OBB":
                # Sheep数据集：使用预计算的patches
                bmnet_data = self.dataset_manager.get_bmnet_data(sample_data['image_id'])
                if bmnet_data is None:
                    return None
                
                pt = bmnet_data['patches']
                sc = bmnet_data['scales']
                
                if pt.numel() == 0:
                    pt = torch.zeros(0, 3, 128, 128)
                    sc = torch.zeros(0, dtype=torch.long)
                
                img_t = self.bmnet_transform(Image.fromarray(sample_data['image'])).unsqueeze(0).to(bmnet_device)
                patches_dict = {
                    "patches": pt.unsqueeze(0).to(bmnet_device),
                    "scale_embedding": sc.unsqueeze(0).to(bmnet_device),
                }
                
                with torch.no_grad():
                    out = self.bmnet_model(img_t, patches_dict, is_train=False)
                density_map = out.squeeze(0).squeeze(0)
                
            else:  # FSC147
                # FSC147数据集：从示例框生成patches
                if len(ref_prompt) > 0:
                    patches_list = []
                    scales_list = []
                    
                    for box in ref_prompt:
                        x1, y1, x2, y2 = map(int, box)
                        h, w = sample_data['image'].shape[:2]
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(w, x2), min(h, y2)
                        
                        if x2 > x1 and y2 > y1:
                            patch = sample_data['image'][y1:y2, x1:x2]
                            patch_pil = Image.fromarray(patch).resize((128, 128))
                            patch_tensor = self.bmnet_transform(patch_pil)
                            patches_list.append(patch_tensor)
                            
                            patch_area = (x2 - x1) * (y2 - y1)
                            scale = min(3, max(0, int(np.log2(patch_area / 1000))))
                            scales_list.append(scale)
                    
                    if patches_list:
                        pt = torch.stack(patches_list)
                        sc = torch.tensor(scales_list, dtype=torch.long)
                    else:
                        pt = torch.zeros(0, 3, 128, 128)
                        sc = torch.zeros(0, dtype=torch.long)
                    
                    img_t = self.bmnet_transform(Image.fromarray(sample_data['image'])).unsqueeze(0).to(bmnet_device)
                    patches_dict = {
                        "patches": pt.unsqueeze(0).to(bmnet_device),
                        "scale_embedding": sc.unsqueeze(0).to(bmnet_device),
                    }
                    
                    with torch.no_grad():
                        out = self.bmnet_model(img_t, patches_dict, is_train=False)
                    density_map = out.squeeze(0).squeeze(0)
                else:
                    density_map = None
            
            # 上采样密度图到原图尺寸
            if density_map is not None:
                h_orig, w_orig = sample_data['image'].shape[0], sample_data['image'].shape[1]
                if density_map.shape[0] != h_orig or density_map.shape[1] != w_orig:
                    density_map = torch.nn.functional.interpolate(
                        density_map.unsqueeze(0).unsqueeze(0),
                        size=(h_orig, w_orig),
                        mode="bilinear",
                    ).squeeze(0).squeeze(0)
                
                external_point_coords = get_pred_points_from_density(density_map)
                if isinstance(external_point_coords, np.ndarray) and len(external_point_coords) == 0:
                    external_point_coords = None
                return external_point_coords
            
        except Exception as e:
            print(f"BMNet 处理失败: {e}")
        
        return None
    
    def _compute_sample_metrics(self, sample_data: Dict[str, Any], masks: List) -> Dict[str, float]:
        """计算单个样本的指标"""
        gt_cnt = sample_data['gt_count']
        pred_cnt = len(masks)
        
        # 计数指标
        err = abs(gt_cnt - pred_cnt)
        rel_err = (err / gt_cnt) if gt_cnt > 0 else 0.0
        
        # 定位指标：提取预测点
        pred_points = []
        for m in masks:
            pc = m.get("point_coords", None)
            if pc is None or len(pc) == 0:
                continue
            # 兼容不同格式
            if isinstance(pc[0], (list, tuple, np.ndarray)):
                pred_points.append(pc[0])
            else:
                pred_points.append(pc)
        
        pred_points = np.asarray(pred_points, dtype=np.float32) if len(pred_points) > 0 else np.array([], dtype=np.float32).reshape(0, 2)
        
        # 生成密度图
        pred_density_map = self._points_to_density_map(pred_points, sample_data['image'])
        
        # 计算定位指标
        f1, precision, recall = evaluate_detection_metrics(
            pred_density_map=pred_density_map,
            gt_points=sample_data['gt_points'],
            distance_thresh=self.config['evaluation']['distance_thresh']
        )
        
        return {
            'pred_count': pred_cnt,
            'gt_count': gt_cnt,
            'count_abs_err': err,
            'count_rel_err': rel_err,
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'masks': masks,
            'pred_density_map': pred_density_map
        }
    
    def _points_to_density_map(self, pred_points, image, sigma=None):
        """将预测点转换为密度图"""
        if sigma is None:
            sigma = self.config['evaluation']['vis_sigma']
        
        H, W = image.shape[0], image.shape[1]
        density_map = np.zeros((H, W), dtype=np.float32)
        
        for (x, y) in pred_points:
            x = np.clip(x, 0, W - 1)
            y = np.clip(y, 0, H - 1)
            x_int, y_int = int(np.round(x)), int(np.round(y))
            density_map[y_int, x_int] += 1
        
        density_map = gaussian_filter(density_map, sigma=sigma)
        pred_density_map = torch.from_numpy(density_map)
        
        return pred_density_map
    
    def _accumulate_metrics(self, metrics: Dict[str, float], sample_metrics: Dict[str, float]):
        """累加指标"""
        metrics['MAE'] += sample_metrics['count_abs_err']
        metrics['RMSE'] += sample_metrics['count_abs_err'] ** 2
        metrics['NAE'] += sample_metrics['count_rel_err']
        metrics['SRE'] += (sample_metrics['count_abs_err'] ** 2) / sample_metrics['gt_count'] if sample_metrics['gt_count'] > 0 else 0
        metrics['total_f1'] += sample_metrics['f1']
        metrics['total_precision'] += sample_metrics['precision']
        metrics['total_recall'] += sample_metrics['recall']
    
    def _save_visualization(self, sample_data: Dict[str, Any], masks: List, vis_dir: str):
        """保存可视化结果"""
        if vis_dir is None:
            return
        
        image_id = sample_data['image_id']
        fname_base = os.path.splitext(os.path.basename(str(image_id)))[0]
        fname_safe = fname_base.replace("/", "_").replace("\\", "_")
        
        # 保存masks可视化
        mask_vis_path = os.path.join(vis_dir, f"{fname_safe}_masks.png")
        visualize_masks(sample_data['image'], masks, mask_vis_path, title="Masks")
        
        # 保存GT点和预测点的可视化
        # 获取预测点
        pred_points = []
        for m in masks:
            pc = m.get("point_coords", None)
            if pc is None or len(pc) == 0:
                continue
            # 兼容不同格式
            if isinstance(pc[0], (list, tuple, np.ndarray)):
                pred_points.append(pc[0])
            else:
                pred_points.append(pc)
        
        pred_points = np.asarray(pred_points, dtype=np.float32) if len(pred_points) > 0 else np.array([], dtype=np.float32).reshape(0, 2)
        
        # 生成密度图
        pred_density_map = self._points_to_density_map(pred_points, sample_data['image'])
        
        # 保存GT和预测点的可视化
        from myutil.localization import evaluate_detection_metrics
        evaluate_detection_metrics(
            pred_density_map=pred_density_map,
            gt_points=sample_data['gt_points'],
            distance_thresh=self.config['evaluation']['distance_thresh'],
            vis_dir=vis_dir,
            image_id=image_id
        )
    
    def _write_sample_log(self, log_file, image_id: str, sample_metrics: Dict[str, float]):
        """写入单个样本的日志"""
        log_file.write(
            f"{image_id},{sample_metrics['pred_count']},{sample_metrics['gt_count']},"
            f"{sample_metrics['count_abs_err']},{sample_metrics['count_rel_err']:.6f},"
            f"{sample_metrics['f1']:.6f},{sample_metrics['precision']:.6f},"
            f"{sample_metrics['recall']:.6f},{sample_metrics['prompt_k']}\n"
        )
        log_file.flush()
    
    def _compute_final_metrics(self, metrics: Dict[str, float], num_images: int) -> Dict[str, float]:
        """计算最终指标"""
        n = max(1, num_images)
        return {
            'avg_MAE': metrics['MAE'] / n,
            'avg_RMSE': math.sqrt(metrics['RMSE'] / n),
            'avg_NAE': metrics['NAE'] / n,
            'avg_SRE': math.sqrt(metrics['SRE'] / n) if n > 0 else 0.0,
            'avg_f1': metrics['total_f1'] / n,
            'avg_precision': metrics['total_precision'] / n,
            'avg_recall': metrics['total_recall'] / n,
            'num_images': num_images
        }
    
    def _print_and_save_results(self, final_metrics: Dict[str, float], log_file, split: str):
        """打印和保存最终结果"""
        prompt_tag = self.config['training']['prompt_type']
        if self.config['bmnet']['use_bmnet']:
            prompt_tag += "_bmnet"
        
        summary = (
            f"\n[{split} | {prompt_tag}] 最终结果:\n"
            f"计数指标: MAE={final_metrics['avg_MAE']:.4f}, RMSE={final_metrics['avg_RMSE']:.4f}, "
            f"NAE={final_metrics['avg_NAE']:.4f}, SRE={final_metrics['avg_SRE']:.4f}\n"
            f"定位指标: F1={final_metrics['avg_f1']:.4f}, P={final_metrics['avg_precision']:.4f}, "
            f"R={final_metrics['avg_recall']:.4f}\n"
            f"处理图片数: {final_metrics['num_images']}\n"
        )
        
        print(summary)
        log_file.write(summary)