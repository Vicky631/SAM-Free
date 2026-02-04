#!/usr/bin/env python3
"""
统一的主文件 - 支持多数据集的计数和定位评估
使用YAML配置文件管理不同数据集的参数

使用方法:
python main.py --config configs/sheep.yaml --split test --device cuda:0
python main.py --config configs/fsc147.yaml --split test --device cuda:1
"""

import os
import sys
import yaml
import argparse
from pathlib import Path

# 添加当前目录到Python路径
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from evaluator import CountingEvaluator


def load_config(config_path: str) -> dict:
    """加载YAML配置文件"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="多数据集计数和定位评估工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 评估Sheep数据集
  python main.py --config configs/sheep.yaml --split test --device cuda:0
  
  # 评估FSC147数据集
  python main.py --config configs/fsc147.yaml --split test --device cuda:1
  
  # 使用不同的prompt类型
  python main.py --config configs/sheep.yaml --split test --prompt_type point
  
  # 启用可视化
  python main.py --config configs/fsc147.yaml --split test --save_vis
        """
    )
    
    # 必需参数
    parser.add_argument(
        '--config', '-c', 
        type=str, 
        required=True,
        help='配置文件路径 (例如: configs/sheep.yaml)'
    )
    
    # 可选参数 - 会覆盖配置文件中的设置
    parser.add_argument(
        '--split', '-s',
        type=str,
        default=None,
        choices=['train', 'test', 'val'],
        help='数据集划分 (默认使用配置文件中的设置)'
    )
    
    parser.add_argument(
        '--device', '-d',
        type=str,
        default=None,
        help='设备 (例如: cuda:0, cuda:1, cpu)'
    )
    
    parser.add_argument(
        '--prompt_type',
        type=str,
        default=None,
        choices=['box', 'point'],
        help='Prompt类型 (覆盖配置文件设置)'
    )
    
    parser.add_argument(
        '--prompt_box_num',
        type=int,
        default=None,
        help='Prompt框/点的数量 (覆盖配置文件设置)'
    )
    
    parser.add_argument(
        '--use_bmnet',
        action='store_true',
        help='启用BMNet生成候选点 (覆盖配置文件设置)'
    )
    
    parser.add_argument(
        '--no_bmnet',
        action='store_true',
        help='禁用BMNet生成候选点 (覆盖配置文件设置)'
    )
    
    parser.add_argument(
        '--save_vis',
        action='store_true',
        help='保存可视化结果 (覆盖配置文件设置)'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='输出目录 (覆盖配置文件设置)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='随机种子 (覆盖配置文件设置)'
    )
    
    # 调试参数
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='详细输出模式'
    )
    
    parser.add_argument(
        '--verbose_point_stats',
        action='store_true',
        help='显示候选点各阶段的详细统计信息'
    )
    
    parser.add_argument(
        '--use_ref_sim_filter',
        action='store_true',
        help='使用与参考的相似度进行质量过滤（替代IoU过滤）'
    )
    
    # SAM参数覆盖
    parser.add_argument(
        '--distance_thresh',
        type=float,
        default=None,
        help='定位匹配距离阈值（像素）'
    )
    
    parser.add_argument(
        '--vis_sigma',
        type=float,
        default=None,
        help='密度图高斯滤波的sigma值'
    )
    
    parser.add_argument(
        '--points_per_side',
        type=int,
        default=None,
        help='网格每边点数基准，越大越密'
    )
    
    parser.add_argument(
        '--points_per_batch',
        type=int,
        default=None,
        help='SAM每次处理的候选点数'
    )
    
    parser.add_argument(
        '--pred_iou_thresh',
        type=float,
        default=None,
        help='SAM mask质量阈值（BMNet时建议0.7）'
    )
    
    parser.add_argument(
        '--stability_score_thresh',
        type=float,
        default=None,
        help='SAM稳定性阈值（BMNet时建议0.7）'
    )
    
    parser.add_argument(
        '--ref_sim_thresh',
        type=float,
        default=None,
        help='与参考的余弦相似度阈值'
    )
    
    # BMNet参数覆盖
    parser.add_argument(
        '--bmnet_cfg',
        type=str,
        default=None,
        help='BMNet配置文件路径'
    )
    
    parser.add_argument(
        '--bmnet_checkpoint',
        type=str,
        default=None,
        help='BMNet权重文件路径'
    )
    
    parser.add_argument(
        '--bmnet_device',
        type=str,
        default=None,
        help='BMNet运行设备'
    )
    
    # 训练参数覆盖
    parser.add_argument(
        '--prompt_select',
        type=str,
        default=None,
        choices=['head', 'random'],
        help='示例框选择策略'
    )
    
    parser.add_argument(
        '--dry_run',
        action='store_true',
        help='干运行模式 - 只打印配置不实际运行'
    )
    
    return parser.parse_args()


def override_config_with_args(config: dict, args) -> dict:
    """用命令行参数覆盖配置文件中的设置"""
    # 设备设置
    if args.device is not None:
        config['device'] = args.device
    elif 'device' not in config:
        config['device'] = 'cuda:0'
    
    # 训练参数覆盖
    if args.prompt_type is not None:
        config['training']['prompt_type'] = args.prompt_type
    
    if args.prompt_box_num is not None:
        config['training']['prompt_box_num'] = args.prompt_box_num
    
    if args.prompt_select is not None:
        config['training']['prompt_select'] = args.prompt_select
    
    if args.seed is not None:
        config['training']['seed'] = args.seed
    
    # BMNet设置覆盖
    if args.use_bmnet:
        config['bmnet']['use_bmnet'] = True
    elif args.no_bmnet:
        config['bmnet']['use_bmnet'] = False
    
    # 输出设置覆盖
    if args.save_vis:
        config['output']['save_visualization'] = True
    
    if args.output_dir is not None:
        config['output']['base_dir'] = args.output_dir
    
    # SAM参数覆盖
    if args.distance_thresh is not None:
        config['evaluation']['distance_thresh'] = args.distance_thresh
    
    if args.vis_sigma is not None:
        config['evaluation']['vis_sigma'] = args.vis_sigma
    
    if args.points_per_side is not None:
        config['sam_params']['points_per_side'] = args.points_per_side
    
    if args.points_per_batch is not None:
        config['sam_params']['points_per_batch'] = args.points_per_batch
    
    if args.pred_iou_thresh is not None:
        config['sam_params']['pred_iou_thresh'] = args.pred_iou_thresh
    
    if args.stability_score_thresh is not None:
        config['sam_params']['stability_score_thresh'] = args.stability_score_thresh
    
    if args.ref_sim_thresh is not None:
        config['sam_params']['ref_sim_thresh'] = args.ref_sim_thresh
    
    # BMNet参数覆盖
    if args.bmnet_cfg is not None:
        config['bmnet']['bmnet_cfg'] = args.bmnet_cfg
    
    if args.bmnet_checkpoint is not None:
        config['bmnet']['bmnet_checkpoint'] = args.bmnet_checkpoint
    
    if args.bmnet_device is not None:
        config['bmnet']['bmnet_device'] = args.bmnet_device
    
    # 调试参数覆盖
    if args.verbose_point_stats:
        config['sam_params']['verbose_point_stats'] = True
    
    if args.use_ref_sim_filter:
        config['sam_params']['use_ref_sim_filter'] = True
    
    # Split设置
    if args.split is not None:
        if config['dataset']['name'] == "FSC147":
            config['split'] = args.split
        else:  # Sheep等其他数据集
            config['output']['split_tag'] = args.split
    
    return config


def print_config_summary(config: dict, args):
    """打印配置摘要"""
    print("=" * 80)
    print("配置摘要 (Configuration Summary)")
    print("=" * 80)
    
    # 基本信息
    print(f"配置文件: {args.config}")
    print(f"数据集: {config['dataset']['name']}")
    print(f"设备: {config.get('device', 'cuda:0')}")
    
    # 数据集信息
    dataset_info = []
    if config['dataset']['name'] == "Sheep_OBB":
        dataset_info.extend([
            f"图像目录: {config['dataset']['image_dir']}",
            f"标注目录: {config['dataset']['annotation_dir']}",
            f"Split标签: {config['output']['split_tag']}"
        ])
    elif config['dataset']['name'] == "FSC147":
        dataset_info.extend([
            f"标注文件: {config['dataset']['annotation_file']}",
            f"图像目录: {config['dataset']['image_root']}",
            f"测试集: {config.get('split', 'test')}"
        ])
    elif config["dataset"]["name"] == "ShanghaiTech":
        dataset_info.extend([
            f"数据目录: {config['dataset']['data_dir']}",
            f"Part: {config['dataset'].get('part', 'A')}",
            f"Split标签: {config['output']['split_tag']}"
        ])
    
    for info in dataset_info:
        print(f"  {info}")
    
    # 模型信息
    print(f"SAM模型: {config['model']['model_type']}")
    print(f"SAM权重: {config['model']['sam_checkpoint']}")
    
    # 训练参数
    print(f"Prompt类型: {config['training']['prompt_type']}")
    print(f"Prompt数量: {config['training']['prompt_box_num']}")
    print(f"随机种子: {config['training']['seed']}")
    
    # BMNet设置
    bmnet_status = "启用" if config['bmnet']['use_bmnet'] else "禁用"
    print(f"BMNet: {bmnet_status}")
    
    # 输出设置
    print(f"输出目录: {config['output']['base_dir']}")
    vis_status = "启用" if config['output']['save_visualization'] else "禁用"
    print(f"保存可视化: {vis_status}")
    
    print("=" * 80)


def validate_config(config: dict):
    """验证配置文件的有效性"""
    errors = []
    
    # 检查必需的配置项
    required_sections = ['dataset', 'model', 'training', 'evaluation', 'output']
    for section in required_sections:
        if section not in config:
            errors.append(f"缺少必需的配置节: {section}")
    
    # 检查数据集配置
    if 'dataset' in config:
        dataset_config = config['dataset']
        if 'name' not in dataset_config:
            errors.append("数据集配置中缺少 'name' 字段")
        
        if dataset_config.get('name') == "Sheep_OBB":
            required_fields = ['image_dir', 'annotation_dir']
            for field in required_fields:
                if field not in dataset_config:
                    errors.append(f"Sheep数据集配置中缺少 '{field}' 字段")
        
        elif dataset_config.get('name') == "FSC147":
            required_fields = ['annotation_file', 'image_root', 'split_file']
            for field in required_fields:
                if field not in dataset_config:
                    errors.append(f"FSC147数据集配置中缺少 '{field}' 字段")
        elif dataset_config.get("name") == "ShanghaiTech":
            required_fields = ["data_dir"]
            for field in required_fields:
                if field not in dataset_config:
                    errors.append(f"ShanghaiTech数据集配置中缺少 '{field}' 字段")
            if "part" in dataset_config and str(dataset_config["part"]).upper() not in ["A", "B"]:
                errors.append(f"ShanghaiTech数据集配置 part 必须是 A/B，当前: {dataset_config['part']}")
            if "data_dir" in dataset_config and not os.path.exists(dataset_config["data_dir"]):
                errors.append(f"ShanghaiTech数据目录不存在: {dataset_config['data_dir']}")
    
    # 检查模型配置
    if 'model' in config:
        model_config = config['model']
        required_fields = ['sam_checkpoint', 'model_type']
        for field in required_fields:
            if field not in model_config:
                errors.append(f"模型配置中缺少 '{field}' 字段")
            elif field == 'sam_checkpoint' and not os.path.exists(model_config[field]):
                errors.append(f"SAM权重文件不存在: {model_config[field]}")
    
    # 检查BMNet配置
    if config.get('bmnet', {}).get('use_bmnet', False):
        bmnet_config = config['bmnet']
        if 'bmnet_cfg' in bmnet_config and not os.path.exists(bmnet_config['bmnet_cfg']):
            errors.append(f"BMNet配置文件不存在: {bmnet_config['bmnet_cfg']}")
    
    if errors:
        print("配置验证失败:")
        for error in errors:
            print(f"  错误: {error}")
        sys.exit(1)
    
    print("✓ 配置验证通过")


def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()
    
    try:
        # 加载配置文件
        print(f"加载配置文件: {args.config}")
        config = load_config(args.config)
        
        # 用命令行参数覆盖配置
        config = override_config_with_args(config, args)
        
        # 验证配置
        validate_config(config)
        
        # 打印配置摘要
        if args.verbose or args.dry_run:
            print_config_summary(config, args)
        
        # 干运行模式
        if args.dry_run:
            print("干运行模式 - 配置验证完成，退出")
            return
        
        # 创建评估器并运行评估
        print("初始化评估器...")
        evaluator = CountingEvaluator(config)
        
        # 确定split参数
        if config['dataset']['name'] == "FSC147":
            split = config.get('split', 'test')
        else:
            split = config['output']['split_tag']
        
        print(f"开始评估 - 数据集: {config['dataset']['name']}, Split: {split}")
        
        # 运行评估
        final_metrics = evaluator.run_evaluation(split)
        
        print("\n" + "=" * 80)
        print("评估完成！")
        print("=" * 80)
        
        return final_metrics
        
    except KeyboardInterrupt:
        print("\n用户中断评估")
        sys.exit(1)
    except Exception as e:
        print(f"评估过程中发生错误: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()