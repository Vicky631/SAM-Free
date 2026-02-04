# SAM-Free 多数据集计数评估系统

基于YAML配置文件的统一多数据集计数和定位评估系统，支持Sheep、FSC147等多种数据集格式。

## 项目结构

```
SAM-Free/
├── main.py                 # 统一主文件
├── dataset_manager.py      # 数据集管理器
├── evaluator.py           # 评估器
├── configs/               # 配置文件目录
│   ├── sheep.yaml         # Sheep数据集配置
│   └── fsc147.yaml        # FSC147数据集配置
├── utils.py               # 工具函数
├── myutil/                # 数据集加载器
│   ├── SheepOBB.py
│   ├── FSC_147.py
│   └── localization.py
└── shi_segment_anything/  # SAM模型
```

## 快速开始

### 1. 环境准备

确保已安装必要的依赖包：
```bash
pip install torch torchvision opencv-python pillow scipy tqdm pyyaml numpy
```

### 2. 配置文件

配置文件位于 `configs/` 目录下，每个数据集对应一个YAML文件。

#### Sheep数据集配置 (configs/sheep.yaml)
```yaml
dataset:
  name: "Sheep_OBB"
  image_dir: "/path/to/sheep/images/"
  annotation_dir: "/path/to/sheep/annotations/"
  
model:
  sam_checkpoint: "/path/to/sam_weights.pth"
  
training:
  prompt_type: "box"
  prompt_box_num: 3
```

#### FSC147数据集配置 (configs/fsc147.yaml)
```yaml
dataset:
  name: "FSC147"
  annotation_file: "/path/to/fsc147/annotations.json"
  image_root: "/path/to/fsc147/images/"
  
model:
  sam_checkpoint: "/path/to/sam_weights.pth"
```

### 3. 运行评估

#### 基本用法
```bash
# 评估Sheep数据集
python main.py --config configs/sheep.yaml --device cuda:0

# 评估FSC147数据集
python main.py --config configs/fsc147.yaml --device cuda:1

# 指定数据集划分
python main.py --config configs/fsc147.yaml --split test --device cuda:0
```

#### 高级用法
```bash
# 使用不同的prompt类型
python main.py --config configs/sheep.yaml --prompt_type point

# 启用BMNet候选点生成
python main.py --config configs/sheep.yaml --use_bmnet

# 保存可视化结果
python main.py --config configs/fsc147.yaml --save_vis

# 干运行模式（仅验证配置）
python main.py --config configs/sheep.yaml --dry_run

# 详细输出模式
python main.py --config configs/sheep.yaml --verbose

# 显示候选点各阶段统计信息（调试用）
python main.py --config configs/sheep.yaml --verbose_point_stats
```

## 支持的数据集

### 1. Sheep_OBB 数据集
- **标注格式**: DOTA格式的旋转框标注
- **文件结构**:
  ```
  Sheep_obb/
  ├── img/           # 图像文件
  └── DOTA/          # 标注文件(.txt)
  ```
- **特点**: 支持BMNet候选点生成

### 2. FSC147 数据集
- **标注格式**: JSON格式，包含点标注和示例框
- **文件结构**:
  ```
  FSC_147/
  ├── images_384_VarV2/              # 图像文件
  ├── annotation_FSC147_384_with_gt.json  # 标注文件
  └── Train_Test_Val_FSC_147.json    # 数据集划分
  ```
- **特点**: 预定义的训练/测试/验证集划分

## 配置文件详解

### 数据集配置 (dataset)
```yaml
dataset:
  name: "数据集名称"           # Sheep_OBB 或 FSC147
  type: "标注类型"            # obb_annotation, point_annotation
  # 数据集特定的路径配置...
```

### 模型配置 (model)
```yaml
model:
  sam_checkpoint: "SAM权重路径"
  model_type: "vit_b"         # SAM模型类型
```

### 训练参数 (training)
```yaml
training:
  prompt_type: "box"          # box 或 point
  prompt_box_num: 3           # prompt数量
  prompt_select: "head"       # head 或 random
  seed: 0                     # 随机种子
```

### BMNet配置 (bmnet)
```yaml
bmnet:
  use_bmnet: true             # 是否使用BMNet
  bmnet_cfg: "BMNet配置路径"
  bmnet_checkpoint: "BMNet权重路径"
```

### SAM参数 (sam_params)
```yaml
sam_params:
  points_per_side: 64         # 网格点密度
  points_per_batch: 64        # 批处理大小
  pred_iou_thresh: 0.88       # IoU阈值
  stability_score_thresh: 0.85 # 稳定性阈值
```

### 评估配置 (evaluation)
```yaml
evaluation:
  metrics: ["MAE", "RMSE", "F1"]  # 评估指标
  distance_thresh: 10.0           # 定位匹配距离阈值
```

### 输出配置 (output)
```yaml
output:
  base_dir: "./logsSave"      # 输出根目录
  save_visualization: true    # 是否保存可视化
```

## 输出结果

### 日志文件
- 位置: `{base_dir}/{dataset_name}/logs/log-{split}-{prompt_type}.csv`
- 格式: CSV文件，包含每张图片的详细指标

### 可视化文件（可选）
- 位置: `{base_dir}/{dataset_name}/{split}/{prompt_type}/vis_*/`
- 内容: 预测mask的可视化图像

### 控制台输出
```
[test | box] 最终结果:
计数指标: MAE=2.45, RMSE=3.21, NAE=0.15, SRE=0.18
定位指标: F1=0.85, P=0.88, R=0.82
处理图片数: 1000
```

## VS Code 调试配置

项目包含预配置的VS Code调试设置，支持以下场景：

1. **Sheep Dataset (BMNet + Visualization)** - 启用BMNet和可视化
2. **Sheep Dataset (Basic)** - 基础配置
3. **Sheep Dataset (Point Prompt)** - 使用点提示
4. **FSC147 Dataset (Test Set)** - FSC147测试集
5. **FSC147 Dataset (Val Set)** - FSC147验证集
6. **Config Validation (Dry Run)** - 配置验证模式
7. **CPU Mode (Debug)** - CPU调试模式
8. **Custom Output Directory** - 自定义输出目录

## 扩展新数据集

### 1. 创建配置文件
在 `configs/` 目录下创建新的YAML配置文件。

### 2. 实现数据加载器
在 `myutil/` 目录下实现新的数据集加载器类。

### 3. 更新数据集管理器
在 `dataset_manager.py` 中添加对新数据集的支持。

### 示例：添加CARPK数据集
```yaml
# configs/carpk.yaml
dataset:
  name: "CARPK"
  type: "box_annotation"
  data_path: "./dataset/CARPK/"
  annotation_dir: "Annotations"
  split_dir: "ImageSets"
  image_dir: "Images"
```

## 故障排除

### 常见问题

1. **配置文件路径错误**
   ```bash
   python main.py --config configs/sheep.yaml --dry_run
   ```

2. **CUDA内存不足**
   - 减少 `points_per_batch` 参数
   - 使用CPU: `--device cpu`

3. **BMNet加载失败**
   - 检查BMNet路径配置
   - 使用 `--no_bmnet` 禁用BMNet

4. **数据集路径不存在**
   - 检查配置文件中的路径设置
   - 使用绝对路径

### 调试模式
```bash
# 详细输出
python main.py --config configs/sheep.yaml --verbose

# 干运行（仅验证配置）
python main.py --config configs/sheep.yaml --dry_run
```

## 性能优化

1. **GPU内存优化**
   - 调整 `points_per_batch` 参数
   - 使用更小的 `points_per_side`

2. **速度优化**
   - 启用BMNet候选点生成
   - 适当调整质量阈值

3. **存储优化**
   - 禁用可视化: `save_visualization: false`
   - 定期清理输出目录