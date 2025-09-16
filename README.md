# Reddit Data Analysis Pipeline

A comprehensive data analysis pipeline for Reddit data, featuring data preprocessing, text classification, network analysis, and visualization.

## Project Overview

This project provides a complete end-to-end pipeline for analyzing Reddit data, from raw JSON files to publication-ready results. It includes sentiment analysis, user interaction modeling, and information diffusion visualization.

### Key Features
- **Scalable Data Processing**: Handles large Reddit datasets efficiently
- **Machine Learning Pipeline**: TF-IDF + Logistic Regression baseline model
- **Network Analysis**: User interaction graphs and community detection
- **Visualization**: Comprehensive charts and plots for analysis
- **One-Click Execution**: Automated batch scripts for easy deployment

## 数据来源

- **数据源**: Academic Torrents Reddit 2024-12
- **时间范围**: 2024年12月1日-7日（一周数据）
- **目标子版块**: politics, worldnews, AskReddit
- **文件格式**: JSON Lines (.json)

## 环境要求

### 推荐环境
- **Python**: 3.10+ (推荐使用conda环境)
- **内存**: 16GB+ RAM
- **存储**: SSD推荐

### 依赖安装

```bash
# 基础依赖
pip install -U pip wheel setuptools

# 核心依赖
pip install "pyarrow>=17" "fastparquet>=2024.5" pandas tqdm scikit-learn matplotlib networkx joblib pydantic ruamel.yaml
```

### Conda环境（推荐）

```bash
conda create -n at310 python=3.10 -y
conda activate at310
pip install "pyarrow>=17" "fastparquet>=2024.5" pandas tqdm scikit-learn matplotlib networkx joblib pydantic ruamel.yaml
```

## 项目结构

```
project_root/
├── configs/                 # 配置文件
│   ├── exp_small.yaml      # 小样本配置
│   └── exp_full.yaml       # 全量配置
├── scripts/                 # 分析脚本
│   ├── 00_env_check.py     # 环境检查
│   ├── 01_slice_dataset.py # 数据切片
│   ├── 02_verify_parquet.py# 数据验证
│   ├── 03_prepare_corpus.py# 语料准备
│   ├── 04_baseline_tfidf_lr.py # 基线模型
│   ├── 05_build_graph.py   # 图构建
│   ├── 06_visualize_diffusion.py # 扩散可视化
│   └── 07_eval_report.py   # 评估报告
├── artifacts/              # 中间产物
├── figures/                # 图表输出
├── mini_dataset/           # 切片数据
├── run_small.bat          # 小样本一键运行
├── run_full.bat           # 全量一键运行
└── README.md              # 本文档
```

## 快速开始

### 小样本模式（推荐首次运行）

```bash
# Windows
run_small.bat

# 或手动运行
python scripts/00_env_check.py --config configs/exp_small.yaml
python scripts/01_slice_dataset.py --config configs/exp_small.yaml
# ... 其他脚本
```

**预期时间**: 30-60分钟  
**数据量**: 10K submissions, 200K comments

### 全量模式

```bash
# Windows
run_full.bat

# 或手动运行
python scripts/00_env_check.py --config configs/exp_full.yaml
python scripts/01_slice_dataset.py --config configs/exp_full.yaml
# ... 其他脚本
```

**预期时间**: 数小时  
**数据量**: 一周全量数据

## 分析流程

### 1. 环境检查 (`00_env_check.py`)
- 检查Python版本和依赖包
- 自动安装缺失的包
- 验证parquet引擎可用性

### 2. 数据切片 (`01_slice_dataset.py`)
- 按子版块和时间过滤原始JSON数据
- 分片写入Parquet格式
- 生成submissions和comments的切片文件

### 3. 数据验证 (`02_verify_parquet.py`)
- 验证切片数据完整性
- 生成数据统计摘要
- 创建数据分布图表

### 4. 语料准备 (`03_prepare_corpus.py`)
- 构建弱标注文本语料
- 文本清洗和预处理
- 基于关键词的代理标签生成

### 5. 基线模型 (`04_baseline_tfidf_lr.py`)
- TF-IDF特征提取
- 逻辑回归分类器
- 时间序列数据切分
- 性能评估和可视化

### 6. 图构建 (`05_build_graph.py`)
- 用户交互图构建
- 回复关系和共评论关系
- 图统计分析和度分布

### 7. 扩散可视化 (`06_visualize_diffusion.py`)
- 热门讨论线程分析
- 层级树和力导向图
- 讨论深度和时间跨度分析

### 8. 评估报告 (`07_eval_report.py`)
- 汇总所有分析结果
- 生成PPT要点
- 创建数据表格

## 输出文件

### artifacts/ 目录
- `summary.json` - 数据统计摘要
- `corpus.parquet` - 文本语料
- `baseline_metrics.json` - 基线模型指标
- `baseline_model.joblib` - 训练好的模型
- `graph_user.gpickle` - 用户交互图
- `graph_stats.json` - 图统计信息
- `thread_*_stats.json` - 热门线程统计
- `table_numbers.csv` - 汇总数据表
- `slide_bullets.md` - PPT要点

### figures/ 目录
- `subreddit_bar.png` - 子版块分布
- `timestamp_hist.png` - 时间戳分布
- `label_balance.png` - 标签平衡
- `confusion_matrix.png` - 混淆矩阵
- `pr_curve.png` - 精确率-召回率曲线
- `feature_top_terms.png` - 特征重要性
- `degree_hist.png` - 度分布
- `thread_tree_*.png` - 讨论树
- `thread_force_*.png` - 力导向图

## 配置说明

### 小样本配置 (exp_small.yaml)
- 限制数据量便于快速测试
- 适合开发和调试
- 内存需求较低

### 全量配置 (exp_full.yaml)
- 处理完整一周数据
- 适合生产分析
- 需要更多计算资源

## 重要说明

### 弱标注标签
当前使用的文本标签为**代理标签/弱标注**，基于简单关键词匹配生成，仅用于演示流程。正式分析需要：
- 人工标注数据
- 公开标注数据集
- 更复杂的标签生成方法

### 数据隐私
- 仅处理公开可用的Reddit数据
- 遵循Academic Torrents使用条款
- 不存储个人敏感信息

### 性能优化
- 使用Parquet格式提高I/O效率
- 分片处理避免内存溢出
- 可配置的数据量限制

## 故障排除

### 常见问题

1. **内存不足**
   - 使用小样本模式
   - 减少chunksize参数
   - 增加虚拟内存

2. **Parquet引擎错误**
   - 运行环境检查脚本
   - 重新安装pyarrow/fastparquet
   - 检查Python版本兼容性

3. **数据文件缺失**
   - 确认原始JSON文件路径正确
   - 检查文件名格式
   - 验证文件权限

### 日志和调试
- 所有脚本都有详细的控制台输出
- 错误信息会显示具体失败步骤
- 可以单独运行每个脚本进行调试

## 扩展和定制

### 添加新的分析模块
1. 在`scripts/`目录创建新脚本
2. 更新配置文件添加新参数
3. 修改批处理文件包含新步骤

### 修改数据源
1. 更新配置文件中的文件路径
2. 调整数据过滤条件
3. 修改字段映射逻辑

### 自定义可视化
1. 修改matplotlib样式配置
2. 添加新的图表类型
3. 调整图表尺寸和DPI

## 许可证

本项目基于Academic Torrents Reddit数据，遵循相应的使用条款。

## 联系方式

如有问题或建议，请通过项目仓库提交Issue。
