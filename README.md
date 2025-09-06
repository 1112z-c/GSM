# KPI框架 - 基于知识的电解质属性预测集成框架

Knowledge-based electrolyte Property prediction Integration (KPI) Framework

## 概述

KPI框架是一个基于深度学习的分子属性预测系统，专门用于从Materials Project数据库获取分子数据，并通过知识驱动的机器学习方法预测分子的热力学属性（熔点、沸点、闪点等）。

## 框架架构

框架包含三个主要模块，对应图片中的设计：

### 模块(a): 数据组织和统计分析
- **数据获取**: 从Materials Project数据库获取分子SMILES数据
- **数据预处理**: 数据清洗、统计分析、分子描述符生成
- **输出**: Origin Sheet（原始数据表）和Organised Sheet（组织化数据表）

### 模块(b): 可解释性和知识发现
- **分子嵌入**: 将SMILES字符串转换为数值表示
- **知识向量化**: 将统计知识转换为向量形式
- **输出**: 嵌入分子和向量化知识

### 模块(c): 基于知识的分子属性预测
- **深度学习模型**: 神经网络架构和知识控制器
- **属性预测**: 预测熔点、沸点、闪点等属性
- **反馈系统**: 结果可视化和反馈生成

## 文件结构

```
workspace/
├── data_acquisition.py      # 数据获取模块
├── data_preprocessing.py    # 数据预处理模块
├── molecular_embedding.py   # 分子嵌入模块
├── deep_learning_model.py   # 深度学习模型模块
├── prediction_feedback.py   # 预测和反馈模块
├── main.py                  # 主函数
├── requirements.txt         # 依赖包列表
└── README.md               # 说明文档
```

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 1. 基本使用

```python
from main import KPIFramework

# 创建框架实例
config = {
    'learning_rate': 0.001,
    'batch_size': 32,
    'num_epochs': 100
}
kpi_framework = KPIFramework(config)

# 运行完整流程
results = kpi_framework.run_full_pipeline(
    api_key="your_materials_project_api_key",
    elements=["C", "H", "O"],
    num_materials=100
)
```

### 2. 命令行使用

```bash
# 基本运行
python main.py --api_key your_api_key --elements C H O --num_materials 100

# 指定化学式
python main.py --api_key your_api_key --formula "C2H6O" --num_materials 50

# 使用配置文件
python main.py --config config.json
```

### 3. 分步运行

```python
# 1. 设置模块
kpi_framework.setup_data_acquisition(api_key)
kpi_framework.setup_preprocessing()
kpi_framework.setup_embedding()
kpi_framework.setup_model()

# 2. 数据获取
origin_data = kpi_framework.run_data_acquisition(elements=["C", "H", "O"])

# 3. 数据预处理
organised_data = kpi_framework.run_preprocessing()

# 4. 分子嵌入
molecular_embeddings, knowledge_vectors = kpi_framework.run_embedding()

# 5. 模型训练
training_history = kpi_framework.run_training()

# 6. 属性预测
predictions = kpi_framework.run_prediction(
    molecular_embeddings[0], 
    knowledge_vectors[0]
)
```

## 主要功能

### 数据获取模块 (`data_acquisition.py`)
- 从Materials Project API获取材料数据
- 支持按元素、化学式等条件搜索
- 自动生成SMILES字符串
- 创建原始数据表

### 数据预处理模块 (`data_preprocessing.py`)
- 数据清洗和去重
- 分子描述符计算
- 统计分析
- 数据可视化
- 机器学习预处理

### 分子嵌入模块 (`molecular_embedding.py`)
- SMILES字符串嵌入
- 知识向量化
- 嵌入可视化
- 降维分析

### 深度学习模型模块 (`deep_learning_model.py`)
- 分子属性预测器
- 知识控制器
- 模型训练和评估
- 预测结果可视化

### 预测反馈模块 (`prediction_feedback.py`)
- 属性预测
- 结果可视化
- 反馈生成
- 结果导出

## 配置参数

```python
config = {
    'learning_rate': 0.001,           # 学习率
    'batch_size': 32,                 # 批次大小
    'num_epochs': 100,                # 训练轮数
    'molecular_embedding_dim': 128,   # 分子嵌入维度
    'knowledge_dim': 64,              # 知识向量维度
    'hidden_dim': 256                 # 隐藏层维度
}
```

## 输出文件

运行框架后会生成以下文件：
- `materials_origin_data.csv`: 原始材料数据
- `data_analysis.png`: 数据分析可视化
- `embedding_visualization.png`: 嵌入可视化
- `training_history.png`: 训练历史
- `prediction_evaluation.png`: 预测评估
- `property_predictions.png`: 属性预测结果
- `prediction_results.json`: 预测结果JSON
- `prediction_results.csv`: 预测结果CSV
- `kpi_framework.log`: 运行日志

## 注意事项

1. **API密钥**: 需要有效的Materials Project API密钥
2. **计算资源**: 深度学习训练需要足够的计算资源
3. **数据质量**: 预测结果依赖于输入数据的质量
4. **模型性能**: 建议使用GPU加速训练过程

## 扩展功能

框架设计为模块化结构，可以轻松扩展：
- 添加新的分子描述符
- 实现不同的嵌入方法
- 集成其他数据库
- 添加新的预测属性

## 故障排除

1. **API限制**: Materials Project API有请求限制，建议分批获取数据
2. **内存不足**: 大数据集可能需要调整批次大小
3. **依赖问题**: 确保所有依赖包正确安装

## 许可证

本项目采用MIT许可证。

## 贡献

欢迎提交Issue和Pull Request来改进框架。

## 联系方式

如有问题，请通过GitHub Issues联系。