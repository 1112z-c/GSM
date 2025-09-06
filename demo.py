"""
KPI框架演示脚本
KPI Framework Demo Script
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
from main import KPIFramework

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_demo():
    """运行KPI框架演示"""
    print("=" * 60)
    print("KPI框架演示 - 基于知识的电解质属性预测集成框架")
    print("=" * 60)
    
    # 配置参数
    config = {
        'learning_rate': 0.001,
        'batch_size': 16,  # 较小的批次大小用于演示
        'num_epochs': 20,  # 较少的训练轮数用于演示
        'molecular_embedding_dim': 64,  # 较小的嵌入维度用于演示
        'knowledge_dim': 32,
        'hidden_dim': 128
    }
    
    print("\n1. 初始化KPI框架...")
    kpi_framework = KPIFramework(config)
    
    print("\n2. 设置所有模块...")
    # 使用示例API密钥（实际使用时需要替换为真实密钥）
    kpi_framework.setup_data_acquisition("demo_api_key")
    kpi_framework.setup_preprocessing()
    kpi_framework.setup_embedding(
        molecular_embedding_dim=config['molecular_embedding_dim'],
        knowledge_dim=config['knowledge_dim']
    )
    kpi_framework.setup_model(
        molecular_embedding_dim=config['molecular_embedding_dim'],
        knowledge_dim=config['knowledge_dim'],
        hidden_dim=config['hidden_dim']
    )
    
    print("\n3. 运行数据获取（使用示例数据）...")
    # 由于没有真实API密钥，框架会自动使用示例数据
    origin_data = kpi_framework.run_data_acquisition(
        elements=["C", "H", "O"],
        num_materials=20
    )
    print(f"   获取了 {len(origin_data)} 个材料的数据")
    print("   示例数据预览:")
    print(origin_data[['SMILES', 'Formula', 'Density']].head())
    
    print("\n4. 运行数据预处理...")
    organised_data = kpi_framework.run_preprocessing()
    print(f"   组织化数据包含 {len(organised_data)} 行，{len(organised_data.columns)} 列")
    print("   分子描述符预览:")
    descriptor_cols = [col for col in organised_data.columns if col.startswith('#') or col in ['MW', 'LogP', 'TPSA']]
    if descriptor_cols:
        print(organised_data[descriptor_cols].head())
    
    print("\n5. 运行分子嵌入和知识向量化...")
    molecular_embeddings, knowledge_vectors = kpi_framework.run_embedding()
    print(f"   分子嵌入形状: {molecular_embeddings.shape}")
    print(f"   知识向量形状: {knowledge_vectors.shape}")
    
    print("\n6. 运行模型训练...")
    training_history = kpi_framework.run_training(
        train_ratio=0.7,
        val_ratio=0.15,
        num_epochs=config['num_epochs'],
        batch_size=config['batch_size']
    )
    print("   训练完成！")
    print(f"   最终训练损失: {training_history['total_loss'][-1]:.4f}")
    print(f"   最终验证损失: {kpi_framework.trainer.val_history['total_loss'][-1]:.4f}")
    
    print("\n7. 运行属性预测...")
    if len(molecular_embeddings) > 0:
        predictions = kpi_framework.run_prediction(
            molecular_embeddings[0], 
            knowledge_vectors[0]
        )
        print("   预测结果:")
        for prop_name, pred_data in predictions.items():
            if isinstance(pred_data, dict) and 'value' in pred_data:
                print(f"   {pred_data['name']}: {pred_data['value']:.2f} {pred_data['unit']} (置信度: {pred_data['confidence']:.2f})")
    else:
        print("   没有可用的嵌入数据进行预测")
    
    print("\n8. 生成输出文件...")
    print("   以下文件已生成:")
    print("   - materials_origin_data.csv: 原始材料数据")
    print("   - data_analysis.png: 数据分析可视化")
    print("   - embedding_visualization.png: 嵌入可视化")
    print("   - training_history.png: 训练历史")
    print("   - prediction_evaluation.png: 预测评估")
    print("   - property_predictions.png: 属性预测结果")
    print("   - prediction_results.json: 预测结果JSON")
    print("   - prediction_results.csv: 预测结果CSV")
    print("   - kpi_framework.log: 运行日志")
    
    print("\n" + "=" * 60)
    print("KPI框架演示完成！")
    print("=" * 60)
    
    # 显示框架架构信息
    print("\n框架架构总结:")
    print("模块(a): 数据组织和统计分析")
    print("  - 从Materials Project数据库获取分子SMILES数据")
    print("  - 数据清洗、统计分析、分子描述符生成")
    print("  - 输出: Origin Sheet 和 Organised Sheet")
    
    print("\n模块(b): 可解释性和知识发现")
    print("  - 分子嵌入: SMILES转数值表示")
    print("  - 知识向量化: 统计知识转向量形式")
    print("  - 输出: 嵌入分子和向量化知识")
    
    print("\n模块(c): 基于知识的分子属性预测")
    print("  - 深度学习模型: 神经网络架构和知识控制器")
    print("  - 属性预测: 熔点、沸点、闪点等")
    print("  - 反馈系统: 结果可视化和反馈生成")

def show_framework_info():
    """显示框架信息"""
    print("\nKPI框架特性:")
    print("- 模块化设计，易于扩展和维护")
    print("- 支持从Materials Project数据库获取数据")
    print("- 自动化的数据预处理和特征工程")
    print("- 先进的分子嵌入和知识向量化")
    print("- 基于深度学习的属性预测")
    print("- 完整的可视化和反馈系统")
    print("- 支持多种输出格式")
    
    print("\n适用场景:")
    print("- 分子热力学属性预测")
    print("- 材料科学研究和开发")
    print("- 化学信息学分析")
    print("- 机器学习模型开发")
    print("- 教育和研究目的")

if __name__ == "__main__":
    try:
        run_demo()
        show_framework_info()
    except Exception as e:
        logger.error(f"演示过程中发生错误: {e}")
        print(f"\n错误: {e}")
        print("请检查依赖包是否正确安装，并查看日志文件获取详细信息。")