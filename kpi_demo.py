"""
KPI框架演示脚本 - 符合KPI框架要求
KPI Framework Demo Script - Compliant with KPI Framework Requirements
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
from main import KPIFramework

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_kpi_demo():
    """运行KPI框架演示"""
    print("=" * 80)
    print("KPI框架演示 - 基于知识的电解质属性预测集成框架")
    print("Knowledge-based electrolyte Property prediction Integration (KPI) Framework")
    print("=" * 80)
    
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
    kpi_framework.setup_data_acquisition(
        materials_project_api_key="demo_materials_project_api_key",
        pubchem_api_key="demo_pubchem_api_key",
        crossref_api_key="demo_crossref_api_key"
    )
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
    
    print("\n3. 运行KPI数据获取...")
    # 定义符合KPI要求的搜索参数
    search_terms = ["electrolyte", "ionic liquid", "solvent", "electrolyte solution"]
    elements = ["C", "H", "O", "N", "F", "P", "Cl", "Br", "I"]  # 符合KPI要求的元素
    
    # 由于没有真实API密钥，框架会自动使用示例数据
    origin_data = kpi_framework.run_data_acquisition(
        search_terms=search_terms,
        elements=elements,
        max_materials_per_source=50
    )
    
    print(f"   获取了 {len(origin_data)} 个符合KPI要求的分子数据")
    print("   KPI框架数据预览:")
    print(origin_data[['SMILES', 'Molwt', '#Heavy', 'MP', 'BP', 'FP']].head())
    
    # 显示KPI框架合规性
    print(f"\n   KPI框架合规性检查:")
    print(f"   分子量范围: {origin_data['Molwt'].min():.2f} - {origin_data['Molwt'].max():.2f} (要求: 0-600)")
    print(f"   重原子数范围: {origin_data['#Heavy'].min()} - {origin_data['#Heavy'].max()} (要求: 0-30)")
    
    # 检查元素合规性
    all_elements = set()
    for elements_str in origin_data['Elements'].dropna():
        elements_list = [e.strip() for e in str(elements_str).split(',')]
        all_elements.update(elements_list)
    
    allowed_elements = {'H', 'C', 'N', 'O', 'F', 'Si', 'P', 'Cl', 'Br', 'I'}
    disallowed_elements = all_elements - allowed_elements
    print(f"   元素合规性: {'✓ 合规' if len(disallowed_elements) == 0 else f'✗ 包含不允许的元素: {disallowed_elements}'}")
    
    print("\n4. 运行KPI数据预处理...")
    organised_data = kpi_framework.run_preprocessing()
    print(f"   KPI组织化数据包含 {len(organised_data)} 行，{len(organised_data.columns)} 列")
    print("   KPI组织化数据预览:")
    print(organised_data[['ID', 'SMILES', 'Molwt', '#Heavy', 'MP', 'BP', 'FP']].head())
    
    # 显示目标属性统计
    print(f"\n   目标属性统计:")
    for prop in ['MP', 'BP', 'FP']:
        if prop in organised_data.columns:
            non_null_count = organised_data[prop].notna().sum()
            print(f"   {prop}: {non_null_count} 个有效值")
    
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
    
    print("\n7. 运行KPI属性预测...")
    if len(molecular_embeddings) > 0:
        predictions = kpi_framework.run_prediction(
            molecular_embeddings[0], 
            knowledge_vectors[0]
        )
        print("   KPI框架预测结果:")
        for prop_name, pred_data in predictions.items():
            if isinstance(pred_data, dict) and 'value' in pred_data:
                print(f"   {pred_data['name']}: {pred_data['value']:.2f} {pred_data['unit']} (置信度: {pred_data['confidence']:.2f})")
    else:
        print("   没有可用的嵌入数据进行预测")
    
    print("\n8. 生成KPI框架输出文件...")
    print("   以下文件已生成:")
    print("   - kpi_origin_data.csv: KPI原始材料数据")
    print("   - data_analysis.png: KPI数据分析可视化")
    print("   - embedding_visualization.png: 嵌入可视化")
    print("   - training_history.png: 训练历史")
    print("   - prediction_evaluation.png: 预测评估")
    print("   - property_predictions.png: 属性预测结果")
    print("   - prediction_results.json: 预测结果JSON")
    print("   - prediction_results.csv: 预测结果CSV")
    print("   - kpi_framework.log: 运行日志")
    
    print("\n" + "=" * 80)
    print("KPI框架演示完成！")
    print("=" * 80)
    
    # 显示KPI框架特性
    print("\nKPI框架特性:")
    print("✓ 从论文和公共数据库自动收集数据")
    print("✓ 分子结构采用SMILES表示")
    print("✓ 目标属性：熔点(MP)、沸点(BP)、闪点(FP)")
    print("✓ 分子量限制：0-600 g/mol")
    print("✓ 重原子数限制：0-30")
    print("✓ 元素限制：H, C, N, O, F, Si, P, Cl, Br, I")
    print("✓ 形成分子结构及其对应性质的二进制数据集")
    print("✓ 基于常规电解质的分子性质分布")
    
    print("\n框架架构总结:")
    print("模块(a): 数据组织和统计分析")
    print("  - 利用API从论文和公共数据库收集数据")
    print("  - 数据清洗、统计分析、分子描述符生成")
    print("  - 输出: Origin Sheet 和 Organised Sheet")
    print("  - 符合KPI框架的数据限制要求")
    
    print("\n模块(b): 可解释性和知识发现")
    print("  - 分子嵌入: SMILES转数值表示")
    print("  - 知识向量化: 统计知识转向量形式")
    print("  - 输出: 嵌入分子和向量化知识")
    
    print("\n模块(c): 基于知识的分子属性预测")
    print("  - 深度学习模型: 神经网络架构和知识控制器")
    print("  - 属性预测: 熔点、沸点、闪点等")
    print("  - 反馈系统: 结果可视化和反馈生成")

def show_kpi_requirements():
    """显示KPI框架要求"""
    print("\nKPI框架数据要求:")
    print("1. 数据来源:")
    print("   - 已发表论文（通过CrossRef API）")
    print("   - 公共数据库（Materials Project, PubChem）")
    print("   - 自动收集和格式化")
    
    print("\n2. 分子表示:")
    print("   - 简化分子输入线条系统（SMILES）")
    print("   - 形成分子结构及其对应性质的二进制数据集")
    
    print("\n3. 目标属性:")
    print("   - 熔点（MP, Melting Point）")
    print("   - 沸点（BP, Boiling Point）")
    print("   - 闪点（FP, Flash Point）")
    
    print("\n4. 数据限制:")
    print("   - 分子量（Molwt）: 0-600 g/mol")
    print("   - 重原子数（#Heavy）: 0-30")
    print("   - 元素限制: H, C, N, O, F, Si, P, Cl, Br, I")
    print("   - 基于常规电解质的分子性质分布")
    
    print("\n5. 数据质量:")
    print("   - 自动数据清洗和验证")
    print("   - 合规性检查")
    print("   - 统计分析")
    print("   - 可视化展示")

if __name__ == "__main__":
    try:
        run_kpi_demo()
        show_kpi_requirements()
    except Exception as e:
        logger.error(f"演示过程中发生错误: {e}")
        print(f"\n错误: {e}")
        print("请检查依赖包是否正确安装，并查看日志文件获取详细信息。")