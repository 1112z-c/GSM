"""
可解释性分析演示脚本
Interpretability Analysis Demo Script
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
from interpretability_analysis import InterpretabilityAnalyzer, RDKitFeatureExtractor, SHAPAnalyzer

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_sample_data():
    """创建示例数据用于演示"""
    print("创建示例数据...")
    
    # 创建符合KPI框架要求的示例数据
    sample_data = {
        'SMILES': [
            'CCO',           # 乙醇
            'CC(=O)O',       # 乙酸
            'c1ccccc1',      # 苯
            'CCN',           # 乙胺
            'CCOO',          # 乙二醇
            'CC(C)O',        # 异丙醇
            'CCCC',          # 丁烷
            'c1ccc(cc1)O',   # 苯酚
            'CC(=O)OC',      # 乙酸甲酯
            'CCN(CC)CC',     # 三乙胺
            'c1ccc2ccccc2c1', # 萘
            'CC(C)(C)O',     # 叔丁醇
            'CCCCCC',        # 己烷
            'c1ccc(cc1)Cl',  # 氯苯
            'CC(=O)N',       # 乙酰胺
            'CCCCCCCC',      # 辛烷
            'c1ccc(cc1)Br',  # 溴苯
            'CC(C)(C)CO',    # 新戊醇
            'CCCCCCCCCC',    # 癸烷
            'c1ccc(cc1)I'    # 碘苯
        ],
        'Material_ID': [f'kpi-{i}' for i in range(20)],
        'Formula': [
            'C2H6O', 'C2H4O2', 'C6H6', 'C2H7N', 'C2H6O2',
            'C3H8O', 'C4H10', 'C6H6O', 'C3H6O2', 'C6H15N',
            'C10H8', 'C4H10O', 'C6H14', 'C6H5Cl', 'C2H5NO',
            'C8H18', 'C6H5Br', 'C5H12O', 'C10H22', 'C6H5I'
        ],
        'Molwt': [
            46.07, 60.05, 78.11, 45.08, 62.07,
            60.10, 58.12, 94.11, 74.08, 101.19,
            128.17, 74.12, 86.18, 112.56, 59.07,
            114.23, 157.01, 88.15, 142.28, 204.01
        ],
        '#Heavy': [
            3, 4, 6, 3, 4, 4, 4, 7, 5, 6,
            10, 5, 6, 7, 4, 8, 7, 6, 10, 7
        ],
        'Elements': [
            'C,H,O', 'C,H,O', 'C,H', 'C,H,N', 'C,H,O',
            'C,H,O', 'C,H', 'C,H,O', 'C,H,O', 'C,H,N',
            'C,H', 'C,H,O', 'C,H', 'C,H,Cl', 'C,H,N,O',
            'C,H', 'C,H,Br', 'C,H,O', 'C,H', 'C,H,I'
        ],
        'MP': [
            159.0, 289.0, 278.0, 194.0, 200.0,
            185.0, 134.0, 314.0, 175.0, 158.0,
            353.0, 298.0, 178.0, 228.0, 355.0,
            216.0, 242.0, 256.0, 243.0, 227.0
        ],
        'BP': [
            351.0, 391.0, 353.0, 239.0, 373.0,
            338.0, 272.0, 455.0, 330.0, 363.0,
            491.0, 355.0, 342.0, 405.0, 494.0,
            399.0, 429.0, 375.0, 447.0, 461.0
        ],
        'FP': [
            286.0, 327.0, 262.0, 200.0, 300.0,
            285.0, 213.0, 350.0, 270.0, 250.0,
            350.0, 310.0, 250.0, 320.0, 400.0,
            300.0, 330.0, 290.0, 320.0, 340.0
        ],
        'Source': ['sample'] * 20
    }
    
    return pd.DataFrame(sample_data)

def demonstrate_feature_extraction():
    """演示64维特征提取"""
    print("\n" + "="*60)
    print("64维特征提取演示 (基于5MILES方法)")
    print("=" * 60)
    
    # 创建示例数据
    df = create_sample_data()
    
    # 创建特征提取器
    extractor = RDKitFeatureExtractor()
    
    print(f"\n特征提取器信息:")
    print(f"  特征维度: {extractor.num_features}")
    print(f"  特征名称: {len(extractor.feature_names)} 个")
    
    # 显示特征类型
    print(f"\n特征类型分布:")
    atom_features = [name for name in extractor.feature_names if 'num_' in name and 'atom' in name]
    bond_features = [name for name in extractor.feature_names if 'bond' in name]
    group_features = [name for name in extractor.feature_names if 'group' in name]
    electronic_features = [name for name in extractor.feature_names if any(x in name for x in ['logp', 'tpsa', 'hbd', 'hba', 'charge', 'polar', 'molar', 'dipole', 'electronegativity', 'ionization', 'electron', 'homo', 'lumo', 'band'])]
    
    print(f"  原子特征: {len(atom_features)} 个")
    print(f"  键特征: {len(bond_features)} 个")
    print(f"  官能团特征: {len(group_features)} 个")
    print(f"  电子特征: {len(electronic_features)} 个")
    
    # 提取特征
    print(f"\n开始提取64维特征...")
    smiles_list = df['SMILES'].tolist()
    features = extractor.extract_features_batch(smiles_list)
    
    print(f"  特征矩阵形状: {features.shape}")
    print(f"  特征值范围: [{np.min(features):.3f}, {np.max(features):.3f}]")
    print(f"  特征均值: {np.mean(features):.3f}")
    print(f"  特征标准差: {np.std(features):.3f}")
    
    # 显示前5个分子的前10个特征
    print(f"\n前5个分子的前10个特征值:")
    for i in range(min(5, len(smiles_list))):
        print(f"  {smiles_list[i]}: {features[i, :10]}")
    
    return features

def demonstrate_shap_analysis():
    """演示SHAP分析"""
    print("\n" + "="*60)
    print("SHAP特征重要性分析演示")
    print("=" * 60)
    
    # 创建示例数据
    df = create_sample_data()
    
    # 创建可解释性分析器
    analyzer = InterpretabilityAnalyzer()
    
    # 执行分析
    print(f"\n开始SHAP分析...")
    results = analyzer.analyze_molecular_features(df, ['MP', 'BP', 'FP'])
    
    # 显示分析结果
    target_analysis = results['target_analysis']
    print(f"  分析的目标性质: {list(target_analysis.keys())}")
    
    for target_col, analysis in target_analysis.items():
        print(f"\n{target_col} 分析结果:")
        
        if 'top_features' in analysis:
            top_features = analysis['top_features'][:10]
            print(f"  最重要的10个特征:")
            for i, (idx, name, importance) in enumerate(top_features, 1):
                print(f"    {i:2d}. {name} (索引{idx}): {importance:.4f}")
        
        if 'feature_importance' in analysis:
            importance = analysis['feature_importance']
            print(f"  特征重要性统计:")
            print(f"    最大值: {np.max(importance):.4f}")
            print(f"    最小值: {np.min(importance):.4f}")
            print(f"    平均值: {np.mean(importance):.4f}")
            print(f"    标准差: {np.std(importance):.4f}")
    
    return results

def demonstrate_visualization():
    """演示可视化功能"""
    print("\n" + "="*60)
    print("SHAP可视化演示")
    print("=" * 60)
    
    # 创建示例数据
    df = create_sample_data()
    
    # 创建可解释性分析器
    analyzer = InterpretabilityAnalyzer()
    
    # 执行分析
    print(f"\n开始可解释性分析...")
    results = analyzer.analyze_molecular_features(df, ['MP', 'BP', 'FP'])
    
    # 生成可视化
    print(f"\n生成特征重要性可视化...")
    analyzer.visualize_feature_importance(
        results, 
        save_path='interpretability_feature_importance.png'
    )
    
    print(f"\n生成SHAP摘要可视化...")
    analyzer.visualize_shap_summary(
        results,
        save_path='interpretability_shap_summary.png'
    )
    
    print(f"\n可视化图表已保存:")
    print(f"  • interpretability_feature_importance.png")
    print(f"  • interpretability_shap_summary.png")

def demonstrate_report_generation():
    """演示报告生成"""
    print("\n" + "="*60)
    print("可解释性分析报告生成")
    print("=" * 60)
    
    # 创建示例数据
    df = create_sample_data()
    
    # 创建可解释性分析器
    analyzer = InterpretabilityAnalyzer()
    
    # 执行分析
    print(f"\n开始可解释性分析...")
    results = analyzer.analyze_molecular_features(df, ['MP', 'BP', 'FP'])
    
    # 生成报告
    print(f"\n生成分析报告...")
    report = analyzer.generate_interpretability_report(results)
    
    # 保存报告
    with open('interpretability_analysis_report.txt', 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"\n报告已保存到: interpretability_analysis_report.txt")
    
    # 显示报告摘要
    print(f"\n报告摘要:")
    lines = report.split('\n')
    for line in lines[:20]:  # 显示前20行
        print(f"  {line}")
    print(f"  ... (共 {len(lines)} 行)")

def show_feature_categories():
    """显示特征类别详情"""
    print("\n" + "="*60)
    print("64维特征类别详情")
    print("=" * 60)
    
    extractor = RDKitFeatureExtractor()
    feature_names = extractor.feature_names
    
    print(f"\n1. 原子数量与质量特征 (16维):")
    atom_features = [name for name in feature_names[:16]]
    for i, name in enumerate(atom_features, 1):
        print(f"   {i:2d}. {name}")
    
    print(f"\n2. 键的性质特征 (16维):")
    bond_features = [name for name in feature_names[16:32]]
    for i, name in enumerate(bond_features, 1):
        print(f"   {i:2d}. {name}")
    
    print(f"\n3. 官能团特征 (16维):")
    group_features = [name for name in feature_names[32:48]]
    for i, name in enumerate(group_features, 1):
        print(f"   {i:2d}. {name}")
    
    print(f"\n4. 电子特性特征 (16维):")
    electronic_features = [name for name in feature_names[48:64]]
    for i, name in enumerate(electronic_features, 1):
        print(f"   {i:2d}. {name}")

def show_shap_benefits():
    """显示SHAP分析的优势"""
    print("\n" + "="*60)
    print("SHAP分析优势")
    print("=" * 60)
    
    print(f"\n科学价值:")
    print(f"  ✓ 揭示分子结构与性质的关系")
    print(f"  ✓ 识别关键分子特征")
    print(f"  ✓ 支持分子设计和优化")
    print(f"  ✓ 提供化学直觉和洞察")
    print(f"  ✓ 指导实验设计")
    
    print(f"\n技术优势:")
    print(f"  ✓ 基于SHAP理论的特征重要性")
    print(f"  ✓ 样本级别的特征贡献分析")
    print(f"  ✓ 直观的可视化展示")
    print(f"  ✓ 支持多种机器学习模型")
    print(f"  ✓ 全局和局部解释结合")
    
    print(f"\n应用场景:")
    print(f"  • 分子性质预测模型解释")
    print(f"  • 药物发现中的关键特征识别")
    print(f"  • 材料设计中的结构-性能关系")
    print(f"  • 化学反应的机理理解")
    print(f"  • 分子筛选和优化指导")

def main():
    """主函数"""
    print("=" * 80)
    print("KPI框架可解释性分析演示")
    print("Knowledge-based electrolyte Property prediction Integration Framework")
    print("Interpretability Analysis Demo")
    print("=" * 80)
    
    try:
        # 1. 特征提取演示
        features = demonstrate_feature_extraction()
        
        # 2. SHAP分析演示
        results = demonstrate_shap_analysis()
        
        # 3. 可视化演示
        demonstrate_visualization()
        
        # 4. 报告生成演示
        demonstrate_report_generation()
        
        # 5. 特征类别详情
        show_feature_categories()
        
        # 6. SHAP优势说明
        show_shap_benefits()
        
        print("\n" + "=" * 80)
        print("演示完成！")
        print("=" * 80)
        
        print("\n生成的文件:")
        print("• interpretability_feature_importance.png - 特征重要性分析")
        print("• interpretability_shap_summary.png - SHAP摘要分析")
        print("• interpretability_analysis_report.txt - 详细分析报告")
        
        print("\n功能总结:")
        print("✓ 64维特征提取 (基于5MILES方法)")
        print("✓ SHAP特征重要性分析")
        print("✓ 针对MP、BP、FP的特征排序")
        print("✓ 十大关键特征识别")
        print("✓ 样本点贡献度分析")
        print("✓ 可视化展示")
        print("✓ 详细分析报告")
        
    except Exception as e:
        logger.error(f"演示过程中发生错误: {e}")
        print(f"\n错误: {e}")
        print("请检查依赖包是否正确安装。")

if __name__ == "__main__":
    main()