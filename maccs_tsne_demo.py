"""
MACCS和t-SNE分析演示脚本
MACCS and t-SNE Analysis Demo Script
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
from data_preprocessing import DataPreprocessor

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

def demonstrate_maccs_encoding():
    """演示MACCS编码功能"""
    print("\n" + "="*60)
    print("MACCS分子访问系统编码演示")
    print("="*60)
    
    # 创建示例数据
    df = create_sample_data()
    
    # 创建预处理器
    preprocessor = DataPreprocessor()
    
    # 计算MACCS指纹
    print("\n1. 计算MACCS分子指纹...")
    maccs_fingerprints = preprocessor.calculate_maccs_fingerprints(df)
    print(f"   MACCS指纹形状: {maccs_fingerprints.shape}")
    print(f"   MACCS键数量: {maccs_fingerprints.shape[1]}")
    
    # 显示MACCS键信息
    print(f"\n2. MACCS键信息:")
    print(f"   总键数: {preprocessor.maccs_encoder.num_keys}")
    print(f"   键类型: {preprocessor.maccs_encoder.maccs_keys[:10]}...")  # 显示前10个键
    
    # 分析MACCS指纹
    print(f"\n3. MACCS指纹分析:")
    key_activity = np.sum(maccs_fingerprints, axis=0)
    active_keys = np.sum(key_activity > 0)
    print(f"   活跃键数: {active_keys}")
    print(f"   最活跃的键: {np.argmax(key_activity)} (活跃度: {np.max(key_activity)})")
    
    return maccs_fingerprints

def demonstrate_tsne_clustering():
    """演示t-SNE聚类功能"""
    print("\n" + "="*60)
    print("t-SNE聚类分析演示")
    print("="*60)
    
    # 创建示例数据
    df = create_sample_data()
    
    # 创建预处理器
    preprocessor = DataPreprocessor()
    
    # 计算MACCS指纹
    maccs_fingerprints = preprocessor.calculate_maccs_fingerprints(df)
    
    # 执行t-SNE聚类（根据熔点颜色标注）
    print("\n1. 执行t-SNE聚类分析...")
    tsne_results = preprocessor.perform_tsne_clustering(
        maccs_fingerprints, 
        df['MP'].values, 
        "熔点(MP)"
    )
    
    print(f"   t-SNE坐标形状: {tsne_results['tsne_coords'].shape}")
    print(f"   聚类标签形状: {tsne_results['cluster_labels'].shape}")
    print(f"   性质值形状: {tsne_results['property_values'].shape}")
    
    # 聚类统计
    unique_clusters, cluster_counts = np.unique(tsne_results['cluster_labels'], return_counts=True)
    print(f"\n2. 聚类统计:")
    print(f"   聚类数: {len(unique_clusters)}")
    for cluster_id, count in zip(unique_clusters, cluster_counts):
        print(f"   聚类 {cluster_id}: {count} 个分子")
    
    return tsne_results

def demonstrate_visualization():
    """演示可视化功能"""
    print("\n" + "="*60)
    print("分子分布可视化演示")
    print("="*60)
    
    # 创建示例数据
    df = create_sample_data()
    
    # 创建预处理器
    preprocessor = DataPreprocessor()
    
    # 执行完整的MACCS和t-SNE分析
    print("\n1. 执行完整的MACCS和t-SNE分析...")
    analysis_results = preprocessor.perform_maccs_tsne_analysis(
        df, 
        property_column='MP',
        save_path='molecular_distribution_analysis.png'
    )
    
    print("   分析完成！")
    print(f"   聚类数: {analysis_results['cluster_analysis']['cluster_statistics']['num_clusters']}")
    
    # 显示聚类分析结果
    print(f"\n2. 聚类分析结果:")
    cluster_stats = analysis_results['cluster_analysis']['cluster_statistics']
    print(f"   最大聚类: {cluster_stats['largest_cluster']} ({cluster_stats['cluster_sizes'][cluster_stats['largest_cluster']]} 个分子)")
    print(f"   最小聚类: {cluster_stats['smallest_cluster']} ({cluster_stats['cluster_sizes'][cluster_stats['smallest_cluster']]} 个分子)")
    
    # 显示各聚类的性质统计
    if 'cluster_property_analysis' in analysis_results['cluster_analysis']:
        print(f"\n3. 各聚类熔点统计:")
        prop_analysis = analysis_results['cluster_analysis']['cluster_property_analysis']
        for cluster_id, stats in prop_analysis.items():
            print(f"   聚类 {cluster_id}: 均值={stats['mean']:.1f}K, 标准差={stats['std']:.1f}K")
    
    return analysis_results

def demonstrate_multiple_properties():
    """演示多个性质的颜色标注"""
    print("\n" + "="*60)
    print("多性质颜色标注演示")
    print("="*60)
    
    # 创建示例数据
    df = create_sample_data()
    
    # 创建预处理器
    preprocessor = DataPreprocessor()
    
    # 对每个性质进行分析
    properties = ['MP', 'BP', 'FP']
    property_names = ['熔点', '沸点', '闪点']
    
    for prop, prop_name in zip(properties, property_names):
        print(f"\n{prop_name} ({prop}) 分析:")
        
        try:
            analysis_results = preprocessor.perform_maccs_tsne_analysis(
                df, 
                property_column=prop,
                save_path=f'molecular_distribution_{prop}.png'
            )
            
            cluster_stats = analysis_results['cluster_analysis']['cluster_statistics']
            print(f"   聚类数: {cluster_stats['num_clusters']}")
            
            if 'cluster_property_analysis' in analysis_results['cluster_analysis']:
                prop_analysis = analysis_results['cluster_analysis']['cluster_property_analysis']
                for cluster_id, stats in prop_analysis.items():
                    print(f"   聚类 {cluster_id}: 均值={stats['mean']:.1f}K")
        
        except Exception as e:
            print(f"   {prop_name}分析失败: {e}")

def main():
    """主函数"""
    print("="*80)
    print("KPI框架MACCS和t-SNE分析演示")
    print("Knowledge-based electrolyte Property prediction Integration (KPI) Framework")
    print("MACCS and t-SNE Analysis Demo")
    print("="*80)
    
    try:
        # 1. MACCS编码演示
        maccs_fingerprints = demonstrate_maccs_encoding()
        
        # 2. t-SNE聚类演示
        tsne_results = demonstrate_tsne_clustering()
        
        # 3. 可视化演示
        analysis_results = demonstrate_visualization()
        
        # 4. 多性质分析演示
        demonstrate_multiple_properties()
        
        print("\n" + "="*80)
        print("演示完成！")
        print("="*80)
        
        print("\n生成的文件:")
        print("• molecular_distribution_analysis.png - 分子分布分析（熔点）")
        print("• molecular_distribution_MP.png - 熔点分析")
        print("• molecular_distribution_BP.png - 沸点分析")
        print("• molecular_distribution_FP.png - 闪点分析")
        
        print("\n功能总结:")
        print("✓ MACCS分子访问系统编码")
        print("✓ t-SNE聚类分析")
        print("✓ 根据性质进行颜色标注")
        print("✓ 分子分布可视化")
        print("✓ 聚类统计分析")
        print("✓ 多性质对比分析")
        
    except Exception as e:
        logger.error(f"演示过程中发生错误: {e}")
        print(f"\n错误: {e}")
        print("请检查依赖包是否正确安装。")

if __name__ == "__main__":
    main()