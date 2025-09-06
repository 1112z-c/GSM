"""
KPI框架主函数
Knowledge-based electrolyte Property prediction Integration (KPI) Framework Main Function
"""

import os
import sys
import logging
import argparse
import numpy as np
import pandas as pd
import torch
from typing import Optional, Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# 导入各个模块
from data_acquisition import KPIDataAcquisition
from data_preprocessing import DataPreprocessor, MolecularDescriptorCalculator
from molecular_embedding import MolecularEmbedder, KnowledgeVectorizer, EmbeddingVisualizer
from deep_learning_model import (
    MolecularPropertyPredictor, ModelTrainer, ModelEvaluator, 
    MolecularDataset
)
from prediction_feedback import (
    PropertyPredictor, ResultVisualizer, FeedbackGenerator, ResultExporter
)

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('kpi_framework.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class KPIFramework:
    """KPI框架主类"""
    
    def __init__(self, config: Dict):
        """
        初始化KPI框架
        
        Args:
            config (Dict): 配置参数
        """
        self.config = config
        self.data_acquirer = None
        self.preprocessor = None
        self.embedder = None
        self.vectorizer = None
        self.model = None
        self.trainer = None
        self.evaluator = None
        self.predictor = None
        
        # 数据存储
        self.origin_data = None
        self.organised_data = None
        self.molecular_embeddings = None
        self.knowledge_vectors = None
        self.training_data = None
        
        logger.info("KPI框架初始化完成")
    
    def setup_data_acquisition(self, 
                              materials_project_api_key: str = None,
                              pubchem_api_key: str = None,
                              crossref_api_key: str = None):
        """
        设置数据获取模块
        
        Args:
            materials_project_api_key (str, optional): Materials Project API密钥
            pubchem_api_key (str, optional): PubChem API密钥
            crossref_api_key (str, optional): CrossRef API密钥
        """
        logger.info("设置KPI数据获取模块...")
        self.data_acquirer = KPIDataAcquisition(
            materials_project_api_key=materials_project_api_key,
            pubchem_api_key=pubchem_api_key,
            crossref_api_key=crossref_api_key
        )
        logger.info("KPI数据获取模块设置完成")
    
    def setup_preprocessing(self):
        """设置数据预处理模块"""
        logger.info("设置数据预处理模块...")
        self.preprocessor = DataPreprocessor()
        logger.info("数据预处理模块设置完成")
    
    def setup_embedding(self, molecular_embedding_dim: int = 128, knowledge_dim: int = 64):
        """
        设置嵌入模块
        
        Args:
            molecular_embedding_dim (int): 分子嵌入维度
            knowledge_dim (int): 知识向量维度
        """
        logger.info("设置嵌入模块...")
        self.embedder = MolecularEmbedder(embedding_dim=molecular_embedding_dim)
        self.vectorizer = KnowledgeVectorizer(vector_dim=knowledge_dim)
        logger.info("嵌入模块设置完成")
    
    def setup_model(self, 
                   molecular_embedding_dim: int = 128,
                   knowledge_dim: int = 64,
                   hidden_dim: int = 256,
                   num_properties: int = 3):
        """
        设置深度学习模型
        
        Args:
            molecular_embedding_dim (int): 分子嵌入维度
            knowledge_dim (int): 知识向量维度
            hidden_dim (int): 隐藏层维度
            num_properties (int): 属性数量
        """
        logger.info("设置深度学习模型...")
        self.model = MolecularPropertyPredictor(
            molecular_embedding_dim=molecular_embedding_dim,
            knowledge_dim=knowledge_dim,
            hidden_dim=hidden_dim,
            num_properties=num_properties
        )
        self.trainer = ModelTrainer(self.model, learning_rate=self.config.get('learning_rate', 0.001))
        self.evaluator = ModelEvaluator(self.model)
        self.predictor = PropertyPredictor(self.model)
        logger.info("深度学习模型设置完成")
    
    def run_data_acquisition(self, 
                           search_terms: Optional[List[str]] = None,
                           elements: Optional[List[str]] = None,
                           max_materials_per_source: int = 200) -> pd.DataFrame:
        """
        运行KPI数据获取
        
        Args:
            search_terms (List[str], optional): 搜索词列表
            elements (List[str], optional): 元素列表
            max_materials_per_source (int): 每个数据源的最大材料数量
            
        Returns:
            pd.DataFrame: 获取的数据
        """
        logger.info("开始KPI数据获取...")
        
        if self.data_acquirer is None:
            raise ValueError("数据获取模块未设置，请先调用setup_data_acquisition")
        
        # 设置默认搜索词
        if search_terms is None:
            search_terms = ["electrolyte", "ionic liquid", "solvent", "electrolyte solution"]
        
        # 设置默认元素（符合KPI要求）
        if elements is None:
            elements = ["C", "H", "O", "N", "F", "P", "Cl", "Br", "I"]
        
        # 从所有数据源收集数据
        all_data = self.data_acquirer.collect_all_data(
            search_terms=search_terms,
            elements=elements,
            max_materials_per_source=max_materials_per_source
        )
        
        if all_data.empty:
            logger.warning("未找到符合KPI要求的材料，使用示例数据")
            return self._create_sample_data()
        
        # 创建原始数据表
        self.origin_data = self.data_acquirer.create_origin_sheet(all_data)
        
        logger.info(f"KPI数据获取完成，获得 {len(self.origin_data)} 个符合要求的材料")
        return self.origin_data
    
    def run_preprocessing(self) -> pd.DataFrame:
        """
        运行数据预处理
        
        Returns:
            pd.DataFrame: 预处理后的数据
        """
        logger.info("开始数据预处理...")
        
        if self.preprocessor is None:
            raise ValueError("数据预处理模块未设置，请先调用setup_preprocessing")
        
        if self.origin_data is None:
            logger.warning("原始数据为空，使用示例数据")
            self.origin_data = self._create_sample_data()
        
        # 数据清洗
        cleaned_data = self.preprocessor.clean_data(self.origin_data)
        
        # 计算分子描述符
        enhanced_data = self.preprocessor.calculate_molecular_descriptors(cleaned_data)
        
        # 创建组织化数据表
        self.organised_data = self.preprocessor.create_organised_sheet(enhanced_data)
        
        # 统计分析
        stats = self.preprocessor.perform_statistical_analysis(self.organised_data)
        
        # 可视化
        self.preprocessor.visualize_data(self.organised_data, 'data_analysis.png')
        
        logger.info(f"数据预处理完成，组织化数据包含 {len(self.organised_data)} 行")
        return self.organised_data
    
    def run_embedding(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        运行分子嵌入和知识向量化
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: 分子嵌入和知识向量
        """
        logger.info("开始分子嵌入和知识向量化...")
        
        if self.embedder is None or self.vectorizer is None:
            raise ValueError("嵌入模块未设置，请先调用setup_embedding")
        
        if self.organised_data is None:
            raise ValueError("组织化数据为空，请先运行数据预处理")
        
        # 提取SMILES
        smiles_list = self.organised_data['SMILES'].tolist()
        
        # 创建分子嵌入
        self.molecular_embeddings = self.embedder.create_molecular_embeddings(smiles_list)
        
        # 创建知识向量
        self.knowledge_vectors = self.vectorizer.vectorize_knowledge(self.organised_data)
        
        # 可视化嵌入结果
        visualizer = EmbeddingVisualizer()
        visualizer.visualize_embeddings(
            self.molecular_embeddings,
            self.knowledge_vectors,
            save_path='embedding_visualization.png'
        )
        
        logger.info(f"嵌入完成，分子嵌入形状: {self.molecular_embeddings.shape}，知识向量形状: {self.knowledge_vectors.shape}")
        return self.molecular_embeddings, self.knowledge_vectors
    
    def run_training(self, 
                    train_ratio: float = 0.7,
                    val_ratio: float = 0.15,
                    num_epochs: int = 100,
                    batch_size: int = 32) -> Dict:
        """
        运行模型训练
        
        Args:
            train_ratio (float): 训练集比例
            val_ratio (float): 验证集比例
            num_epochs (int): 训练轮数
            batch_size (int): 批次大小
            
        Returns:
            Dict: 训练历史
        """
        logger.info("开始模型训练...")
        
        if self.model is None:
            raise ValueError("模型未设置，请先调用setup_model")
        
        if self.molecular_embeddings is None or self.knowledge_vectors is None:
            raise ValueError("嵌入数据为空，请先运行嵌入步骤")
        
        # 创建目标数据（模拟属性值）
        targets = self._create_target_data(len(self.molecular_embeddings))
        
        # 创建数据集
        dataset = MolecularDataset(
            self.molecular_embeddings,
            self.knowledge_vectors,
            targets
        )
        
        # 划分数据集
        train_size = int(train_ratio * len(dataset))
        val_size = int(val_ratio * len(dataset))
        test_size = len(dataset) - train_size - val_size
        
        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size, test_size]
        )
        
        # 创建数据加载器
        from torch.utils.data import DataLoader
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        # 训练模型
        history = self.trainer.train(
            train_loader, val_loader, 
            num_epochs=num_epochs, 
            patience=20
        )
        
        # 绘制训练历史
        self.trainer.plot_training_history('training_history.png')
        
        # 评估模型
        evaluation_results = self.evaluator.evaluate(test_loader)
        logger.info("模型评估结果:")
        for prop, metrics in evaluation_results.items():
            logger.info(f"{prop}: {metrics}")
        
        # 绘制预测结果
        self.evaluator.plot_predictions(test_loader, 'prediction_evaluation.png')
        
        logger.info("模型训练完成")
        return history
    
    def run_prediction(self, 
                      molecular_embedding: np.ndarray,
                      knowledge_vector: np.ndarray) -> Dict:
        """
        运行属性预测
        
        Args:
            molecular_embedding (np.ndarray): 分子嵌入
            knowledge_vector (np.ndarray): 知识向量
            
        Returns:
            Dict: 预测结果
        """
        logger.info("开始属性预测...")
        
        if self.predictor is None:
            raise ValueError("预测器未设置，请先调用setup_model")
        
        # 进行预测
        predictions = self.predictor.predict_properties(molecular_embedding, knowledge_vector)
        
        # 生成可视化
        visualizer = ResultVisualizer()
        visualizer.plot_property_predictions(predictions, 'property_predictions.png')
        
        # 生成反馈
        feedback_generator = FeedbackGenerator()
        feedback = feedback_generator.generate_feedback(predictions)
        
        # 生成报告
        report = feedback_generator.generate_summary_report(predictions, feedback)
        logger.info("预测报告:")
        logger.info(report)
        
        # 导出结果
        exporter = ResultExporter()
        exporter.export_to_json(predictions, feedback, 'prediction_results.json')
        exporter.export_to_csv(predictions, 'prediction_results.csv')
        
        logger.info("属性预测完成")
        return predictions
    
    def run_full_pipeline(self, 
                         materials_project_api_key: str = None,
                         pubchem_api_key: str = None,
                         crossref_api_key: str = None,
                         search_terms: Optional[List[str]] = None,
                         elements: Optional[List[str]] = None,
                         max_materials_per_source: int = 200) -> Dict:
        """
        运行完整KPI框架流程
        
        Args:
            materials_project_api_key (str, optional): Materials Project API密钥
            pubchem_api_key (str, optional): PubChem API密钥
            crossref_api_key (str, optional): CrossRef API密钥
            search_terms (List[str], optional): 搜索词列表
            elements (List[str], optional): 元素列表
            max_materials_per_source (int): 每个数据源的最大材料数量
            
        Returns:
            Dict: 完整流程结果
        """
        logger.info("开始运行完整KPI框架流程...")
        
        try:
            # 1. 设置所有模块
            self.setup_data_acquisition(
                materials_project_api_key=materials_project_api_key,
                pubchem_api_key=pubchem_api_key,
                crossref_api_key=crossref_api_key
            )
            self.setup_preprocessing()
            self.setup_embedding()
            self.setup_model()
            
            # 2. 数据获取
            origin_data = self.run_data_acquisition(
                search_terms=search_terms,
                elements=elements,
                max_materials_per_source=max_materials_per_source
            )
            
            # 3. 数据预处理
            organised_data = self.run_preprocessing()
            
            # 3.5. MACCS和t-SNE分析
            if not organised_data.empty:
                logger.info("执行MACCS和t-SNE分析...")
                maccs_tsne_results = self.preprocessor.perform_maccs_tsne_analysis(
                    organised_data, 
                    property_column='MP',
                    save_path='kpi_molecular_distribution_analysis.png'
                )
                logger.info("MACCS和t-SNE分析完成")
            else:
                maccs_tsne_results = None
            
            # 4. 分子嵌入和知识向量化
            molecular_embeddings, knowledge_vectors = self.run_embedding()
            
            # 5. 模型训练
            training_history = self.run_training()
            
            # 6. 示例预测
            if len(molecular_embeddings) > 0:
                sample_prediction = self.run_prediction(
                    molecular_embeddings[0], 
                    knowledge_vectors[0]
                )
            else:
                sample_prediction = None
            
            results = {
                'origin_data': origin_data,
                'organised_data': organised_data,
                'maccs_tsne_results': maccs_tsne_results,
                'molecular_embeddings': molecular_embeddings,
                'knowledge_vectors': knowledge_vectors,
                'training_history': training_history,
                'sample_prediction': sample_prediction
            }
            
            logger.info("KPI框架完整流程运行完成")
            return results
            
        except Exception as e:
            logger.error(f"运行完整流程时发生错误: {e}")
            raise
    
    def _create_sample_data(self) -> pd.DataFrame:
        """创建符合KPI框架要求的示例数据"""
        logger.info("创建符合KPI框架要求的示例数据...")
        
        sample_data = {
            'SMILES': ['CCO', 'CC(=O)O', 'c1ccccc1', 'CCN', 'CCOO', 'CC(C)O', 'CCCC', 'c1ccc(cc1)O'],
            'Material_ID': [f'kpi-{i}' for i in range(8)],
            'Formula': ['C2H6O', 'C2H4O2', 'C6H6', 'C2H7N', 'C2H6O2', 'C3H8O', 'C4H10', 'C6H6O'],
            'Molwt': [46.07, 60.05, 78.11, 45.08, 62.07, 60.10, 58.12, 94.11],
            '#Heavy': [3, 4, 6, 3, 4, 4, 4, 7],
            'Elements': ['C,H,O', 'C,H,O', 'C,H', 'C,H,N', 'C,H,O', 'C,H,O', 'C,H', 'C,H,O'],
            'MP': [159.0, 289.0, 278.0, 194.0, 200.0, 185.0, 134.0, 314.0],
            'BP': [351.0, 391.0, 353.0, 239.0, 373.0, 338.0, 272.0, 455.0],
            'FP': [286.0, 327.0, 262.0, 200.0, 300.0, 285.0, 213.0, 350.0],
            'Source': ['sample'] * 8
        }
        
        return pd.DataFrame(sample_data)
    
    def _create_target_data(self, num_samples: int) -> np.ndarray:
        """创建目标数据（模拟属性值）"""
        # 模拟熔点、沸点、闪点数据
        np.random.seed(42)
        targets = np.random.randn(num_samples, 3) * 100 + 300
        return targets

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='KPI框架 - 基于知识的电解质属性预测集成框架')
    parser.add_argument('--materials_project_api_key', type=str, help='Materials Project API密钥')
    parser.add_argument('--pubchem_api_key', type=str, help='PubChem API密钥')
    parser.add_argument('--crossref_api_key', type=str, help='CrossRef API密钥')
    parser.add_argument('--search_terms', nargs='+', help='搜索词列表')
    parser.add_argument('--elements', nargs='+', help='要搜索的元素列表')
    parser.add_argument('--max_materials_per_source', type=int, default=200, help='每个数据源的最大材料数量')
    parser.add_argument('--config', type=str, default='config.json', help='配置文件路径')
    
    args = parser.parse_args()
    
    # 默认配置
    config = {
        'learning_rate': 0.001,
        'batch_size': 32,
        'num_epochs': 100,
        'molecular_embedding_dim': 128,
        'knowledge_dim': 64,
        'hidden_dim': 256
    }
    
    # 创建KPI框架
    kpi_framework = KPIFramework(config)
    
    # 设置API密钥
    materials_project_api_key = args.materials_project_api_key or "your_materials_project_api_key_here"
    pubchem_api_key = args.pubchem_api_key or "your_pubchem_api_key_here"
    crossref_api_key = args.crossref_api_key or "your_crossref_api_key_here"
    
    if materials_project_api_key == "your_materials_project_api_key_here":
        logger.warning("使用默认API密钥，请设置有效的API密钥")
    
    try:
        # 运行完整流程
        results = kpi_framework.run_full_pipeline(
            materials_project_api_key=materials_project_api_key,
            pubchem_api_key=pubchem_api_key,
            crossref_api_key=crossref_api_key,
            search_terms=args.search_terms,
            elements=args.elements,
            max_materials_per_source=args.max_materials_per_source
        )
        
        logger.info("KPI框架运行成功！")
        logger.info(f"处理了 {len(results['origin_data'])} 个材料")
        logger.info(f"生成了 {results['molecular_embeddings'].shape[0]} 个分子嵌入")
        logger.info(f"生成了 {results['knowledge_vectors'].shape[0]} 个知识向量")
        
        if results['sample_prediction']:
            logger.info("示例预测结果:")
            for prop, data in results['sample_prediction'].items():
                if isinstance(data, dict) and 'value' in data:
                    logger.info(f"{data['name']}: {data['value']:.2f} {data['unit']}")
        
    except Exception as e:
        logger.error(f"运行KPI框架时发生错误: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()