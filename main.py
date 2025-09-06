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
from data_acquisition import MaterialsProjectDataAcquisition
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
    
    def setup_data_acquisition(self, api_key: str):
        """
        设置数据获取模块
        
        Args:
            api_key (str): Materials Project API密钥
        """
        logger.info("设置数据获取模块...")
        self.data_acquirer = MaterialsProjectDataAcquisition(api_key)
        logger.info("数据获取模块设置完成")
    
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
                           elements: Optional[List[str]] = None,
                           formula: Optional[str] = None,
                           num_materials: int = 100) -> pd.DataFrame:
        """
        运行数据获取
        
        Args:
            elements (List[str], optional): 元素列表
            formula (str, optional): 化学式
            num_materials (int): 材料数量
            
        Returns:
            pd.DataFrame: 获取的数据
        """
        logger.info("开始数据获取...")
        
        if self.data_acquirer is None:
            raise ValueError("数据获取模块未设置，请先调用setup_data_acquisition")
        
        # 搜索材料
        materials_df = self.data_acquirer.search_materials(
            elements=elements,
            formula=formula,
            num_chunks=num_materials // 100 + 1
        )
        
        if materials_df.empty:
            logger.warning("未找到符合条件的材料，使用示例数据")
            return self._create_sample_data()
        
        # 获取详细属性
        material_ids = materials_df["material_id"].tolist()[:num_materials]
        detailed_df = self.data_acquirer.get_material_details(material_ids)
        
        # 创建原始数据表
        self.origin_data = self.data_acquirer.create_origin_sheet(detailed_df)
        
        logger.info(f"数据获取完成，获得 {len(self.origin_data)} 个材料")
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
                         api_key: str,
                         elements: Optional[List[str]] = None,
                         formula: Optional[str] = None,
                         num_materials: int = 100) -> Dict:
        """
        运行完整流程
        
        Args:
            api_key (str): Materials Project API密钥
            elements (List[str], optional): 元素列表
            formula (str, optional): 化学式
            num_materials (int): 材料数量
            
        Returns:
            Dict: 完整流程结果
        """
        logger.info("开始运行完整KPI框架流程...")
        
        try:
            # 1. 设置所有模块
            self.setup_data_acquisition(api_key)
            self.setup_preprocessing()
            self.setup_embedding()
            self.setup_model()
            
            # 2. 数据获取
            origin_data = self.run_data_acquisition(elements, formula, num_materials)
            
            # 3. 数据预处理
            organised_data = self.run_preprocessing()
            
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
        """创建示例数据"""
        logger.info("创建示例数据...")
        
        sample_data = {
            'SMILES': ['CCO', 'CC(=O)O', 'c1ccccc1', 'CCN', 'CCOO', 'CC(C)O', 'CCCC', 'c1ccc(cc1)O'],
            'Material_ID': [f'mp-{i}' for i in range(8)],
            'Formula': ['C2H6O', 'C2H4O2', 'C6H6', 'C2H7N', 'C2H6O2', 'C3H8O', 'C4H10', 'C6H6O'],
            'Density': [0.789, 1.049, 0.876, 0.682, 1.11, 0.785, 0.626, 0.949],
            'Volume': [58.0, 57.0, 78.0, 45.0, 62.0, 60.0, 74.0, 94.0],
            'NSites': [9, 8, 12, 8, 10, 10, 14, 13],
            'Elements': ['C,H,O', 'C,H,O', 'C,H', 'C,H,N', 'C,H,O', 'C,H,O', 'C,H', 'C,H,O'],
            'Formation_Energy': [-0.5, -0.3, -0.8, -0.4, -0.6, -0.7, -1.0, -0.9],
            'Band_Gap': [2.5, 3.2, 1.8, 2.8, 3.0, 2.6, 1.5, 2.9],
            'Is_Stable': [True, True, True, True, True, True, True, True]
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
    parser.add_argument('--api_key', type=str, help='Materials Project API密钥')
    parser.add_argument('--elements', nargs='+', help='要搜索的元素列表')
    parser.add_argument('--formula', type=str, help='化学式过滤')
    parser.add_argument('--num_materials', type=int, default=100, help='材料数量')
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
    api_key = args.api_key or "your_materials_project_api_key_here"
    
    if api_key == "your_materials_project_api_key_here":
        logger.warning("使用默认API密钥，请设置有效的Materials Project API密钥")
    
    try:
        # 运行完整流程
        results = kpi_framework.run_full_pipeline(
            api_key=api_key,
            elements=args.elements,
            formula=args.formula,
            num_materials=args.num_materials
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