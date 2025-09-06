"""
深度学习模型模块
Deep Learning Model Module
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Tuple, Optional, Union
import logging
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MolecularDataset(Dataset):
    """分子数据集类"""
    
    def __init__(self, molecular_embeddings: np.ndarray, 
                 knowledge_vectors: np.ndarray, 
                 targets: np.ndarray):
        """
        初始化分子数据集
        
        Args:
            molecular_embeddings (np.ndarray): 分子嵌入
            knowledge_vectors (np.ndarray): 知识向量
            targets (np.ndarray): 目标属性
        """
        self.molecular_embeddings = torch.FloatTensor(molecular_embeddings)
        self.knowledge_vectors = torch.FloatTensor(knowledge_vectors)
        self.targets = torch.FloatTensor(targets)
        
    def __len__(self):
        return len(self.molecular_embeddings)
    
    def __getitem__(self, idx):
        return {
            'molecular_embedding': self.molecular_embeddings[idx],
            'knowledge_vector': self.knowledge_vectors[idx],
            'target': self.targets[idx]
        }

class KnowledgeController(nn.Module):
    """知识控制器"""
    
    def __init__(self, knowledge_dim: int, hidden_dim: int = 64):
        """
        初始化知识控制器
        
        Args:
            knowledge_dim (int): 知识向量维度
            hidden_dim (int): 隐藏层维度
        """
        super(KnowledgeController, self).__init__()
        
        self.knowledge_dim = knowledge_dim
        self.hidden_dim = hidden_dim
        
        # 知识处理网络
        self.knowledge_processor = nn.Sequential(
            nn.Linear(knowledge_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # 注意力机制
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            dropout=0.1,
            batch_first=True
        )
        
        # 知识门控
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )
        
        # 知识纯度调节器
        self.purity_regulator = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, knowledge_vector: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            knowledge_vector (torch.Tensor): 知识向量
            
        Returns:
            Dict[str, torch.Tensor]: 处理后的知识信息
        """
        # 处理知识向量
        processed_knowledge = self.knowledge_processor(knowledge_vector)
        
        # 自注意力
        attended_knowledge, attention_weights = self.attention(
            processed_knowledge.unsqueeze(1),
            processed_knowledge.unsqueeze(1),
            processed_knowledge.unsqueeze(1)
        )
        attended_knowledge = attended_knowledge.squeeze(1)
        
        # 知识门控
        gate_weights = self.gate(processed_knowledge)
        gated_knowledge = attended_knowledge * gate_weights
        
        # 知识纯度
        purity = self.purity_regulator(processed_knowledge)
        
        return {
            'processed_knowledge': gated_knowledge,
            'attention_weights': attention_weights,
            'gate_weights': gate_weights,
            'purity': purity
        }

class MolecularPropertyPredictor(nn.Module):
    """分子属性预测器"""
    
    def __init__(self, 
                 molecular_embedding_dim: int,
                 knowledge_dim: int,
                 hidden_dim: int = 128,
                 num_properties: int = 3):
        """
        初始化分子属性预测器
        
        Args:
            molecular_embedding_dim (int): 分子嵌入维度
            knowledge_dim (int): 知识向量维度
            hidden_dim (int): 隐藏层维度
            num_properties (int): 属性数量
        """
        super(MolecularPropertyPredictor, self).__init__()
        
        self.molecular_embedding_dim = molecular_embedding_dim
        self.knowledge_dim = knowledge_dim
        self.hidden_dim = hidden_dim
        self.num_properties = num_properties
        
        # 分子嵌入处理网络
        self.molecular_processor = nn.Sequential(
            nn.Linear(molecular_embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # 知识控制器
        self.knowledge_controller = KnowledgeController(knowledge_dim, hidden_dim)
        
        # 融合网络
        self.fusion_network = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU()
        )
        
        # 属性预测头
        self.property_heads = nn.ModuleList([
            nn.Linear(hidden_dim // 4, 1) for _ in range(num_properties)
        ])
        
        # 属性名称
        self.property_names = ['melting_point', 'boiling_point', 'flash_point']
        
    def forward(self, molecular_embedding: torch.Tensor, 
                knowledge_vector: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            molecular_embedding (torch.Tensor): 分子嵌入
            knowledge_vector (torch.Tensor): 知识向量
            
        Returns:
            Dict[str, torch.Tensor]: 预测结果
        """
        # 处理分子嵌入
        processed_molecular = self.molecular_processor(molecular_embedding)
        
        # 处理知识向量
        knowledge_info = self.knowledge_controller(knowledge_vector)
        processed_knowledge = knowledge_info['processed_knowledge']
        
        # 融合分子和知识信息
        fused_features = torch.cat([processed_molecular, processed_knowledge], dim=1)
        fused_output = self.fusion_network(fused_features)
        
        # 预测各个属性
        predictions = {}
        for i, head in enumerate(self.property_heads):
            property_name = self.property_names[i] if i < len(self.property_names) else f'property_{i}'
            predictions[property_name] = head(fused_output).squeeze(-1)
        
        # 添加知识控制器信息
        predictions.update(knowledge_info)
        
        return predictions

class ModelTrainer:
    """模型训练器"""
    
    def __init__(self, 
                 model: nn.Module,
                 learning_rate: float = 0.001,
                 weight_decay: float = 1e-5):
        """
        初始化模型训练器
        
        Args:
            model (nn.Module): 要训练的模型
            learning_rate (float): 学习率
            weight_decay (float): 权重衰减
        """
        self.model = model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # 优化器
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # 损失函数
        self.criterion = nn.MSELoss()
        
        # 训练历史
        self.train_history = defaultdict(list)
        self.val_history = defaultdict(list)
        
    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        训练一个epoch
        
        Args:
            dataloader (DataLoader): 训练数据加载器
            
        Returns:
            Dict[str, float]: 训练指标
        """
        self.model.train()
        total_loss = 0.0
        property_losses = defaultdict(float)
        num_batches = 0
        
        for batch in dataloader:
            # 移动数据到设备
            molecular_embedding = batch['molecular_embedding'].to(self.device)
            knowledge_vector = batch['knowledge_vector'].to(self.device)
            targets = batch['target'].to(self.device)
            
            # 前向传播
            self.optimizer.zero_grad()
            predictions = self.model(molecular_embedding, knowledge_vector)
            
            # 计算损失
            total_batch_loss = 0.0
            for i, property_name in enumerate(self.model.property_names):
                if i < targets.shape[1]:
                    property_loss = self.criterion(
                        predictions[property_name], 
                        targets[:, i]
                    )
                    total_batch_loss += property_loss
                    property_losses[property_name] += property_loss.item()
            
            # 反向传播
            total_batch_loss.backward()
            self.optimizer.step()
            
            total_loss += total_batch_loss.item()
            num_batches += 1
        
        # 计算平均损失
        avg_loss = total_loss / num_batches
        avg_property_losses = {k: v / num_batches for k, v in property_losses.items()}
        
        return {
            'total_loss': avg_loss,
            **avg_property_losses
        }
    
    def validate_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        验证一个epoch
        
        Args:
            dataloader (DataLoader): 验证数据加载器
            
        Returns:
            Dict[str, float]: 验证指标
        """
        self.model.eval()
        total_loss = 0.0
        property_losses = defaultdict(float)
        num_batches = 0
        
        with torch.no_grad():
            for batch in dataloader:
                # 移动数据到设备
                molecular_embedding = batch['molecular_embedding'].to(self.device)
                knowledge_vector = batch['knowledge_vector'].to(self.device)
                targets = batch['target'].to(self.device)
                
                # 前向传播
                predictions = self.model(molecular_embedding, knowledge_vector)
                
                # 计算损失
                total_batch_loss = 0.0
                for i, property_name in enumerate(self.model.property_names):
                    if i < targets.shape[1]:
                        property_loss = self.criterion(
                            predictions[property_name], 
                            targets[:, i]
                        )
                        total_batch_loss += property_loss
                        property_losses[property_name] += property_loss.item()
                
                total_loss += total_batch_loss.item()
                num_batches += 1
        
        # 计算平均损失
        avg_loss = total_loss / num_batches
        avg_property_losses = {k: v / num_batches for k, v in property_losses.items()}
        
        return {
            'total_loss': avg_loss,
            **avg_property_losses
        }
    
    def train(self, 
              train_loader: DataLoader,
              val_loader: DataLoader,
              num_epochs: int = 100,
              patience: int = 10) -> Dict[str, List[float]]:
        """
        训练模型
        
        Args:
            train_loader (DataLoader): 训练数据加载器
            val_loader (DataLoader): 验证数据加载器
            num_epochs (int): 训练轮数
            patience (int): 早停耐心值
            
        Returns:
            Dict[str, List[float]]: 训练历史
        """
        logger.info(f"开始训练模型，设备: {self.device}")
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(num_epochs):
            # 训练
            train_metrics = self.train_epoch(train_loader)
            for key, value in train_metrics.items():
                self.train_history[key].append(value)
            
            # 验证
            val_metrics = self.validate_epoch(val_loader)
            for key, value in val_metrics.items():
                self.val_history[key].append(value)
            
            # 打印进度
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch + 1}/{num_epochs}")
                logger.info(f"Train Loss: {train_metrics['total_loss']:.4f}")
                logger.info(f"Val Loss: {val_metrics['total_loss']:.4f}")
            
            # 早停检查
            if val_metrics['total_loss'] < best_val_loss:
                best_val_loss = val_metrics['total_loss']
                patience_counter = 0
                # 保存最佳模型
                torch.save(self.model.state_dict(), 'best_model.pth')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"早停于第 {epoch + 1} 轮")
                    break
        
        logger.info("训练完成")
        return dict(self.train_history)
    
    def plot_training_history(self, save_path: Optional[str] = None):
        """绘制训练历史"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('模型训练历史', fontsize=16)
        
        # 总损失
        axes[0, 0].plot(self.train_history['total_loss'], label='训练损失')
        axes[0, 0].plot(self.val_history['total_loss'], label='验证损失')
        axes[0, 0].set_title('总损失')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # 各属性损失
        property_names = [k for k in self.train_history.keys() if k != 'total_loss']
        for i, prop in enumerate(property_names[:3]):  # 最多显示3个属性
            ax = axes[i//2 + 1, i%2] if i < 2 else axes[1, 1]
            ax.plot(self.train_history[prop], label=f'训练 {prop}')
            ax.plot(self.val_history[prop], label=f'验证 {prop}')
            ax.set_title(f'{prop} 损失')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.legend()
            ax.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"训练历史图表已保存到 {save_path}")
        
        plt.show()

class ModelEvaluator:
    """模型评估器"""
    
    def __init__(self, model: nn.Module):
        """
        初始化模型评估器
        
        Args:
            model (nn.Module): 要评估的模型
        """
        self.model = model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
    
    def evaluate(self, dataloader: DataLoader) -> Dict[str, Dict[str, float]]:
        """
        评估模型
        
        Args:
            dataloader (DataLoader): 测试数据加载器
            
        Returns:
            Dict[str, Dict[str, float]]: 评估指标
        """
        logger.info("开始评估模型...")
        
        all_predictions = defaultdict(list)
        all_targets = defaultdict(list)
        
        with torch.no_grad():
            for batch in dataloader:
                # 移动数据到设备
                molecular_embedding = batch['molecular_embedding'].to(self.device)
                knowledge_vector = batch['knowledge_vector'].to(self.device)
                targets = batch['target'].to(self.device)
                
                # 前向传播
                predictions = self.model(molecular_embedding, knowledge_vector)
                
                # 收集预测和目标
                for i, property_name in enumerate(self.model.property_names):
                    if i < targets.shape[1]:
                        all_predictions[property_name].extend(
                            predictions[property_name].cpu().numpy()
                        )
                        all_targets[property_name].extend(
                            targets[:, i].cpu().numpy()
                        )
        
        # 计算评估指标
        evaluation_results = {}
        for property_name in self.model.property_names:
            if property_name in all_predictions:
                pred = np.array(all_predictions[property_name])
                target = np.array(all_targets[property_name])
                
                mse = mean_squared_error(target, pred)
                mae = mean_absolute_error(target, pred)
                r2 = r2_score(target, pred)
                rmse = np.sqrt(mse)
                
                evaluation_results[property_name] = {
                    'mse': mse,
                    'mae': mae,
                    'rmse': rmse,
                    'r2': r2
                }
        
        logger.info("模型评估完成")
        return evaluation_results
    
    def plot_predictions(self, dataloader: DataLoader, save_path: Optional[str] = None):
        """绘制预测结果"""
        logger.info("生成预测结果可视化...")
        
        all_predictions = defaultdict(list)
        all_targets = defaultdict(list)
        
        with torch.no_grad():
            for batch in dataloader:
                molecular_embedding = batch['molecular_embedding'].to(self.device)
                knowledge_vector = batch['knowledge_vector'].to(self.device)
                targets = batch['target'].to(self.device)
                
                predictions = self.model(molecular_embedding, knowledge_vector)
                
                for i, property_name in enumerate(self.model.property_names):
                    if i < targets.shape[1]:
                        all_predictions[property_name].extend(
                            predictions[property_name].cpu().numpy()
                        )
                        all_targets[property_name].extend(
                            targets[:, i].cpu().numpy()
                        )
        
        # 绘制预测vs实际
        num_properties = len(self.model.property_names)
        fig, axes = plt.subplots(1, num_properties, figsize=(5*num_properties, 5))
        if num_properties == 1:
            axes = [axes]
        
        for i, property_name in enumerate(self.model.property_names):
            if property_name in all_predictions:
                pred = np.array(all_predictions[property_name])
                target = np.array(all_targets[property_name])
                
                axes[i].scatter(target, pred, alpha=0.6)
                axes[i].plot([target.min(), target.max()], [target.min(), target.max()], 'r--', lw=2)
                axes[i].set_xlabel('实际值')
                axes[i].set_ylabel('预测值')
                axes[i].set_title(f'{property_name} 预测结果')
                axes[i].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"预测结果图表已保存到 {save_path}")
        
        plt.show()

def main():
    """主函数示例"""
    # 创建示例数据
    np.random.seed(42)
    n_samples = 1000
    
    # 分子嵌入和知识向量
    molecular_embeddings = np.random.randn(n_samples, 64)
    knowledge_vectors = np.random.randn(n_samples, 32)
    
    # 目标属性（模拟熔点、沸点、闪点）
    targets = np.random.randn(n_samples, 3) * 100 + 200
    
    # 创建数据集
    dataset = MolecularDataset(molecular_embeddings, knowledge_vectors, targets)
    
    # 划分数据集
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # 创建模型
    model = MolecularPropertyPredictor(
        molecular_embedding_dim=64,
        knowledge_dim=32,
        hidden_dim=128,
        num_properties=3
    )
    
    # 创建训练器
    trainer = ModelTrainer(model, learning_rate=0.001)
    
    # 训练模型
    history = trainer.train(train_loader, val_loader, num_epochs=50, patience=10)
    
    # 绘制训练历史
    trainer.plot_training_history('training_history.png')
    
    # 评估模型
    evaluator = ModelEvaluator(model)
    evaluation_results = evaluator.evaluate(test_loader)
    
    print("评估结果:")
    for prop, metrics in evaluation_results.items():
        print(f"{prop}: {metrics}")
    
    # 绘制预测结果
    evaluator.plot_predictions(test_loader, 'prediction_results.png')
    
    print("深度学习模型训练和评估完成！")

if __name__ == "__main__":
    main()