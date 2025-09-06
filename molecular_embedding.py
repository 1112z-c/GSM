"""
分子嵌入和知识向量化模块
Molecular Embedding and Knowledge Vectorization Module
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Union
import logging
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MolecularEmbedder:
    """分子嵌入器"""
    
    def __init__(self, embedding_dim: int = 128):
        """
        初始化分子嵌入器
        
        Args:
            embedding_dim (int): 嵌入维度
        """
        self.embedding_dim = embedding_dim
        self.atom_vocab = {}
        self.bond_vocab = {}
        self.is_fitted = False
        
    def build_vocabulary(self, smiles_list: List[str]):
        """
        构建原子和键的词汇表
        
        Args:
            smiles_list (List[str]): SMILES字符串列表
        """
        logger.info("构建分子词汇表...")
        
        atom_counter = Counter()
        bond_counter = Counter()
        
        for smiles in smiles_list:
            # 提取原子
            atoms = self._extract_atoms(smiles)
            atom_counter.update(atoms)
            
            # 提取键
            bonds = self._extract_bonds(smiles)
            bond_counter.update(bonds)
        
        # 创建词汇表
        self.atom_vocab = {atom: idx for idx, atom in enumerate(atom_counter.keys())}
        self.bond_vocab = {bond: idx for idx, bond in enumerate(bond_counter.keys())}
        
        # 添加特殊标记
        self.atom_vocab['<PAD>'] = len(self.atom_vocab)
        self.atom_vocab['<UNK>'] = len(self.atom_vocab)
        self.bond_vocab['<PAD>'] = len(self.bond_vocab)
        self.bond_vocab['<UNK>'] = len(self.bond_vocab)
        
        logger.info(f"原子词汇表大小: {len(self.atom_vocab)}")
        logger.info(f"键词汇表大小: {len(self.bond_vocab)}")
        
    def _extract_atoms(self, smiles: str) -> List[str]:
        """从SMILES提取原子"""
        atoms = []
        i = 0
        while i < len(smiles):
            char = smiles[i]
            if char.isupper():
                # 单字符原子
                if i + 1 < len(smiles) and smiles[i + 1].islower():
                    # 双字符原子
                    atoms.append(smiles[i:i+2])
                    i += 2
                else:
                    atoms.append(char)
                    i += 1
            elif char in ['[', ']', '(', ')', '=', '#', '-', '+', '@']:
                # 跳过特殊字符
                i += 1
            else:
                i += 1
        return atoms
    
    def _extract_bonds(self, smiles: str) -> List[str]:
        """从SMILES提取键类型"""
        bonds = []
        for char in smiles:
            if char in ['-', '=', '#', ':']:
                bonds.append(char)
        return bonds
    
    def smiles_to_sequence(self, smiles: str, max_length: int = 100) -> Tuple[List[int], List[int]]:
        """
        将SMILES转换为原子和键的序列
        
        Args:
            smiles (str): SMILES字符串
            max_length (int): 最大序列长度
            
        Returns:
            Tuple[List[int], List[int]]: 原子序列和键序列
        """
        atoms = self._extract_atoms(smiles)
        bonds = self._extract_bonds(smiles)
        
        # 转换为索引
        atom_seq = [self.atom_vocab.get(atom, self.atom_vocab['<UNK>']) for atom in atoms]
        bond_seq = [self.bond_vocab.get(bond, self.bond_vocab['<UNK>']) for bond in bonds]
        
        # 填充或截断到固定长度
        atom_seq = self._pad_or_truncate(atom_seq, max_length, self.atom_vocab['<PAD>'])
        bond_seq = self._pad_or_truncate(bond_seq, max_length, self.bond_vocab['<PAD>'])
        
        return atom_seq, bond_seq
    
    def _pad_or_truncate(self, sequence: List[int], max_length: int, pad_value: int) -> List[int]:
        """填充或截断序列"""
        if len(sequence) > max_length:
            return sequence[:max_length]
        else:
            return sequence + [pad_value] * (max_length - len(sequence))
    
    def create_molecular_embeddings(self, smiles_list: List[str]) -> np.ndarray:
        """
        创建分子嵌入
        
        Args:
            smiles_list (List[str]): SMILES字符串列表
            
        Returns:
            np.ndarray: 分子嵌入矩阵
        """
        logger.info("创建分子嵌入...")
        
        if not self.is_fitted:
            self.build_vocabulary(smiles_list)
            self.is_fitted = True
        
        embeddings = []
        
        for smiles in smiles_list:
            # 获取原子和键序列
            atom_seq, bond_seq = self.smiles_to_sequence(smiles)
            
            # 创建嵌入向量（简化版本）
            embedding = self._create_embedding_vector(atom_seq, bond_seq)
            embeddings.append(embedding)
        
        embeddings_array = np.array(embeddings)
        logger.info(f"创建了 {embeddings_array.shape[0]} 个分子嵌入，维度: {embeddings_array.shape[1]}")
        
        return embeddings_array
    
    def _create_embedding_vector(self, atom_seq: List[int], bond_seq: List[int]) -> np.ndarray:
        """
        创建单个分子的嵌入向量
        
        Args:
            atom_seq (List[int]): 原子序列
            bond_seq (List[int]): 键序列
            
        Returns:
            np.ndarray: 嵌入向量
        """
        # 简化的嵌入方法：基于原子和键的统计特征
        embedding = np.zeros(self.embedding_dim)
        
        # 原子特征
        atom_features = self._extract_atom_features(atom_seq)
        embedding[:len(atom_features)] = atom_features
        
        # 键特征
        bond_features = self._extract_bond_features(bond_seq)
        start_idx = len(atom_features)
        embedding[start_idx:start_idx + len(bond_features)] = bond_features
        
        # 分子结构特征
        structure_features = self._extract_structure_features(atom_seq, bond_seq)
        start_idx = len(atom_features) + len(bond_features)
        embedding[start_idx:start_idx + len(structure_features)] = structure_features
        
        return embedding
    
    def _extract_atom_features(self, atom_seq: List[int]) -> np.ndarray:
        """提取原子特征"""
        features = []
        
        # 原子类型分布
        unique_atoms, counts = np.unique(atom_seq, return_counts=True)
        atom_dist = np.zeros(len(self.atom_vocab))
        for atom, count in zip(unique_atoms, counts):
            if atom < len(atom_dist):
                atom_dist[atom] = count
        
        # 选择前20个特征
        features.extend(atom_dist[:20])
        
        return np.array(features)
    
    def _extract_bond_features(self, bond_seq: List[int]) -> np.ndarray:
        """提取键特征"""
        features = []
        
        # 键类型分布
        unique_bonds, counts = np.unique(bond_seq, return_counts=True)
        bond_dist = np.zeros(len(self.bond_vocab))
        for bond, count in zip(unique_bonds, counts):
            if bond < len(bond_dist):
                bond_dist[bond] = count
        
        # 选择前10个特征
        features.extend(bond_dist[:10])
        
        return np.array(features)
    
    def _extract_structure_features(self, atom_seq: List[int], bond_seq: List[int]) -> np.ndarray:
        """提取结构特征"""
        features = []
        
        # 序列长度
        features.append(len([x for x in atom_seq if x != self.atom_vocab['<PAD>']]))
        
        # 原子多样性
        unique_atoms = len(set([x for x in atom_seq if x != self.atom_vocab['<PAD>']]))
        features.append(unique_atoms)
        
        # 键密度
        total_bonds = len([x for x in bond_seq if x != self.bond_vocab['<PAD>']])
        features.append(total_bonds)
        
        # 填充剩余维度
        while len(features) < 20:
            features.append(0.0)
        
        return np.array(features[:20])

class KnowledgeVectorizer:
    """知识向量化器"""
    
    def __init__(self, vector_dim: int = 64):
        """
        初始化知识向量化器
        
        Args:
            vector_dim (int): 向量维度
        """
        self.vector_dim = vector_dim
        self.scaler = StandardScaler()
        self.is_fitted = False
        
    def vectorize_knowledge(self, knowledge_data: pd.DataFrame) -> np.ndarray:
        """
        将知识数据向量化
        
        Args:
            knowledge_data (pd.DataFrame): 知识数据
            
        Returns:
            np.ndarray: 向量化的知识
        """
        logger.info("向量化知识数据...")
        
        # 选择数值列
        numeric_columns = knowledge_data.select_dtypes(include=[np.number]).columns
        numeric_data = knowledge_data[numeric_columns].fillna(0)
        
        # 标准化
        if not self.is_fitted:
            vectorized_data = self.scaler.fit_transform(numeric_data)
            self.is_fitted = True
        else:
            vectorized_data = self.scaler.transform(numeric_data)
        
        # 降维到指定维度
        if vectorized_data.shape[1] > self.vector_dim:
            pca = PCA(n_components=self.vector_dim)
            vectorized_data = pca.fit_transform(vectorized_data)
            logger.info(f"使用PCA降维到 {self.vector_dim} 维")
        elif vectorized_data.shape[1] < self.vector_dim:
            # 如果维度不足，进行填充
            padding = np.zeros((vectorized_data.shape[0], self.vector_dim - vectorized_data.shape[1]))
            vectorized_data = np.hstack([vectorized_data, padding])
        
        logger.info(f"知识向量化完成，形状: {vectorized_data.shape}")
        
        return vectorized_data
    
    def extract_knowledge_patterns(self, knowledge_data: pd.DataFrame) -> Dict:
        """
        提取知识模式
        
        Args:
            knowledge_data (pd.DataFrame): 知识数据
            
        Returns:
            Dict: 知识模式字典
        """
        logger.info("提取知识模式...")
        
        patterns = {}
        
        # 数值列
        numeric_columns = knowledge_data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            if col in knowledge_data.columns:
                data = knowledge_data[col].dropna()
                
                # 统计模式
                patterns[col] = {
                    'mean': data.mean(),
                    'std': data.std(),
                    'min': data.min(),
                    'max': data.max(),
                    'median': data.median(),
                    'q25': data.quantile(0.25),
                    'q75': data.quantile(0.75)
                }
        
        # 相关性模式
        if len(numeric_columns) > 1:
            corr_matrix = knowledge_data[numeric_columns].corr()
            patterns['correlations'] = corr_matrix.to_dict()
        
        logger.info(f"提取了 {len(patterns)} 个知识模式")
        
        return patterns

class EmbeddingVisualizer:
    """嵌入可视化器"""
    
    def __init__(self):
        """初始化嵌入可视化器"""
        pass
    
    def visualize_embeddings(self, 
                           molecular_embeddings: np.ndarray, 
                           knowledge_vectors: np.ndarray,
                           labels: Optional[List[str]] = None,
                           save_path: Optional[str] = None):
        """
        可视化嵌入结果
        
        Args:
            molecular_embeddings (np.ndarray): 分子嵌入
            knowledge_vectors (np.ndarray): 知识向量
            labels (List[str], optional): 标签
            save_path (str, optional): 保存路径
        """
        logger.info("生成嵌入可视化...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('分子嵌入和知识向量化可视化', fontsize=16)
        
        # 1. 分子嵌入PCA可视化
        if molecular_embeddings.shape[1] > 2:
            pca = PCA(n_components=2)
            mol_pca = pca.fit_transform(molecular_embeddings)
        else:
            mol_pca = molecular_embeddings
        
        axes[0, 0].scatter(mol_pca[:, 0], mol_pca[:, 1], alpha=0.6, c='blue')
        axes[0, 0].set_title('分子嵌入 PCA 可视化')
        axes[0, 0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
        axes[0, 0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
        
        # 2. 分子嵌入t-SNE可视化
        if molecular_embeddings.shape[1] > 2:
            tsne = TSNE(n_components=2, random_state=42)
            mol_tsne = tsne.fit_transform(molecular_embeddings)
        else:
            mol_tsne = molecular_embeddings
        
        axes[0, 1].scatter(mol_tsne[:, 0], mol_tsne[:, 1], alpha=0.6, c='red')
        axes[0, 1].set_title('分子嵌入 t-SNE 可视化')
        axes[0, 1].set_xlabel('t-SNE 1')
        axes[0, 1].set_ylabel('t-SNE 2')
        
        # 3. 知识向量PCA可视化
        if knowledge_vectors.shape[1] > 2:
            pca_knowledge = PCA(n_components=2)
            know_pca = pca_knowledge.fit_transform(knowledge_vectors)
        else:
            know_pca = knowledge_vectors
        
        axes[1, 0].scatter(know_pca[:, 0], know_pca[:, 1], alpha=0.6, c='green')
        axes[1, 0].set_title('知识向量 PCA 可视化')
        axes[1, 0].set_xlabel(f'PC1 ({pca_knowledge.explained_variance_ratio_[0]:.2%})')
        axes[1, 0].set_ylabel(f'PC2 ({pca_knowledge.explained_variance_ratio_[1]:.2%})')
        
        # 4. 聚类可视化
        if molecular_embeddings.shape[0] > 3:
            kmeans = KMeans(n_clusters=min(5, molecular_embeddings.shape[0]//2), random_state=42)
            clusters = kmeans.fit_predict(molecular_embeddings)
            
            scatter = axes[1, 1].scatter(mol_pca[:, 0], mol_pca[:, 1], c=clusters, alpha=0.6, cmap='viridis')
            axes[1, 1].set_title('分子嵌入聚类可视化')
            axes[1, 1].set_xlabel('PC1')
            axes[1, 1].set_ylabel('PC2')
            plt.colorbar(scatter, ax=axes[1, 1])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"可视化图表已保存到 {save_path}")
        
        plt.show()

def main():
    """主函数示例"""
    # 创建示例数据
    sample_smiles = ['CCO', 'CC(=O)O', 'c1ccccc1', 'CCN', 'CCOO', 'CC(C)O', 'CCCC', 'c1ccc(cc1)O']
    sample_knowledge = pd.DataFrame({
        'MW': [46.07, 60.05, 78.11, 45.08, 62.07, 60.10, 58.12, 94.11],
        'LogP': [0.31, -0.17, 2.13, 0.16, 0.31, 0.16, 1.97, 1.46],
        'TPSA': [20.23, 37.30, 0.00, 26.02, 40.46, 20.23, 0.00, 20.23],
        'NumRotBonds': [0, 0, 0, 0, 1, 0, 1, 0]
    })
    
    # 创建分子嵌入器
    embedder = MolecularEmbedder(embedding_dim=64)
    
    # 创建分子嵌入
    molecular_embeddings = embedder.create_molecular_embeddings(sample_smiles)
    print(f"分子嵌入形状: {molecular_embeddings.shape}")
    
    # 创建知识向量化器
    vectorizer = KnowledgeVectorizer(vector_dim=32)
    
    # 向量化知识
    knowledge_vectors = vectorizer.vectorize_knowledge(sample_knowledge)
    print(f"知识向量形状: {knowledge_vectors.shape}")
    
    # 提取知识模式
    patterns = vectorizer.extract_knowledge_patterns(sample_knowledge)
    print(f"提取了 {len(patterns)} 个知识模式")
    
    # 可视化
    visualizer = EmbeddingVisualizer()
    visualizer.visualize_embeddings(
        molecular_embeddings, 
        knowledge_vectors,
        save_path='embedding_visualization.png'
    )
    
    print("分子嵌入和知识向量化完成！")

if __name__ == "__main__":
    main()