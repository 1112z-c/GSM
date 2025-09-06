"""
数据预处理和特征工程模块
Data Preprocessing and Feature Engineering Module
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional
import logging
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MolecularDescriptorCalculator:
    """分子描述符计算器"""
    
    def __init__(self):
        """初始化分子描述符计算器"""
        self.descriptor_names = [
            'num_heavy_atoms', 'num_h_bond_donors', 'num_h_bond_acceptors',
            'num_aromatic_rings', 'molecular_weight', 'logp', 'tpsa',
            'num_rotatable_bonds', 'num_heteroatoms', 'num_carbons',
            'num_nitrogens', 'num_oxygens', 'num_halogens'
        ]
    
    def calculate_smiles_descriptors(self, smiles: str) -> Dict[str, float]:
        """
        从SMILES字符串计算分子描述符
        
        Args:
            smiles (str): SMILES字符串
            
        Returns:
            Dict[str, float]: 分子描述符字典
        """
        try:
            # 简化的分子描述符计算（实际应用中应使用RDKit）
            descriptors = {}
            
            # 基本原子计数
            descriptors['num_heavy_atoms'] = self._count_heavy_atoms(smiles)
            descriptors['num_carbons'] = smiles.count('C')
            descriptors['num_nitrogens'] = smiles.count('N')
            descriptors['num_oxygens'] = smiles.count('O')
            descriptors['num_halogens'] = smiles.count('F') + smiles.count('Cl') + smiles.count('Br') + smiles.count('I')
            
            # 氢键相关
            descriptors['num_h_bond_donors'] = self._count_h_bond_donors(smiles)
            descriptors['num_h_bond_acceptors'] = self._count_h_bond_acceptors(smiles)
            
            # 芳香环计数（简化）
            descriptors['num_aromatic_rings'] = self._count_aromatic_rings(smiles)
            
            # 分子量估算（简化）
            descriptors['molecular_weight'] = self._estimate_molecular_weight(smiles)
            
            # LogP估算（简化）
            descriptors['logp'] = self._estimate_logp(smiles)
            
            # TPSA估算（简化）
            descriptors['tpsa'] = self._estimate_tpsa(smiles)
            
            # 可旋转键计数
            descriptors['num_rotatable_bonds'] = self._count_rotatable_bonds(smiles)
            
            # 杂原子计数
            descriptors['num_heteroatoms'] = self._count_heteroatoms(smiles)
            
            return descriptors
            
        except Exception as e:
            logger.warning(f"计算SMILES {smiles} 描述符时发生错误: {e}")
            return {name: 0.0 for name in self.descriptor_names}
    
    def _count_heavy_atoms(self, smiles: str) -> int:
        """计算重原子数"""
        heavy_atoms = ['C', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I']
        count = 0
        for atom in heavy_atoms:
            count += smiles.count(atom)
        return count
    
    def _count_h_bond_donors(self, smiles: str) -> int:
        """计算氢键供体数"""
        # 简化的氢键供体计数
        donors = ['OH', 'NH', 'SH']
        count = 0
        for donor in donors:
            count += smiles.count(donor)
        return count
    
    def _count_h_bond_acceptors(self, smiles: str) -> int:
        """计算氢键受体数"""
        # 简化的氢键受体计数
        acceptors = ['O', 'N', 'S']
        count = 0
        for acceptor in acceptors:
            count += smiles.count(acceptor)
        return count
    
    def _count_aromatic_rings(self, smiles: str) -> int:
        """计算芳香环数（简化）"""
        # 简化的芳香环计数
        aromatic_patterns = ['c1ccccc1', 'c1ccccc1c', 'c1ccccc1n']
        count = 0
        for pattern in aromatic_patterns:
            count += smiles.count(pattern)
        return max(count, 0)
    
    def _estimate_molecular_weight(self, smiles: str) -> float:
        """估算分子量"""
        # 简化的分子量计算
        atomic_weights = {
            'C': 12.01, 'N': 14.01, 'O': 16.00, 'S': 32.07,
            'P': 30.97, 'F': 19.00, 'Cl': 35.45, 'Br': 79.90, 'I': 126.90
        }
        
        weight = 0.0
        for atom, mass in atomic_weights.items():
            weight += smiles.count(atom) * mass
        
        # 添加氢原子（简化估算）
        weight += smiles.count('H') * 1.01
        
        return weight
    
    def _estimate_logp(self, smiles: str) -> float:
        """估算LogP值"""
        # 简化的LogP计算
        hydrophobic_atoms = ['C', 'H']
        hydrophilic_atoms = ['O', 'N', 'S']
        
        hydrophobic_count = sum(smiles.count(atom) for atom in hydrophobic_atoms)
        hydrophilic_count = sum(smiles.count(atom) for atom in hydrophilic_atoms)
        
        # 简化的LogP估算公式
        logp = hydrophobic_count * 0.1 - hydrophilic_count * 0.2
        return logp
    
    def _estimate_tpsa(self, smiles: str) -> float:
        """估算拓扑极性表面积"""
        # 简化的TPSA计算
        polar_atoms = ['O', 'N', 'S']
        tpsa = sum(smiles.count(atom) * 20.0 for atom in polar_atoms)
        return tpsa
    
    def _count_rotatable_bonds(self, smiles: str) -> int:
        """计算可旋转键数"""
        # 简化的可旋转键计数
        rotatable_patterns = ['-C-', '-N-', '-O-', '-S-']
        count = 0
        for pattern in rotatable_patterns:
            count += smiles.count(pattern)
        return count
    
    def _count_heteroatoms(self, smiles: str) -> int:
        """计算杂原子数"""
        heteroatoms = ['N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I']
        count = 0
        for atom in heteroatoms:
            count += smiles.count(atom)
        return count

class DataPreprocessor:
    """数据预处理器 - 符合KPI框架要求"""
    
    def __init__(self):
        """初始化数据预处理器"""
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='median')
        self.descriptor_calculator = MolecularDescriptorCalculator()
        self.is_fitted = False
        
        # KPI框架数据限制
        self.allowed_elements = {'H', 'C', 'N', 'O', 'F', 'Si', 'P', 'Cl', 'Br', 'I'}
        self.min_molecular_weight = 0
        self.max_molecular_weight = 600
        self.min_heavy_atoms = 0
        self.max_heavy_atoms = 30
        
        # 目标属性
        self.target_properties = ['melting_point', 'boiling_point', 'flash_point']
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        数据清洗
        
        Args:
            df (pd.DataFrame): 原始数据
            
        Returns:
            pd.DataFrame: 清洗后的数据
        """
        logger.info("开始数据清洗...")
        
        # 复制数据
        cleaned_df = df.copy()
        
        # 删除重复行
        initial_rows = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates()
        logger.info(f"删除了 {initial_rows - len(cleaned_df)} 个重复行")
        
        # 删除空SMILES
        cleaned_df = cleaned_df.dropna(subset=['SMILES'])
        logger.info(f"删除了空SMILES行，剩余 {len(cleaned_df)} 行")
        
        # 删除无效的SMILES（长度过短或过长）
        cleaned_df = cleaned_df[
            (cleaned_df['SMILES'].str.len() >= 1) & 
            (cleaned_df['SMILES'].str.len() <= 200)
        ]
        logger.info(f"过滤无效SMILES后，剩余 {len(cleaned_df)} 行")
        
        return cleaned_df
    
    def calculate_molecular_descriptors(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算分子描述符
        
        Args:
            df (pd.DataFrame): 包含SMILES的数据
            
        Returns:
            pd.DataFrame: 添加了分子描述符的数据
        """
        logger.info("开始计算分子描述符...")
        
        # 复制数据
        enhanced_df = df.copy()
        
        # 计算每个SMILES的分子描述符
        descriptors_list = []
        for idx, row in enhanced_df.iterrows():
            smiles = row['SMILES']
            descriptors = self.descriptor_calculator.calculate_smiles_descriptors(smiles)
            descriptors_list.append(descriptors)
        
        # 将描述符添加到DataFrame
        descriptors_df = pd.DataFrame(descriptors_list)
        enhanced_df = pd.concat([enhanced_df, descriptors_df], axis=1)
        
        logger.info(f"成功计算了 {len(descriptors_df.columns)} 个分子描述符")
        
        return enhanced_df
    
    def create_organised_sheet(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        创建组织化数据表（Organised Sheet）
        
        Args:
            df (pd.DataFrame): 包含分子描述符的数据
            
        Returns:
            pd.DataFrame: 组织化数据表
        """
        logger.info("创建组织化数据表...")
        
        # 选择相关列
        organised_columns = ['SMILES', 'Material_ID', 'Formula'] + self.descriptor_calculator.descriptor_names
        
        # 过滤存在的列
        available_columns = [col for col in organised_columns if col in df.columns]
        organised_df = df[available_columns].copy()
        
        # 重命名列以匹配框架要求
        column_mapping = {
            'num_heavy_atoms': '#Heavy',
            'num_h_bond_donors': '#Donor',
            'num_aromatic_rings': '#ArR',
            'logp': 'Avgl',
            'molecular_weight': 'MW',
            'tpsa': 'TPSA',
            'num_rotatable_bonds': '#RotBonds',
            'num_heteroatoms': '#Hetero'
        }
        
        organised_df = organised_df.rename(columns=column_mapping)
        
        # 添加ID列
        organised_df.insert(0, 'ID', range(len(organised_df)))
        
        logger.info(f"组织化数据表包含 {len(organised_df)} 行，{len(organised_df.columns)} 列")
        
        return organised_df
    
    def perform_statistical_analysis(self, df: pd.DataFrame) -> Dict:
        """
        执行统计分析
        
        Args:
            df (pd.DataFrame): 数据表
            
        Returns:
            Dict: 统计分析结果
        """
        logger.info("执行统计分析...")
        
        # 数值列
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        stats = {}
        
        # 基本统计信息
        stats['basic_stats'] = df[numeric_columns].describe()
        
        # 相关性分析
        if len(numeric_columns) > 1:
            stats['correlation_matrix'] = df[numeric_columns].corr()
        
        # 缺失值统计
        stats['missing_values'] = df.isnull().sum()
        
        # 数据分布
        stats['value_counts'] = {}
        for col in df.columns:
            if df[col].dtype == 'object':
                stats['value_counts'][col] = df[col].value_counts().head(10)
        
        logger.info("统计分析完成")
        
        return stats
    
    def visualize_data(self, df: pd.DataFrame, save_path: str = None):
        """
        数据可视化
        
        Args:
            df (pd.DataFrame): 数据表
            save_path (str, optional): 保存路径
        """
        logger.info("生成数据可视化...")
        
        # 设置图形样式
        plt.style.use('default')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('数据可视化分析', fontsize=16)
        
        # 数值列
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_columns) > 0:
            # 1. 数值特征分布
            if len(numeric_columns) >= 4:
                for i, col in enumerate(numeric_columns[:4]):
                    ax = axes[i//2, i%2]
                    df[col].hist(bins=30, ax=ax, alpha=0.7)
                    ax.set_title(f'{col} 分布')
                    ax.set_xlabel(col)
                    ax.set_ylabel('频次')
            else:
                for i, col in enumerate(numeric_columns):
                    ax = axes[i//2, i%2]
                    df[col].hist(bins=30, ax=ax, alpha=0.7)
                    ax.set_title(f'{col} 分布')
                    ax.set_xlabel(col)
                    ax.set_ylabel('频次')
        
        # 2. 相关性热图
        if len(numeric_columns) > 1:
            ax = axes[1, 1]
            corr_matrix = df[numeric_columns].corr()
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)
            ax.set_title('特征相关性热图')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"可视化图表已保存到 {save_path}")
        
        plt.show()
    
    def preprocess_for_ml(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """
        为机器学习预处理数据
        
        Args:
            df (pd.DataFrame): 输入数据
            
        Returns:
            Tuple[pd.DataFrame, Dict]: 预处理后的数据和预处理参数
        """
        logger.info("为机器学习预处理数据...")
        
        # 选择数值特征
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        ml_df = df[numeric_columns].copy()
        
        # 处理缺失值
        ml_df = ml_df.fillna(ml_df.median())
        
        # 标准化
        if not self.is_fitted:
            ml_df_scaled = pd.DataFrame(
                self.scaler.fit_transform(ml_df),
                columns=ml_df.columns,
                index=ml_df.index
            )
            self.is_fitted = True
        else:
            ml_df_scaled = pd.DataFrame(
                self.scaler.transform(ml_df),
                columns=ml_df.columns,
                index=ml_df.index
            )
        
        # 保存预处理参数
        preprocessing_params = {
            'scaler_mean': self.scaler.mean_,
            'scaler_scale': self.scaler.scale_,
            'feature_columns': list(ml_df.columns)
        }
        
        logger.info(f"机器学习预处理完成，特征数: {len(ml_df_scaled.columns)}")
        
        return ml_df_scaled, preprocessing_params

def main():
    """主函数示例"""
    # 创建示例数据
    sample_data = {
        'SMILES': ['CCO', 'CC(=O)O', 'c1ccccc1', 'CCN', 'CCOO'],
        'Material_ID': ['mp-1', 'mp-2', 'mp-3', 'mp-4', 'mp-5'],
        'Formula': ['C2H6O', 'C2H4O2', 'C6H6', 'C2H7N', 'C2H6O2'],
        'Density': [0.789, 1.049, 0.876, 0.682, 1.11],
        'Volume': [58.0, 57.0, 78.0, 45.0, 62.0]
    }
    
    df = pd.DataFrame(sample_data)
    
    # 创建预处理器
    preprocessor = DataPreprocessor()
    
    # 数据清洗
    cleaned_df = preprocessor.clean_data(df)
    print("清洗后的数据:")
    print(cleaned_df)
    
    # 计算分子描述符
    enhanced_df = preprocessor.calculate_molecular_descriptors(cleaned_df)
    print("\n添加分子描述符后的数据:")
    print(enhanced_df.head())
    
    # 创建组织化数据表
    organised_df = preprocessor.create_organised_sheet(enhanced_df)
    print("\n组织化数据表:")
    print(organised_df)
    
    # 统计分析
    stats = preprocessor.perform_statistical_analysis(organised_df)
    print("\n基本统计信息:")
    print(stats['basic_stats'])
    
    # 可视化
    preprocessor.visualize_data(organised_df, 'data_visualization.png')
    
    # 机器学习预处理
    ml_df, params = preprocessor.preprocess_for_ml(organised_df)
    print("\n机器学习预处理后的数据:")
    print(ml_df.head())

if __name__ == "__main__":
    main()