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
        数据清洗 - 符合KPI框架要求
        
        Args:
            df (pd.DataFrame): 原始数据
            
        Returns:
            pd.DataFrame: 清洗后的数据
        """
        logger.info("开始KPI框架数据清洗...")
        
        # 复制数据
        cleaned_df = df.copy()
        initial_rows = len(cleaned_df)
        
        # 1. 删除重复行
        cleaned_df = cleaned_df.drop_duplicates()
        logger.info(f"删除了 {initial_rows - len(cleaned_df)} 个重复行")
        
        # 2. 删除空SMILES
        cleaned_df = cleaned_df.dropna(subset=['SMILES'])
        logger.info(f"删除了空SMILES行，剩余 {len(cleaned_df)} 行")
        
        # 3. 删除无效的SMILES（长度过短或过长）
        cleaned_df = cleaned_df[
            (cleaned_df['SMILES'].str.len() >= 1) & 
            (cleaned_df['SMILES'].str.len() <= 200)
        ]
        logger.info(f"过滤无效SMILES后，剩余 {len(cleaned_df)} 行")
        
        # 4. KPI框架特定过滤条件
        cleaned_df = self._apply_kpi_filters(cleaned_df)
        
        return cleaned_df
    
    def _apply_kpi_filters(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        应用KPI框架特定的过滤条件
        
        Args:
            df (pd.DataFrame): 输入数据
            
        Returns:
            pd.DataFrame: 过滤后的数据
        """
        logger.info("应用KPI框架过滤条件...")
        
        initial_rows = len(df)
        
        # 1. 元素限制：仅允许H, C, N, O, F, Si, P, Cl, Br, I
        df = self._filter_by_elements(df)
        logger.info(f"元素过滤后，剩余 {len(df)} 行")
        
        # 2. 分子量限制：0-600
        df = self._filter_by_molecular_weight(df)
        logger.info(f"分子量过滤后，剩余 {len(df)} 行")
        
        # 3. 重原子数限制：0-30
        df = self._filter_by_heavy_atoms(df)
        logger.info(f"重原子数过滤后，剩余 {len(df)} 行")
        
        # 4. 确保目标属性列存在
        df = self._ensure_target_properties(df)
        
        logger.info(f"KPI过滤完成，从 {initial_rows} 行减少到 {len(df)} 行")
        return df
    
    def _filter_by_elements(self, df: pd.DataFrame) -> pd.DataFrame:
        """按元素限制过滤数据"""
        if 'Elements' not in df.columns:
            logger.warning("Elements列不存在，跳过元素过滤")
            return df
        
        def check_elements(elements_str):
            if pd.isna(elements_str):
                return False
            
            elements = [e.strip() for e in str(elements_str).split(',')]
            return all(elem in self.allowed_elements for elem in elements)
        
        mask = df['Elements'].apply(check_elements)
        filtered_df = df[mask].copy()
        
        logger.info(f"元素过滤：删除了 {len(df) - len(filtered_df)} 个包含不允许元素的分子")
        return filtered_df
    
    def _filter_by_molecular_weight(self, df: pd.DataFrame) -> pd.DataFrame:
        """按分子量限制过滤数据"""
        if 'Molwt' not in df.columns:
            logger.warning("Molwt列不存在，跳过分子量过滤")
            return df
        
        # 计算分子量（如果不存在）
        if df['Molwt'].isna().all():
            df['Molwt'] = df['SMILES'].apply(self._calculate_molecular_weight_from_smiles)
        
        mask = (df['Molwt'] >= self.min_molecular_weight) & (df['Molwt'] <= self.max_molecular_weight)
        filtered_df = df[mask].copy()
        
        logger.info(f"分子量过滤：删除了 {len(df) - len(filtered_df)} 个分子量超出范围的分子")
        logger.info(f"分子量范围：{filtered_df['Molwt'].min():.2f} - {filtered_df['Molwt'].max():.2f}")
        return filtered_df
    
    def _filter_by_heavy_atoms(self, df: pd.DataFrame) -> pd.DataFrame:
        """按重原子数限制过滤数据"""
        if '#Heavy' not in df.columns:
            logger.warning("#Heavy列不存在，跳过重原子数过滤")
            return df
        
        # 计算重原子数（如果不存在）
        if df['#Heavy'].isna().all():
            df['#Heavy'] = df['SMILES'].apply(self._count_heavy_atoms_from_smiles)
        
        mask = (df['#Heavy'] >= self.min_heavy_atoms) & (df['#Heavy'] <= self.max_heavy_atoms)
        filtered_df = df[mask].copy()
        
        logger.info(f"重原子数过滤：删除了 {len(df) - len(filtered_df)} 个重原子数超出范围的分子")
        logger.info(f"重原子数范围：{filtered_df['#Heavy'].min()} - {filtered_df['#Heavy'].max()}")
        return filtered_df
    
    def _ensure_target_properties(self, df: pd.DataFrame) -> pd.DataFrame:
        """确保目标属性列存在"""
        for prop in self.target_properties:
            prop_col = prop.upper()
            if prop_col not in df.columns:
                df[prop_col] = None
                logger.info(f"添加目标属性列：{prop_col}")
        
        return df
    
    def _calculate_molecular_weight_from_smiles(self, smiles: str) -> float:
        """从SMILES计算分子量"""
        try:
            atomic_weights = {
                'H': 1.008, 'C': 12.011, 'N': 14.007, 'O': 15.999,
                'F': 18.998, 'Si': 28.085, 'P': 30.974,
                'Cl': 35.453, 'Br': 79.904, 'I': 126.904, 'S': 32.065
            }
            
            total_weight = 0.0
            i = 0
            while i < len(smiles):
                if smiles[i].isupper():
                    # 原子符号
                    atom = smiles[i]
                    if i + 1 < len(smiles) and smiles[i + 1].islower():
                        atom += smiles[i + 1]
                        i += 1
                    
                    # 原子数量
                    count = 1
                    if i + 1 < len(smiles) and smiles[i + 1].isdigit():
                        count_str = ""
                        i += 1
                        while i < len(smiles) and smiles[i].isdigit():
                            count_str += smiles[i]
                            i += 1
                        count = int(count_str) if count_str else 1
                        i -= 1
                    
                    if atom in atomic_weights:
                        total_weight += atomic_weights[atom] * count
                
                i += 1
            
            return total_weight
            
        except Exception:
            return 0.0
    
    def _count_heavy_atoms_from_smiles(self, smiles: str) -> int:
        """从SMILES计算重原子数"""
        heavy_atoms = {'C', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I', 'Si'}
        count = 0
        
        i = 0
        while i < len(smiles):
            if smiles[i].isupper():
                atom = smiles[i]
                if i + 1 < len(smiles) and smiles[i + 1].islower():
                    atom += smiles[i + 1]
                    i += 1
                
                if atom in heavy_atoms:
                    # 获取原子数量
                    atom_count = 1
                    if i + 1 < len(smiles) and smiles[i + 1].isdigit():
                        count_str = ""
                        i += 1
                        while i < len(smiles) and smiles[i].isdigit():
                            count_str += smiles[i]
                            i += 1
                        atom_count = int(count_str) if count_str else 1
                        i -= 1
                    
                    count += atom_count
            
            i += 1
        
        return count
    
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
        创建组织化数据表（Organised Sheet）- 符合KPI框架要求
        
        Args:
            df (pd.DataFrame): 包含分子描述符的数据
            
        Returns:
            pd.DataFrame: 组织化数据表
        """
        logger.info("创建KPI框架组织化数据表...")
        
        # 确保数据符合KPI要求
        df = self._apply_kpi_filters(df)
        
        # 选择KPI框架要求的列
        kpi_columns = [
            'SMILES', 'Material_ID', 'Formula', 'Molwt', '#Heavy', 'Elements',
            'MELTING_POINT', 'BOILING_POINT', 'FLASH_POINT'
        ]
        
        # 添加分子描述符列
        descriptor_columns = [
            'num_h_bond_donors', 'num_h_bond_acceptors', 'num_aromatic_rings',
            'logp', 'tpsa', 'num_rotatable_bonds', 'num_heteroatoms'
        ]
        
        all_columns = kpi_columns + descriptor_columns
        
        # 过滤存在的列
        available_columns = [col for col in all_columns if col in df.columns]
        organised_df = df[available_columns].copy()
        
        # 重命名描述符列以匹配KPI框架
        column_mapping = {
            'num_h_bond_donors': '#Donor',
            'num_h_bond_acceptors': '#Acceptor', 
            'num_aromatic_rings': '#ArR',
            'logp': 'LogP',
            'tpsa': 'TPSA',
            'num_rotatable_bonds': '#RotBonds',
            'num_heteroatoms': '#Hetero'
        }
        
        organised_df = organised_df.rename(columns=column_mapping)
        
        # 添加ID列
        organised_df.insert(0, 'ID', range(len(organised_df)))
        
        # 确保目标属性列存在并重命名
        property_mapping = {
            'MELTING_POINT': 'MP',
            'BOILING_POINT': 'BP', 
            'FLASH_POINT': 'FP'
        }
        
        for old_name, new_name in property_mapping.items():
            if old_name in organised_df.columns:
                organised_df = organised_df.rename(columns={old_name: new_name})
            elif new_name not in organised_df.columns:
                organised_df[new_name] = None
        
        # 重新排列列顺序以符合KPI框架
        final_columns = ['ID', 'SMILES', 'Molwt', '#Heavy', 'MP', 'BP', 'FP']
        descriptor_cols = [col for col in organised_df.columns if col not in final_columns and col != 'Material_ID' and col != 'Formula' and col != 'Elements']
        final_columns.extend(descriptor_cols)
        
        # 只保留存在的列
        final_columns = [col for col in final_columns if col in organised_df.columns]
        organised_df = organised_df[final_columns]
        
        logger.info(f"KPI组织化数据表包含 {len(organised_df)} 行，{len(organised_df.columns)} 列")
        logger.info(f"目标属性分布：")
        for prop in ['MP', 'BP', 'FP']:
            if prop in organised_df.columns:
                non_null_count = organised_df[prop].notna().sum()
                logger.info(f"  {prop}: {non_null_count} 个有效值")
        
        return organised_df
    
    def perform_statistical_analysis(self, df: pd.DataFrame) -> Dict:
        """
        执行KPI框架统计分析
        
        Args:
            df (pd.DataFrame): 数据表
            
        Returns:
            Dict: 统计分析结果
        """
        logger.info("执行KPI框架统计分析...")
        
        # 数值列
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        stats = {}
        
        # 1. 基本统计信息
        stats['basic_stats'] = df[numeric_columns].describe()
        
        # 2. KPI框架特定统计
        stats['kpi_stats'] = self._calculate_kpi_statistics(df)
        
        # 3. 目标属性统计
        stats['property_stats'] = self._calculate_property_statistics(df)
        
        # 4. 元素分布统计
        stats['element_distribution'] = self._calculate_element_distribution(df)
        
        # 5. 分子量分布统计
        stats['molecular_weight_distribution'] = self._calculate_molecular_weight_distribution(df)
        
        # 6. 重原子数分布统计
        stats['heavy_atom_distribution'] = self._calculate_heavy_atom_distribution(df)
        
        # 7. 相关性分析
        if len(numeric_columns) > 1:
            stats['correlation_matrix'] = df[numeric_columns].corr()
        
        # 8. 缺失值统计
        stats['missing_values'] = df.isnull().sum()
        
        # 9. 数据质量评估
        stats['data_quality'] = self._assess_data_quality(df)
        
        logger.info("KPI框架统计分析完成")
        
        return stats
    
    def _calculate_kpi_statistics(self, df: pd.DataFrame) -> Dict:
        """计算KPI框架特定统计信息"""
        kpi_stats = {}
        
        # 数据量统计
        kpi_stats['total_molecules'] = len(df)
        
        # 分子量统计
        if 'Molwt' in df.columns:
            molwt_data = df['Molwt'].dropna()
            kpi_stats['molecular_weight'] = {
                'count': len(molwt_data),
                'mean': molwt_data.mean(),
                'std': molwt_data.std(),
                'min': molwt_data.min(),
                'max': molwt_data.max(),
                'within_range': ((molwt_data >= self.min_molecular_weight) & 
                               (molwt_data <= self.max_molecular_weight)).sum()
            }
        
        # 重原子数统计
        if '#Heavy' in df.columns:
            heavy_data = df['#Heavy'].dropna()
            kpi_stats['heavy_atoms'] = {
                'count': len(heavy_data),
                'mean': heavy_data.mean(),
                'std': heavy_data.std(),
                'min': heavy_data.min(),
                'max': heavy_data.max(),
                'within_range': ((heavy_data >= self.min_heavy_atoms) & 
                               (heavy_data <= self.max_heavy_atoms)).sum()
            }
        
        # 元素统计
        if 'Elements' in df.columns:
            all_elements = []
            for elements_str in df['Elements'].dropna():
                elements = [e.strip() for e in str(elements_str).split(',')]
                all_elements.extend(elements)
            
            element_counts = Counter(all_elements)
            kpi_stats['element_counts'] = dict(element_counts)
            kpi_stats['unique_elements'] = len(element_counts)
            kpi_stats['allowed_elements_only'] = all(elem in self.allowed_elements for elem in element_counts.keys())
        
        return kpi_stats
    
    def _calculate_property_statistics(self, df: pd.DataFrame) -> Dict:
        """计算目标属性统计信息"""
        property_stats = {}
        
        for prop in ['MP', 'BP', 'FP']:
            if prop in df.columns:
                prop_data = df[prop].dropna()
                if len(prop_data) > 0:
                    property_stats[prop] = {
                        'count': len(prop_data),
                        'mean': prop_data.mean(),
                        'std': prop_data.std(),
                        'min': prop_data.min(),
                        'max': prop_data.max(),
                        'median': prop_data.median(),
                        'q25': prop_data.quantile(0.25),
                        'q75': prop_data.quantile(0.75)
                    }
                else:
                    property_stats[prop] = {'count': 0, 'message': 'No data available'}
        
        return property_stats
    
    def _calculate_element_distribution(self, df: pd.DataFrame) -> Dict:
        """计算元素分布"""
        if 'Elements' not in df.columns:
            return {}
        
        element_dist = {}
        for elements_str in df['Elements'].dropna():
            elements = [e.strip() for e in str(elements_str).split(',')]
            for element in elements:
                element_dist[element] = element_dist.get(element, 0) + 1
        
        return element_dist
    
    def _calculate_molecular_weight_distribution(self, df: pd.DataFrame) -> Dict:
        """计算分子量分布"""
        if 'Molwt' not in df.columns:
            return {}
        
        molwt_data = df['Molwt'].dropna()
        if len(molwt_data) == 0:
            return {}
        
        # 分区间统计
        bins = [0, 100, 200, 300, 400, 500, 600]
        hist, bin_edges = np.histogram(molwt_data, bins=bins)
        
        distribution = {}
        for i in range(len(hist)):
            range_str = f"{bin_edges[i]}-{bin_edges[i+1]}"
            distribution[range_str] = int(hist[i])
        
        return distribution
    
    def _calculate_heavy_atom_distribution(self, df: pd.DataFrame) -> Dict:
        """计算重原子数分布"""
        if '#Heavy' not in df.columns:
            return {}
        
        heavy_data = df['#Heavy'].dropna()
        if len(heavy_data) == 0:
            return {}
        
        # 分区间统计
        bins = [0, 5, 10, 15, 20, 25, 30]
        hist, bin_edges = np.histogram(heavy_data, bins=bins)
        
        distribution = {}
        for i in range(len(hist)):
            range_str = f"{bin_edges[i]}-{bin_edges[i+1]}"
            distribution[range_str] = int(hist[i])
        
        return distribution
    
    def _assess_data_quality(self, df: pd.DataFrame) -> Dict:
        """评估数据质量"""
        quality = {}
        
        # 完整性评估
        total_cells = len(df) * len(df.columns)
        missing_cells = df.isnull().sum().sum()
        quality['completeness'] = {
            'total_cells': total_cells,
            'missing_cells': missing_cells,
            'completeness_rate': (total_cells - missing_cells) / total_cells
        }
        
        # 目标属性完整性
        property_completeness = {}
        for prop in ['MP', 'BP', 'FP']:
            if prop in df.columns:
                non_null_count = df[prop].notna().sum()
                property_completeness[prop] = {
                    'count': non_null_count,
                    'rate': non_null_count / len(df)
                }
        quality['property_completeness'] = property_completeness
        
        # 数据一致性检查
        quality['consistency'] = {
            'molecular_weight_in_range': self._check_molecular_weight_consistency(df),
            'heavy_atoms_in_range': self._check_heavy_atom_consistency(df),
            'elements_allowed': self._check_element_consistency(df)
        }
        
        return quality
    
    def _check_molecular_weight_consistency(self, df: pd.DataFrame) -> Dict:
        """检查分子量一致性"""
        if 'Molwt' not in df.columns:
            return {'status': 'no_data'}
        
        molwt_data = df['Molwt'].dropna()
        in_range = ((molwt_data >= self.min_molecular_weight) & 
                   (molwt_data <= self.max_molecular_weight)).sum()
        
        return {
            'status': 'good' if in_range == len(molwt_data) else 'warning',
            'in_range': int(in_range),
            'total': len(molwt_data),
            'rate': in_range / len(molwt_data) if len(molwt_data) > 0 else 0
        }
    
    def _check_heavy_atom_consistency(self, df: pd.DataFrame) -> Dict:
        """检查重原子数一致性"""
        if '#Heavy' not in df.columns:
            return {'status': 'no_data'}
        
        heavy_data = df['#Heavy'].dropna()
        in_range = ((heavy_data >= self.min_heavy_atoms) & 
                   (heavy_data <= self.max_heavy_atoms)).sum()
        
        return {
            'status': 'good' if in_range == len(heavy_data) else 'warning',
            'in_range': int(in_range),
            'total': len(heavy_data),
            'rate': in_range / len(heavy_data) if len(heavy_data) > 0 else 0
        }
    
    def _check_element_consistency(self, df: pd.DataFrame) -> Dict:
        """检查元素一致性"""
        if 'Elements' not in df.columns:
            return {'status': 'no_data'}
        
        all_elements = set()
        for elements_str in df['Elements'].dropna():
            elements = [e.strip() for e in str(elements_str).split(',')]
            all_elements.update(elements)
        
        allowed_elements = all_elements.intersection(self.allowed_elements)
        disallowed_elements = all_elements - self.allowed_elements
        
        return {
            'status': 'good' if len(disallowed_elements) == 0 else 'warning',
            'allowed_elements': list(allowed_elements),
            'disallowed_elements': list(disallowed_elements),
            'total_unique_elements': len(all_elements)
        }
    
    def visualize_data(self, df: pd.DataFrame, save_path: str = None):
        """
        KPI框架数据可视化
        
        Args:
            df (pd.DataFrame): 数据表
            save_path (str, optional): 保存路径
        """
        logger.info("生成KPI框架数据可视化...")
        
        # 设置图形样式
        plt.style.use('default')
        fig, axes = plt.subplots(3, 3, figsize=(18, 15))
        fig.suptitle('KPI框架数据可视化分析', fontsize=16, fontweight='bold')
        
        # 1. 分子量分布
        if 'Molwt' in df.columns:
            ax = axes[0, 0]
            molwt_data = df['Molwt'].dropna()
            ax.hist(molwt_data, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
            ax.axvline(self.min_molecular_weight, color='red', linestyle='--', label=f'Min: {self.min_molecular_weight}')
            ax.axvline(self.max_molecular_weight, color='red', linestyle='--', label=f'Max: {self.max_molecular_weight}')
            ax.set_title('分子量分布 (Molwt)')
            ax.set_xlabel('分子量 (g/mol)')
            ax.set_ylabel('频次')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # 2. 重原子数分布
        if '#Heavy' in df.columns:
            ax = axes[0, 1]
            heavy_data = df['#Heavy'].dropna()
            ax.hist(heavy_data, bins=range(0, 31), alpha=0.7, color='lightgreen', edgecolor='black')
            ax.axvline(self.min_heavy_atoms, color='red', linestyle='--', label=f'Min: {self.min_heavy_atoms}')
            ax.axvline(self.max_heavy_atoms, color='red', linestyle='--', label=f'Max: {self.max_heavy_atoms}')
            ax.set_title('重原子数分布 (#Heavy)')
            ax.set_xlabel('重原子数')
            ax.set_ylabel('频次')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # 3. 元素分布
        if 'Elements' in df.columns:
            ax = axes[0, 2]
            all_elements = []
            for elements_str in df['Elements'].dropna():
                elements = [e.strip() for e in str(elements_str).split(',')]
                all_elements.extend(elements)
            
            element_counts = Counter(all_elements)
            elements = list(element_counts.keys())
            counts = list(element_counts.values())
            
            bars = ax.bar(elements, counts, alpha=0.7, color='orange', edgecolor='black')
            ax.set_title('元素分布')
            ax.set_xlabel('元素')
            ax.set_ylabel('出现次数')
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3)
            
            # 添加数值标签
            for bar, count in zip(bars, counts):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                       str(count), ha='center', va='bottom')
        
        # 4. 目标属性分布 - 熔点
        if 'MP' in df.columns:
            ax = axes[1, 0]
            mp_data = df['MP'].dropna()
            if len(mp_data) > 0:
                ax.hist(mp_data, bins=20, alpha=0.7, color='red', edgecolor='black')
                ax.set_title('熔点分布 (MP)')
                ax.set_xlabel('熔点 (K)')
                ax.set_ylabel('频次')
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, '无数据', ha='center', va='center', transform=ax.transAxes)
                ax.set_title('熔点分布 (MP) - 无数据')
        
        # 5. 目标属性分布 - 沸点
        if 'BP' in df.columns:
            ax = axes[1, 1]
            bp_data = df['BP'].dropna()
            if len(bp_data) > 0:
                ax.hist(bp_data, bins=20, alpha=0.7, color='blue', edgecolor='black')
                ax.set_title('沸点分布 (BP)')
                ax.set_xlabel('沸点 (K)')
                ax.set_ylabel('频次')
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, '无数据', ha='center', va='center', transform=ax.transAxes)
                ax.set_title('沸点分布 (BP) - 无数据')
        
        # 6. 目标属性分布 - 闪点
        if 'FP' in df.columns:
            ax = axes[1, 2]
            fp_data = df['FP'].dropna()
            if len(fp_data) > 0:
                ax.hist(fp_data, bins=20, alpha=0.7, color='green', edgecolor='black')
                ax.set_title('闪点分布 (FP)')
                ax.set_xlabel('闪点 (K)')
                ax.set_ylabel('频次')
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, '无数据', ha='center', va='center', transform=ax.transAxes)
                ax.set_title('闪点分布 (FP) - 无数据')
        
        # 7. 分子量 vs 重原子数散点图
        if 'Molwt' in df.columns and '#Heavy' in df.columns:
            ax = axes[2, 0]
            molwt_data = df['Molwt'].dropna()
            heavy_data = df['#Heavy'].dropna()
            if len(molwt_data) > 0 and len(heavy_data) > 0:
                ax.scatter(heavy_data, molwt_data, alpha=0.6, color='purple')
                ax.set_xlabel('重原子数 (#Heavy)')
                ax.set_ylabel('分子量 (Molwt)')
                ax.set_title('分子量 vs 重原子数')
                ax.grid(True, alpha=0.3)
                
                # 添加趋势线
                z = np.polyfit(heavy_data, molwt_data, 1)
                p = np.poly1d(z)
                ax.plot(heavy_data, p(heavy_data), "r--", alpha=0.8)
        
        # 8. 数据质量概览
        ax = axes[2, 1]
        quality_metrics = []
        quality_values = []
        
        # 计算数据质量指标
        total_molecules = len(df)
        quality_metrics.append('总分子数')
        quality_values.append(total_molecules)
        
        if 'Molwt' in df.columns:
            molwt_complete = df['Molwt'].notna().sum()
            quality_metrics.append('分子量完整')
            quality_values.append(molwt_complete)
        
        if '#Heavy' in df.columns:
            heavy_complete = df['#Heavy'].notna().sum()
            quality_metrics.append('重原子数完整')
            quality_values.append(heavy_complete)
        
        if 'MP' in df.columns:
            mp_complete = df['MP'].notna().sum()
            quality_metrics.append('熔点数据')
            quality_values.append(mp_complete)
        
        if 'BP' in df.columns:
            bp_complete = df['BP'].notna().sum()
            quality_metrics.append('沸点数据')
            quality_values.append(bp_complete)
        
        if 'FP' in df.columns:
            fp_complete = df['FP'].notna().sum()
            quality_metrics.append('闪点数据')
            quality_values.append(fp_complete)
        
        bars = ax.bar(range(len(quality_metrics)), quality_values, alpha=0.7, color='lightcoral', edgecolor='black')
        ax.set_title('数据完整性概览')
        ax.set_xticks(range(len(quality_metrics)))
        ax.set_xticklabels(quality_metrics, rotation=45, ha='right')
        ax.set_ylabel('数量')
        ax.grid(True, alpha=0.3)
        
        # 添加数值标签
        for bar, value in zip(bars, quality_values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                   str(value), ha='center', va='bottom')
        
        # 9. KPI框架合规性检查
        ax = axes[2, 2]
        compliance_checks = []
        compliance_status = []
        
        # 分子量合规性
        if 'Molwt' in df.columns:
            molwt_in_range = ((df['Molwt'] >= self.min_molecular_weight) & 
                             (df['Molwt'] <= self.max_molecular_weight)).sum()
            compliance_checks.append('分子量范围')
            compliance_status.append(molwt_in_range / len(df) if len(df) > 0 else 0)
        
        # 重原子数合规性
        if '#Heavy' in df.columns:
            heavy_in_range = ((df['#Heavy'] >= self.min_heavy_atoms) & 
                             (df['#Heavy'] <= self.max_heavy_atoms)).sum()
            compliance_checks.append('重原子数范围')
            compliance_status.append(heavy_in_range / len(df) if len(df) > 0 else 0)
        
        # 元素合规性
        if 'Elements' in df.columns:
            element_compliant = 0
            for elements_str in df['Elements'].dropna():
                elements = [e.strip() for e in str(elements_str).split(',')]
                if all(elem in self.allowed_elements for elem in elements):
                    element_compliant += 1
            compliance_checks.append('元素限制')
            compliance_status.append(element_compliant / len(df) if len(df) > 0 else 0)
        
        if compliance_checks:
            bars = ax.bar(range(len(compliance_checks)), compliance_status, alpha=0.7, color='gold', edgecolor='black')
            ax.set_title('KPI框架合规性')
            ax.set_xticks(range(len(compliance_checks)))
            ax.set_xticklabels(compliance_checks, rotation=45, ha='right')
            ax.set_ylabel('合规率')
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.3)
            
            # 添加数值标签
            for bar, value in zip(bars, compliance_status):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{value:.2f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"KPI框架可视化图表已保存到 {save_path}")
        
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