"""
数据获取模块 - 从论文和公共数据库收集分子SMILES数据
Data Acquisition Module - Collect molecular SMILES data from papers and public databases
"""

import requests
import pandas as pd
import time
import json
import re
from typing import List, Dict, Optional, Tuple
import logging
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KPIDataAcquisition:
    """KPI数据获取类 - 从多个数据源收集电解质分子数据"""
    
    def __init__(self, 
                 materials_project_api_key: Optional[str] = None,
                 pubchem_api_key: Optional[str] = None,
                 crossref_api_key: Optional[str] = None):
        """
        初始化KPI数据获取器
        
        Args:
            materials_project_api_key (str, optional): Materials Project API密钥
            pubchem_api_key (str, optional): PubChem API密钥
            crossref_api_key (str, optional): CrossRef API密钥
        """
        self.materials_project_api_key = materials_project_api_key
        self.pubchem_api_key = pubchem_api_key
        self.crossref_api_key = crossref_api_key
        
        # API配置
        self.materials_project_url = "https://api.materialsproject.org"
        self.pubchem_url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
        self.crossref_url = "https://api.crossref.org"
        
        # 允许的元素（根据KPI要求）
        self.allowed_elements = {'H', 'C', 'N', 'O', 'F', 'Si', 'P', 'Cl', 'Br', 'I'}
        
        # 分子量限制（0-600）
        self.min_molecular_weight = 0
        self.max_molecular_weight = 600
        
        # 重原子数限制（0-30）
        self.min_heavy_atoms = 0
        self.max_heavy_atoms = 30
        
        # 目标属性
        self.target_properties = ['melting_point', 'boiling_point', 'flash_point']
        
        logger.info("KPI数据获取器初始化完成")
    
    def collect_from_materials_project(self, 
                                     elements: Optional[List[str]] = None,
                                     max_materials: int = 1000) -> pd.DataFrame:
        """
        从Materials Project收集数据
        
        Args:
            elements (List[str], optional): 元素列表
            max_materials (int): 最大材料数量
            
        Returns:
            pd.DataFrame: 材料数据
        """
        logger.info("从Materials Project收集数据...")
        
        if not self.materials_project_api_key:
            logger.warning("Materials Project API密钥未设置，跳过此数据源")
            return pd.DataFrame()
        
        # 过滤允许的元素
        if elements:
            elements = [e for e in elements if e in self.allowed_elements]
        
        try:
            # 搜索材料
            search_params = {
                "num_chunks": max_materials // 100 + 1,
                "chunk_size": 100
            }
            
            if elements:
                search_params["elements"] = ",".join(elements)
            
            response = requests.get(
                f"{self.materials_project_url}/materials/summary",
                headers={"X-API-KEY": self.materials_project_api_key},
                params=search_params,
                timeout=30
            )
            response.raise_for_status()
            
            data = response.json()
            materials = data.get("data", [])
            
            # 处理材料数据
            processed_materials = []
            for material in materials[:max_materials]:
                processed_material = self._process_materials_project_entry(material)
                if processed_material:
                    processed_materials.append(processed_material)
            
            logger.info(f"从Materials Project收集了 {len(processed_materials)} 个材料")
            return pd.DataFrame(processed_materials)
            
        except Exception as e:
            logger.error(f"从Materials Project收集数据时发生错误: {e}")
            return pd.DataFrame()
    
    def collect_from_pubchem(self, 
                           search_terms: List[str],
                           max_compounds: int = 500) -> pd.DataFrame:
        """
        从PubChem收集数据
        
        Args:
            search_terms (List[str]): 搜索词列表
            max_compounds (int): 最大化合物数量
            
        Returns:
            pd.DataFrame: 化合物数据
        """
        logger.info("从PubChem收集数据...")
        
        try:
            all_compounds = []
            
            for term in search_terms:
                # 搜索化合物
                search_url = f"{self.pubchem_url}/compound/name/{term}/property/MolecularFormula,MolecularWeight,CanonicalSMILES/JSON"
                
                response = requests.get(search_url, timeout=30)
                if response.status_code == 200:
                    data = response.json()
                    compounds = data.get("PropertyTable", {}).get("Properties", [])
                    
                    for compound in compounds[:max_compounds // len(search_terms)]:
                        processed_compound = self._process_pubchem_entry(compound)
                        if processed_compound:
                            all_compounds.append(processed_compound)
                
                time.sleep(0.1)  # 避免API限制
            
            logger.info(f"从PubChem收集了 {len(all_compounds)} 个化合物")
            return pd.DataFrame(all_compounds)
            
        except Exception as e:
            logger.error(f"从PubChem收集数据时发生错误: {e}")
            return pd.DataFrame()
    
    def collect_from_papers(self, 
                          keywords: List[str],
                          max_papers: int = 100) -> pd.DataFrame:
        """
        从已发表论文收集数据
        
        Args:
            keywords (List[str]): 关键词列表
            max_papers (int): 最大论文数量
            
        Returns:
            pd.DataFrame: 论文数据
        """
        logger.info("从已发表论文收集数据...")
        
        try:
            all_paper_data = []
            
            for keyword in keywords:
                # 使用CrossRef API搜索论文
                search_url = f"{self.crossref_url}/works"
                params = {
                    "query": f"{keyword} electrolyte molecular properties",
                    "rows": min(50, max_papers // len(keywords)),
                    "filter": "type:journal-article"
                }
                
                if self.crossref_api_key:
                    params["mailto"] = self.crossref_api_key
                
                response = requests.get(search_url, params=params, timeout=30)
                if response.status_code == 200:
                    data = response.json()
                    papers = data.get("message", {}).get("items", [])
                    
                    for paper in papers:
                        paper_data = self._extract_molecular_data_from_paper(paper)
                        if paper_data:
                            all_paper_data.extend(paper_data)
                
                time.sleep(0.1)  # 避免API限制
            
            logger.info(f"从论文中提取了 {len(all_paper_data)} 个分子数据")
            return pd.DataFrame(all_paper_data)
            
        except Exception as e:
            logger.error(f"从论文收集数据时发生错误: {e}")
            return pd.DataFrame()
    
    def _process_materials_project_entry(self, material: Dict) -> Optional[Dict]:
        """处理Materials Project条目"""
        try:
            # 提取基本信息
            material_id = material.get("material_id", "")
            formula = material.get("formula_pretty", "")
            structure = material.get("structure", {})
            
            # 检查元素限制
            elements = material.get("elements", [])
            if not self._check_element_restrictions(elements):
                return None
            
            # 计算分子量
            molecular_weight = self._calculate_molecular_weight(formula)
            if not self._check_molecular_weight_restrictions(molecular_weight):
                return None
            
            # 计算重原子数
            heavy_atoms = self._count_heavy_atoms(formula)
            if not self._check_heavy_atom_restrictions(heavy_atoms):
                return None
            
            # 生成SMILES
            smiles = self._generate_smiles_from_formula(formula)
            if not smiles:
                return None
            
            return {
                "source": "materials_project",
                "material_id": material_id,
                "smiles": smiles,
                "formula": formula,
                "molecular_weight": molecular_weight,
                "heavy_atoms": heavy_atoms,
                "elements": ",".join(elements),
                "density": material.get("density"),
                "volume": material.get("volume"),
                "nsites": material.get("nsites"),
                "formation_energy": material.get("formation_energy_per_atom"),
                "band_gap": material.get("band_gap"),
                "is_stable": material.get("is_stable")
            }
            
        except Exception as e:
            logger.warning(f"处理Materials Project条目时发生错误: {e}")
            return None
    
    def _process_pubchem_entry(self, compound: Dict) -> Optional[Dict]:
        """处理PubChem条目"""
        try:
            # 提取基本信息
            formula = compound.get("MolecularFormula", "")
            molecular_weight = compound.get("MolecularWeight", 0)
            smiles = compound.get("CanonicalSMILES", "")
            
            # 检查分子量限制
            if not self._check_molecular_weight_restrictions(molecular_weight):
                return None
            
            # 检查元素限制
            elements = self._extract_elements_from_formula(formula)
            if not self._check_element_restrictions(elements):
                return None
            
            # 计算重原子数
            heavy_atoms = self._count_heavy_atoms_from_smiles(smiles)
            if not self._check_heavy_atom_restrictions(heavy_atoms):
                return None
            
            return {
                "source": "pubchem",
                "material_id": f"pubchem_{hash(smiles)}",
                "smiles": smiles,
                "formula": formula,
                "molecular_weight": molecular_weight,
                "heavy_atoms": heavy_atoms,
                "elements": ",".join(elements),
                "density": None,
                "volume": None,
                "nsites": None,
                "formation_energy": None,
                "band_gap": None,
                "is_stable": None
            }
            
        except Exception as e:
            logger.warning(f"处理PubChem条目时发生错误: {e}")
            return None
    
    def _extract_molecular_data_from_paper(self, paper: Dict) -> List[Dict]:
        """从论文中提取分子数据"""
        try:
            # 这里是一个简化的实现，实际应用中需要更复杂的文本挖掘
            title = paper.get("title", [""])[0] if paper.get("title") else ""
            abstract = paper.get("abstract", "")
            
            # 简单的分子数据提取（实际应用中需要更复杂的NLP）
            molecular_data = []
            
            # 查找可能的分子式
            formula_pattern = r'([A-Z][a-z]?\d*)'
            formulas = re.findall(formula_pattern, title + " " + abstract)
            
            for formula in formulas[:5]:  # 限制每个论文最多5个分子
                if self._is_valid_formula(formula):
                    elements = self._extract_elements_from_formula(formula)
                    if self._check_element_restrictions(elements):
                        molecular_weight = self._calculate_molecular_weight(formula)
                        if self._check_molecular_weight_restrictions(molecular_weight):
                            heavy_atoms = self._count_heavy_atoms(formula)
                            if self._check_heavy_atom_restrictions(heavy_atoms):
                                smiles = self._generate_smiles_from_formula(formula)
                                if smiles:
                                    molecular_data.append({
                                        "source": "paper",
                                        "material_id": f"paper_{hash(formula)}",
                                        "smiles": smiles,
                                        "formula": formula,
                                        "molecular_weight": molecular_weight,
                                        "heavy_atoms": heavy_atoms,
                                        "elements": ",".join(elements),
                                        "density": None,
                                        "volume": None,
                                        "nsites": None,
                                        "formation_energy": None,
                                        "band_gap": None,
                                        "is_stable": None,
                                        "paper_title": title,
                                        "paper_doi": paper.get("DOI", "")
                                    })
            
            return molecular_data
            
        except Exception as e:
            logger.warning(f"从论文提取分子数据时发生错误: {e}")
            return []
    
    def _check_element_restrictions(self, elements: List[str]) -> bool:
        """检查元素限制"""
        if not elements:
            return False
        
        # 检查是否只包含允许的元素
        for element in elements:
            if element not in self.allowed_elements:
                return False
        
        return True
    
    def _check_molecular_weight_restrictions(self, molecular_weight: float) -> bool:
        """检查分子量限制"""
        return self.min_molecular_weight <= molecular_weight <= self.max_molecular_weight
    
    def _check_heavy_atom_restrictions(self, heavy_atoms: int) -> bool:
        """检查重原子数限制"""
        return self.min_heavy_atoms <= heavy_atoms <= self.max_heavy_atoms
    
    def _calculate_molecular_weight(self, formula: str) -> float:
        """计算分子量"""
        try:
            # 简化的分子量计算
            atomic_weights = {
                'H': 1.008, 'C': 12.011, 'N': 14.007, 'O': 15.999,
                'F': 18.998, 'Si': 28.085, 'P': 30.974,
                'Cl': 35.453, 'Br': 79.904, 'I': 126.904
            }
            
            total_weight = 0.0
            i = 0
            while i < len(formula):
                if formula[i].isupper():
                    # 原子符号
                    atom = formula[i]
                    if i + 1 < len(formula) and formula[i + 1].islower():
                        atom += formula[i + 1]
                        i += 1
                    
                    # 原子数量
                    count = 1
                    if i + 1 < len(formula) and formula[i + 1].isdigit():
                        count_str = ""
                        i += 1
                        while i < len(formula) and formula[i].isdigit():
                            count_str += formula[i]
                            i += 1
                        count = int(count_str) if count_str else 1
                        i -= 1
                    
                    if atom in atomic_weights:
                        total_weight += atomic_weights[atom] * count
                
                i += 1
            
            return total_weight
            
        except Exception:
            return 0.0
    
    def _count_heavy_atoms(self, formula: str) -> int:
        """计算重原子数"""
        heavy_atoms = {'C', 'N', 'O', 'F', 'Si', 'P', 'Cl', 'Br', 'I'}
        count = 0
        
        i = 0
        while i < len(formula):
            if formula[i].isupper():
                atom = formula[i]
                if i + 1 < len(formula) and formula[i + 1].islower():
                    atom += formula[i + 1]
                    i += 1
                
                if atom in heavy_atoms:
                    # 获取原子数量
                    atom_count = 1
                    if i + 1 < len(formula) and formula[i + 1].isdigit():
                        count_str = ""
                        i += 1
                        while i < len(formula) and formula[i].isdigit():
                            count_str += formula[i]
                            i += 1
                        atom_count = int(count_str) if count_str else 1
                        i -= 1
                    
                    count += atom_count
            
            i += 1
        
        return count
    
    def _count_heavy_atoms_from_smiles(self, smiles: str) -> int:
        """从SMILES计算重原子数"""
        heavy_atoms = {'C', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I', 'Si'}
        count = 0
        
        for char in smiles:
            if char.isupper() and char in heavy_atoms:
                count += 1
            elif char.islower() and char in heavy_atoms:
                count += 1
        
        return count
    
    def _extract_elements_from_formula(self, formula: str) -> List[str]:
        """从分子式提取元素"""
        elements = []
        i = 0
        
        while i < len(formula):
            if formula[i].isupper():
                element = formula[i]
                if i + 1 < len(formula) and formula[i + 1].islower():
                    element += formula[i + 1]
                    i += 1
                elements.append(element)
            i += 1
        
        return elements
    
    def _is_valid_formula(self, formula: str) -> bool:
        """检查分子式是否有效"""
        if not formula or len(formula) < 1:
            return False
        
        # 简单检查：包含至少一个字母
        return any(c.isalpha() for c in formula)
    
    def _generate_smiles_from_formula(self, formula: str) -> Optional[str]:
        """从分子式生成SMILES（简化版本）"""
        try:
            # 这是一个简化的实现，实际应用中需要使用RDKit等专业工具
            formula_to_smiles = {
                'H2O': 'O',
                'CO2': 'O=C=O',
                'CH4': 'C',
                'NH3': 'N',
                'H2': '[H][H]',
                'O2': 'O=O',
                'N2': 'N#N',
                'CH3OH': 'CO',
                'C2H5OH': 'CCO',
                'CH3COOH': 'CC(=O)O',
                'C6H6': 'c1ccccc1',
                'C2H4': 'C=C',
                'C2H2': 'C#C'
            }
            
            # 尝试直接匹配
            if formula in formula_to_smiles:
                return formula_to_smiles[formula]
            
            # 简化的SMILES生成
            elements = self._extract_elements_from_formula(formula)
            if len(elements) == 1:
                return elements[0]
            elif len(elements) == 2:
                return f"{elements[0]}-{elements[1]}"
            else:
                return "-".join(elements)
                
        except Exception:
            return None
    
    def collect_all_data(self, 
                        search_terms: List[str] = None,
                        elements: List[str] = None,
                        max_materials_per_source: int = 500) -> pd.DataFrame:
        """
        从所有数据源收集数据
        
        Args:
            search_terms (List[str], optional): 搜索词列表
            elements (List[str], optional): 元素列表
            max_materials_per_source (int): 每个数据源的最大材料数量
            
        Returns:
            pd.DataFrame: 合并的数据
        """
        logger.info("开始从所有数据源收集数据...")
        
        all_data = []
        
        # 从Materials Project收集
        mp_data = self.collect_from_materials_project(
            elements=elements,
            max_materials=max_materials_per_source
        )
        if not mp_data.empty:
            all_data.append(mp_data)
        
        # 从PubChem收集
        if search_terms:
            pubchem_data = self.collect_from_pubchem(
                search_terms=search_terms,
                max_compounds=max_materials_per_source
            )
            if not pubchem_data.empty:
                all_data.append(pubchem_data)
        
        # 从论文收集
        if search_terms:
            paper_data = self.collect_from_papers(
                keywords=search_terms,
                max_papers=max_materials_per_source // 10
            )
            if not paper_data.empty:
                all_data.append(paper_data)
        
        # 合并数据
        if all_data:
            combined_data = pd.concat(all_data, ignore_index=True)
            logger.info(f"总共收集了 {len(combined_data)} 个分子数据")
            return combined_data
        else:
            logger.warning("未收集到任何数据")
            return pd.DataFrame()
    
    def create_origin_sheet(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        创建原始数据表（Origin Sheet）
        
        Args:
            data (pd.DataFrame): 原始数据
            
        Returns:
            pd.DataFrame: 原始数据表
        """
        logger.info("创建原始数据表...")
        
        if data.empty:
            logger.warning("输入数据为空")
            return pd.DataFrame()
        
        # 选择相关列
        origin_columns = [
            'smiles', 'material_id', 'formula', 'molecular_weight', 
            'heavy_atoms', 'elements', 'density', 'volume', 'nsites',
            'formation_energy', 'band_gap', 'is_stable', 'source'
        ]
        
        # 过滤存在的列
        available_columns = [col for col in origin_columns if col in data.columns]
        origin_sheet = data[available_columns].copy()
        
        # 重命名列以匹配KPI要求
        column_mapping = {
            'smiles': 'SMILES',
            'material_id': 'Material_ID',
            'formula': 'Formula',
            'molecular_weight': 'Molwt',
            'heavy_atoms': '#Heavy',
            'elements': 'Elements',
            'density': 'Density',
            'volume': 'Volume',
            'nsites': 'NSites',
            'formation_energy': 'Formation_Energy',
            'band_gap': 'Band_Gap',
            'is_stable': 'Is_Stable',
            'source': 'Source'
        }
        
        origin_sheet = origin_sheet.rename(columns=column_mapping)
        
        # 添加目标属性列（初始化为None）
        for prop in self.target_properties:
            origin_sheet[prop.upper()] = None
        
        logger.info(f"原始数据表包含 {len(origin_sheet)} 行，{len(origin_sheet.columns)} 列")
        return origin_sheet
    
    def save_data(self, df: pd.DataFrame, filename: str):
        """
        保存数据到CSV文件
        
        Args:
            df (pd.DataFrame): 要保存的DataFrame
            filename (str): 文件名
        """
        try:
            df.to_csv(filename, index=False, encoding='utf-8')
            logger.info(f"数据已保存到 {filename}")
        except Exception as e:
            logger.error(f"保存数据时发生错误: {e}")

def main():
    """主函数示例"""
    # 创建数据获取器
    data_acquirer = KPIDataAcquisition(
        materials_project_api_key="your_materials_project_api_key_here",
        pubchem_api_key="your_pubchem_api_key_here",
        crossref_api_key="your_crossref_api_key_here"
    )
    
    # 定义搜索参数
    search_terms = ["electrolyte", "ionic liquid", "solvent", "electrolyte solution"]
    elements = ["C", "H", "O", "N", "F", "P", "Cl", "Br", "I"]
    
    # 收集数据
    all_data = data_acquirer.collect_all_data(
        search_terms=search_terms,
        elements=elements,
        max_materials_per_source=200
    )
    
    if not all_data.empty:
        # 创建原始数据表
        origin_sheet = data_acquirer.create_origin_sheet(all_data)
        
        # 保存数据
        data_acquirer.save_data(origin_sheet, "kpi_origin_data.csv")
        
        print(f"成功收集并保存了 {len(origin_sheet)} 个分子数据")
        print("\n数据预览:")
        print(origin_sheet.head())
        
        print(f"\n数据统计:")
        print(f"分子量范围: {origin_sheet['Molwt'].min():.2f} - {origin_sheet['Molwt'].max():.2f}")
        print(f"重原子数范围: {origin_sheet['#Heavy'].min()} - {origin_sheet['#Heavy'].max()}")
        print(f"数据源分布:")
        print(origin_sheet['Source'].value_counts())
    else:
        print("未收集到任何数据")

if __name__ == "__main__":
    main()