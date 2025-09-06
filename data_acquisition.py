"""
数据获取模块 - 从Materials Project数据库获取分子SMILES数据
Data Acquisition Module - Fetch molecular SMILES data from Materials Project database
"""

import requests
import pandas as pd
import time
from typing import List, Dict, Optional
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MaterialsProjectDataAcquisition:
    """Materials Project数据获取类"""
    
    def __init__(self, api_key: str):
        """
        初始化Materials Project数据获取器
        
        Args:
            api_key (str): Materials Project API密钥
        """
        self.api_key = api_key
        self.base_url = "https://api.materialsproject.org"
        self.headers = {"X-API-KEY": api_key}
        
    def search_materials(self, 
                        formula: Optional[str] = None,
                        elements: Optional[List[str]] = None,
                        exclude_elements: Optional[List[str]] = None,
                        num_sites: Optional[int] = None,
                        num_chunks: int = 10) -> pd.DataFrame:
        """
        搜索材料并获取基本信息
        
        Args:
            formula (str, optional): 化学式过滤
            elements (List[str], optional): 包含元素列表
            exclude_elements (List[str], optional): 排除元素列表
            num_sites (int, optional): 位点数过滤
            num_chunks (int): 分块获取数量
            
        Returns:
            pd.DataFrame: 包含材料基本信息的DataFrame
        """
        logger.info("开始搜索Materials Project数据库...")
        
        # 构建搜索参数
        search_params = {
            "num_chunks": num_chunks,
            "chunk_size": 1000
        }
        
        if formula:
            search_params["formula"] = formula
        if elements:
            search_params["elements"] = ",".join(elements)
        if exclude_elements:
            search_params["exclude_elements"] = ",".join(exclude_elements)
        if num_sites:
            search_params["num_sites"] = num_sites
            
        try:
            # 搜索材料
            search_url = f"{self.base_url}/materials/summary"
            response = requests.get(search_url, headers=self.headers, params=search_params)
            response.raise_for_status()
            
            search_data = response.json()
            materials = search_data.get("data", [])
            
            logger.info(f"找到 {len(materials)} 个材料")
            
            # 提取基本信息
            material_info = []
            for material in materials:
                info = {
                    "material_id": material.get("material_id"),
                    "formula_pretty": material.get("formula_pretty"),
                    "structure": material.get("structure"),
                    "density": material.get("density"),
                    "volume": material.get("volume"),
                    "nsites": material.get("nsites"),
                    "elements": material.get("elements", []),
                    "nelements": material.get("nelements")
                }
                material_info.append(info)
            
            return pd.DataFrame(material_info)
            
        except requests.exceptions.RequestException as e:
            logger.error(f"搜索材料时发生错误: {e}")
            return pd.DataFrame()
    
    def get_material_details(self, material_ids: List[str]) -> pd.DataFrame:
        """
        获取材料的详细属性信息
        
        Args:
            material_ids (List[str]): 材料ID列表
            
        Returns:
            pd.DataFrame: 包含详细属性的DataFrame
        """
        logger.info(f"获取 {len(material_ids)} 个材料的详细属性...")
        
        all_properties = []
        
        for i, material_id in enumerate(material_ids):
            try:
                # 获取材料属性
                properties_url = f"{self.base_url}/materials/{material_id}/thermo"
                response = requests.get(properties_url, headers=self.headers)
                response.raise_for_status()
                
                prop_data = response.json()
                
                # 提取热力学属性
                thermo_data = prop_data.get("data", {})
                
                # 获取结构信息
                structure_url = f"{self.base_url}/materials/{material_id}"
                structure_response = requests.get(structure_url, headers=self.headers)
                structure_response.raise_for_status()
                structure_data = structure_response.json()
                
                # 提取结构信息
                structure = structure_data.get("data", {}).get("structure", {})
                
                material_props = {
                    "material_id": material_id,
                    "formula": structure_data.get("data", {}).get("formula_pretty", ""),
                    "formation_energy_per_atom": thermo_data.get("formation_energy_per_atom"),
                    "energy_above_hull": thermo_data.get("energy_above_hull"),
                    "is_stable": thermo_data.get("is_stable"),
                    "band_gap": structure_data.get("data", {}).get("band_gap"),
                    "density": structure_data.get("data", {}).get("density"),
                    "volume": structure_data.get("data", {}).get("volume"),
                    "nsites": structure_data.get("data", {}).get("nsites"),
                    "elements": structure_data.get("data", {}).get("elements", []),
                    "lattice": structure.get("lattice", {}),
                    "sites": structure.get("sites", [])
                }
                
                all_properties.append(material_props)
                
                # 添加延迟避免API限制
                time.sleep(0.1)
                
                if (i + 1) % 10 == 0:
                    logger.info(f"已处理 {i + 1}/{len(material_ids)} 个材料")
                    
            except requests.exceptions.RequestException as e:
                logger.warning(f"获取材料 {material_id} 属性时发生错误: {e}")
                continue
        
        return pd.DataFrame(all_properties)
    
    def generate_smiles_from_structure(self, structure_data: Dict) -> Optional[str]:
        """
        从结构数据生成SMILES字符串（简化版本）
        注意：这是一个简化的实现，实际应用中需要使用更专业的结构转换工具
        
        Args:
            structure_data (Dict): 结构数据字典
            
        Returns:
            Optional[str]: SMILES字符串或None
        """
        try:
            # 这里是一个简化的SMILES生成逻辑
            # 实际应用中应该使用RDKit或其他专业工具
            elements = structure_data.get("elements", [])
            if not elements:
                return None
                
            # 简单的分子式到SMILES的映射（仅用于演示）
            formula_to_smiles = {
                "H2O": "O",
                "CO2": "O=C=O",
                "CH4": "C",
                "NH3": "N",
                "H2": "[H][H]",
                "O2": "O=O",
                "N2": "N#N"
            }
            
            # 尝试匹配已知的简单分子
            for formula, smiles in formula_to_smiles.items():
                if all(elem in elements for elem in formula.replace("2", "").replace("3", "").replace("4", "")):
                    return smiles
            
            # 如果无法匹配，返回基于元素组合的简单SMILES
            if len(elements) == 1:
                return elements[0]
            elif len(elements) == 2:
                return f"{elements[0]}-{elements[1]}"
            else:
                return "-".join(elements)
                
        except Exception as e:
            logger.warning(f"生成SMILES时发生错误: {e}")
            return None
    
    def create_origin_sheet(self, materials_df: pd.DataFrame) -> pd.DataFrame:
        """
        创建原始数据表（Origin Sheet）
        
        Args:
            materials_df (pd.DataFrame): 材料数据DataFrame
            
        Returns:
            pd.DataFrame: 原始数据表
        """
        logger.info("创建原始数据表...")
        
        origin_data = []
        
        for _, row in materials_df.iterrows():
            # 生成SMILES（简化版本）
            structure_data = {"elements": row.get("elements", [])}
            smiles = self.generate_smiles_from_structure(structure_data)
            
            if smiles:
                origin_row = {
                    "SMILES": smiles,
                    "Material_ID": row.get("material_id", ""),
                    "Formula": row.get("formula_pretty", ""),
                    "Density": row.get("density"),
                    "Volume": row.get("volume"),
                    "NSites": row.get("nsites"),
                    "Elements": ",".join(row.get("elements", [])),
                    "Formation_Energy": row.get("formation_energy_per_atom"),
                    "Band_Gap": row.get("band_gap"),
                    "Is_Stable": row.get("is_stable")
                }
                origin_data.append(origin_row)
        
        origin_df = pd.DataFrame(origin_data)
        logger.info(f"原始数据表包含 {len(origin_df)} 行数据")
        
        return origin_df
    
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
    # 注意：需要替换为实际的API密钥
    API_KEY = "your_materials_project_api_key_here"
    
    if API_KEY == "your_materials_project_api_key_here":
        print("请先设置有效的Materials Project API密钥")
        return
    
    # 创建数据获取器
    data_acquirer = MaterialsProjectDataAcquisition(API_KEY)
    
    # 搜索材料（示例：搜索包含C、H、O的材料）
    materials_df = data_acquirer.search_materials(
        elements=["C", "H", "O"],
        num_chunks=5
    )
    
    if not materials_df.empty:
        # 获取详细属性
        material_ids = materials_df["material_id"].tolist()[:20]  # 限制数量避免API限制
        detailed_df = data_acquirer.get_material_details(material_ids)
        
        # 创建原始数据表
        origin_sheet = data_acquirer.create_origin_sheet(detailed_df)
        
        # 保存数据
        data_acquirer.save_data(origin_sheet, "materials_origin_data.csv")
        
        print(f"成功获取并保存了 {len(origin_sheet)} 个材料的数据")
        print("\n数据预览:")
        print(origin_sheet.head())
    else:
        print("未找到符合条件的材料数据")

if __name__ == "__main__":
    main()