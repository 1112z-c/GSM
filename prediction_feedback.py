"""
预测和反馈模块
Prediction and Feedback Module
"""

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional, Union
import logging
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PropertyPredictor:
    """属性预测器"""
    
    def __init__(self, model: torch.nn.Module, device: str = 'cpu'):
        """
        初始化属性预测器
        
        Args:
            model (torch.nn.Module): 训练好的模型
            device (str): 计算设备
        """
        self.model = model
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()
        
        # 属性名称和单位
        self.property_info = {
            'melting_point': {'name': '熔点', 'unit': 'K', 'range': (50, 2000)},
            'boiling_point': {'name': '沸点', 'unit': 'K', 'range': (100, 3000)},
            'flash_point': {'name': '闪点', 'unit': 'K', 'range': (0, 500)}
        }
    
    def predict_properties(self, 
                          molecular_embedding: np.ndarray, 
                          knowledge_vector: np.ndarray) -> Dict[str, Dict[str, Union[float, str]]]:
        """
        预测分子属性
        
        Args:
            molecular_embedding (np.ndarray): 分子嵌入
            knowledge_vector (np.ndarray): 知识向量
            
        Returns:
            Dict[str, Dict[str, Union[float, str]]]: 预测结果
        """
        logger.info("开始预测分子属性...")
        
        # 转换为张量
        mol_tensor = torch.FloatTensor(molecular_embedding).unsqueeze(0).to(self.device)
        know_tensor = torch.FloatTensor(knowledge_vector).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            predictions = self.model(mol_tensor, know_tensor)
        
        # 处理预测结果
        results = {}
        for property_name, property_data in self.property_info.items():
            if property_name in predictions:
                pred_value = predictions[property_name].cpu().numpy()[0]
                
                # 应用范围限制
                min_val, max_val = property_data['range']
                pred_value = np.clip(pred_value, min_val, max_val)
                
                results[property_name] = {
                    'value': float(pred_value),
                    'name': property_data['name'],
                    'unit': property_data['unit'],
                    'confidence': self._calculate_confidence(pred_value, property_name),
                    'timestamp': datetime.now().isoformat()
                }
        
        logger.info("属性预测完成")
        return results
    
    def _calculate_confidence(self, value: float, property_name: str) -> float:
        """
        计算预测置信度
        
        Args:
            value (float): 预测值
            property_name (str): 属性名称
            
        Returns:
            float: 置信度 (0-1)
        """
        # 简化的置信度计算
        min_val, max_val = self.property_info[property_name]['range']
        range_size = max_val - min_val
        
        # 基于值在合理范围内的程度计算置信度
        if min_val <= value <= max_val:
            distance_from_center = abs(value - (min_val + max_val) / 2)
            max_distance = range_size / 2
            confidence = 1.0 - (distance_from_center / max_distance) * 0.3
        else:
            confidence = 0.1
        
        return max(0.1, min(1.0, confidence))

class ResultVisualizer:
    """结果可视化器"""
    
    def __init__(self):
        """初始化结果可视化器"""
        self.colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
    
    def plot_property_predictions(self, 
                                predictions: Dict[str, Dict[str, Union[float, str]]],
                                save_path: Optional[str] = None):
        """
        绘制属性预测结果
        
        Args:
            predictions (Dict): 预测结果
            save_path (str, optional): 保存路径
        """
        logger.info("生成属性预测可视化...")
        
        properties = list(predictions.keys())
        values = [predictions[prop]['value'] for prop in properties]
        names = [predictions[prop]['name'] for prop in properties]
        units = [predictions[prop]['unit'] for prop in properties]
        confidences = [predictions[prop]['confidence'] for prop in properties]
        
        # 创建子图
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('分子属性预测结果', fontsize=16, fontweight='bold')
        
        # 1. 属性值柱状图
        bars = axes[0, 0].bar(names, values, color=self.colors[:len(properties)])
        axes[0, 0].set_title('预测属性值', fontsize=14)
        axes[0, 0].set_ylabel('温度 (K)')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 添加数值标签
        for bar, value in zip(bars, values):
            axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                           f'{value:.1f}K', ha='center', va='bottom')
        
        # 2. 置信度雷达图
        angles = np.linspace(0, 2 * np.pi, len(properties), endpoint=False).tolist()
        angles += angles[:1]  # 闭合图形
        confidences += confidences[:1]
        
        axes[0, 1].plot(angles, confidences, 'o-', linewidth=2, color='#FF6B6B')
        axes[0, 1].fill(angles, confidences, alpha=0.25, color='#FF6B6B')
        axes[0, 1].set_xticks(angles[:-1])
        axes[0, 1].set_xticklabels(names)
        axes[0, 1].set_ylim(0, 1)
        axes[0, 1].set_title('预测置信度', fontsize=14)
        axes[0, 1].grid(True)
        
        # 3. 属性值饼图
        axes[1, 0].pie(values, labels=[f'{name}\n{value:.1f}K' for name, value in zip(names, values)],
                       colors=self.colors[:len(properties)], autopct='%1.1f%%', startangle=90)
        axes[1, 0].set_title('属性值分布', fontsize=14)
        
        # 4. 置信度条形图
        bars = axes[1, 1].bar(names, confidences, color=self.colors[:len(properties)])
        axes[1, 1].set_title('预测置信度', fontsize=14)
        axes[1, 1].set_ylabel('置信度')
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        # 添加置信度标签
        for bar, conf in zip(bars, confidences):
            axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{conf:.2f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"预测结果图表已保存到 {save_path}")
        
        plt.show()
    
    def plot_comparison(self, 
                       predictions: Dict[str, Dict[str, Union[float, str]]],
                       reference_values: Optional[Dict[str, float]] = None,
                       save_path: Optional[str] = None):
        """
        绘制预测值与参考值的比较
        
        Args:
            predictions (Dict): 预测结果
            reference_values (Dict, optional): 参考值
            save_path (str, optional): 保存路径
        """
        if reference_values is None:
            logger.warning("未提供参考值，跳过比较图")
            return
        
        logger.info("生成预测比较可视化...")
        
        properties = list(predictions.keys())
        pred_values = [predictions[prop]['value'] for prop in properties]
        ref_values = [reference_values.get(prop, 0) for prop in properties]
        names = [predictions[prop]['name'] for prop in properties]
        
        # 创建比较图
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        x = np.arange(len(properties))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, pred_values, width, label='预测值', color='#FF6B6B', alpha=0.8)
        bars2 = ax.bar(x + width/2, ref_values, width, label='参考值', color='#4ECDC4', alpha=0.8)
        
        ax.set_xlabel('属性')
        ax.set_ylabel('温度 (K)')
        ax.set_title('预测值与参考值比较')
        ax.set_xticks(x)
        ax.set_xticklabels(names)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 添加数值标签
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, height + 5,
                       f'{height:.1f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"比较图表已保存到 {save_path}")
        
        plt.show()

class FeedbackGenerator:
    """反馈生成器"""
    
    def __init__(self):
        """初始化反馈生成器"""
        self.feedback_templates = {
            'high_confidence': "预测结果具有高置信度，建议采用。",
            'medium_confidence': "预测结果置信度中等，建议进一步验证。",
            'low_confidence': "预测结果置信度较低，建议谨慎使用。",
            'out_of_range': "预测值超出正常范围，建议检查输入数据。",
            'excellent_prediction': "预测结果与参考值高度一致，模型表现优秀。",
            'good_prediction': "预测结果与参考值较为一致，模型表现良好。",
            'poor_prediction': "预测结果与参考值存在较大差异，建议优化模型。"
        }
    
    def generate_feedback(self, 
                         predictions: Dict[str, Dict[str, Union[float, str]]],
                         reference_values: Optional[Dict[str, float]] = None) -> Dict[str, str]:
        """
        生成反馈信息
        
        Args:
            predictions (Dict): 预测结果
            reference_values (Dict, optional): 参考值
            
        Returns:
            Dict[str, str]: 反馈信息
        """
        logger.info("生成反馈信息...")
        
        feedback = {}
        
        for prop_name, pred_data in predictions.items():
            confidence = pred_data['confidence']
            value = pred_data['value']
            
            # 基于置信度的反馈
            if confidence >= 0.8:
                conf_feedback = self.feedback_templates['high_confidence']
            elif confidence >= 0.6:
                conf_feedback = self.feedback_templates['medium_confidence']
            else:
                conf_feedback = self.feedback_templates['low_confidence']
            
            # 基于预测值的反馈
            if value < 0 or value > 5000:  # 异常值检查
                value_feedback = self.feedback_templates['out_of_range']
            else:
                value_feedback = ""
            
            # 与参考值的比较反馈
            if reference_values and prop_name in reference_values:
                ref_value = reference_values[prop_name]
                error = abs(value - ref_value) / ref_value if ref_value != 0 else 1.0
                
                if error <= 0.05:  # 5%误差内
                    comp_feedback = self.feedback_templates['excellent_prediction']
                elif error <= 0.15:  # 15%误差内
                    comp_feedback = self.feedback_templates['good_prediction']
                else:
                    comp_feedback = self.feedback_templates['poor_prediction']
            else:
                comp_feedback = ""
            
            # 组合反馈
            all_feedback = [conf_feedback, value_feedback, comp_feedback]
            feedback[prop_name] = " ".join([f for f in all_feedback if f])
        
        logger.info("反馈信息生成完成")
        return feedback
    
    def generate_summary_report(self, 
                               predictions: Dict[str, Dict[str, Union[float, str]]],
                               feedback: Dict[str, str],
                               reference_values: Optional[Dict[str, float]] = None) -> str:
        """
        生成总结报告
        
        Args:
            predictions (Dict): 预测结果
            feedback (Dict): 反馈信息
            reference_values (Dict, optional): 参考值
            
        Returns:
            str: 总结报告
        """
        logger.info("生成总结报告...")
        
        report = []
        report.append("=" * 60)
        report.append("分子属性预测结果报告")
        report.append("=" * 60)
        report.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # 预测结果
        report.append("预测结果:")
        report.append("-" * 30)
        for prop_name, pred_data in predictions.items():
            report.append(f"{pred_data['name']}: {pred_data['value']:.2f} {pred_data['unit']}")
            report.append(f"  置信度: {pred_data['confidence']:.2f}")
            report.append(f"  反馈: {feedback[prop_name]}")
            report.append("")
        
        # 与参考值的比较
        if reference_values:
            report.append("与参考值比较:")
            report.append("-" * 30)
            for prop_name, pred_data in predictions.items():
                if prop_name in reference_values:
                    ref_value = reference_values[prop_name]
                    pred_value = pred_data['value']
                    error = abs(pred_value - ref_value) / ref_value * 100 if ref_value != 0 else 100
                    report.append(f"{pred_data['name']}:")
                    report.append(f"  预测值: {pred_value:.2f} {pred_data['unit']}")
                    report.append(f"  参考值: {ref_value:.2f} {pred_data['unit']}")
                    report.append(f"  误差: {error:.2f}%")
                    report.append("")
        
        # 总体评估
        avg_confidence = np.mean([pred_data['confidence'] for pred_data in predictions.values()])
        report.append("总体评估:")
        report.append("-" * 30)
        report.append(f"平均置信度: {avg_confidence:.2f}")
        
        if avg_confidence >= 0.8:
            report.append("模型表现: 优秀")
        elif avg_confidence >= 0.6:
            report.append("模型表现: 良好")
        else:
            report.append("模型表现: 需要改进")
        
        report.append("")
        report.append("=" * 60)
        
        return "\n".join(report)

class ResultExporter:
    """结果导出器"""
    
    def __init__(self):
        """初始化结果导出器"""
        pass
    
    def export_to_json(self, 
                      predictions: Dict[str, Dict[str, Union[float, str]]],
                      feedback: Dict[str, str],
                      filename: str = "prediction_results.json"):
        """
        导出结果到JSON文件
        
        Args:
            predictions (Dict): 预测结果
            feedback (Dict): 反馈信息
            filename (str): 文件名
        """
        logger.info(f"导出结果到 {filename}...")
        
        export_data = {
            'timestamp': datetime.now().isoformat(),
            'predictions': predictions,
            'feedback': feedback
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"结果已导出到 {filename}")
    
    def export_to_csv(self, 
                     predictions: Dict[str, Dict[str, Union[float, str]]],
                     filename: str = "prediction_results.csv"):
        """
        导出结果到CSV文件
        
        Args:
            predictions (Dict): 预测结果
            filename (str): 文件名
        """
        logger.info(f"导出结果到 {filename}...")
        
        data = []
        for prop_name, pred_data in predictions.items():
            data.append({
                'property': prop_name,
                'name': pred_data['name'],
                'value': pred_data['value'],
                'unit': pred_data['unit'],
                'confidence': pred_data['confidence'],
                'timestamp': pred_data['timestamp']
            })
        
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False, encoding='utf-8')
        
        logger.info(f"结果已导出到 {filename}")

def main():
    """主函数示例"""
    # 创建示例预测结果
    sample_predictions = {
        'melting_point': {
            'value': 273.15,
            'name': '熔点',
            'unit': 'K',
            'confidence': 0.85,
            'timestamp': datetime.now().isoformat()
        },
        'boiling_point': {
            'value': 373.15,
            'name': '沸点',
            'unit': 'K',
            'confidence': 0.92,
            'timestamp': datetime.now().isoformat()
        },
        'flash_point': {
            'value': 333.15,
            'name': '闪点',
            'unit': 'K',
            'confidence': 0.78,
            'timestamp': datetime.now().isoformat()
        }
    }
    
    # 创建可视化器
    visualizer = ResultVisualizer()
    
    # 绘制预测结果
    visualizer.plot_property_predictions(sample_predictions, 'property_predictions.png')
    
    # 创建反馈生成器
    feedback_generator = FeedbackGenerator()
    
    # 生成反馈
    feedback = feedback_generator.generate_feedback(sample_predictions)
    print("反馈信息:")
    for prop, fb in feedback.items():
        print(f"{prop}: {fb}")
    
    # 生成总结报告
    report = feedback_generator.generate_summary_report(sample_predictions, feedback)
    print("\n总结报告:")
    print(report)
    
    # 创建结果导出器
    exporter = ResultExporter()
    
    # 导出结果
    exporter.export_to_json(sample_predictions, feedback, 'sample_results.json')
    exporter.export_to_csv(sample_predictions, 'sample_results.csv')
    
    print("预测和反馈模块演示完成！")

if __name__ == "__main__":
    main()