"""
MNIST手写数字识别 - 加载已训练模型进行预测
用法: python predict.py your_image.png
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import sys
import os

# 1. 定义与训练时完全相同的模型结构
class MNISTNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

def load_trained_model(model_path='saved_models/model_weights.pth'):
    """加载已训练的模型"""
    # 创建模型实例
    model = MNISTNet()
    
    # 加载模型权重
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    
    # 如果是在CPU上运行，需要指定map_location
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    
    # 检查加载的是完整检查点还是仅权重
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        # 完整检查点格式
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        # 仅权重格式
        model.load_state_dict(checkpoint)
    
    # 设置为评估模式
    model.eval()
    
    print(f"✓ 模型已从 {model_path} 加载")
    print(f"  模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    return model

def preprocess_image(image_path):
    """预处理输入图片，使其符合MNIST格式"""
    try:
        from PIL import Image
        img = Image.open(image_path)
        print(f"✓ 加载图片: {image_path}")
        print(f"  原始尺寸: {img.size}, 模式: {img.mode}")
        
        # 转换为灰度图
        if img.mode != 'L':
            img = img.convert('L')
            print(f"  已转换为灰度图")
        
        # 转换为numpy数组
        img_array = np.array(img, dtype=np.float32)
        print(f"  像素值范围: [{img_array.min():.1f}, {img_array.max():.1f}]")
        
        # ★ 重要：MNIST图片是黑底白字，不要自动反相 ★
        # 直接调整大小
        img = img.resize((28, 28), Image.Resampling.LANCZOS)
        
        # 转换为Tensor并标准化（与训练时完全一致）
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        img_tensor = transform(img)
        
        # 添加批次维度
        img_tensor = img_tensor.unsqueeze(0)  # [1, 1, 28, 28]
        
        # 验证
        print(f"  预处理后形状: {img_tensor.shape}")
        print(f"  预处理后范围: [{img_tensor.min():.3f}, {img_tensor.max():.3f}]")
        
        # 反标准化查看
        img_denorm = img_tensor * 0.3081 + 0.1307
        print(f"  反标准化范围: [{img_denorm.min():.3f}, {img_denorm.max():.3f}]")
        print(f"  反标准化均值: {img_denorm.mean():.3f} (应该接近0)")
        
        return img_tensor, img
    
    except Exception as e:
        print(f"✗ 图片处理失败: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def predict_digit(model, image_tensor, show_top_k=3):
    """使用模型预测数字"""
    with torch.no_grad():  # 不计算梯度，节省内存
        # 前向传播
        output = model(image_tensor)
        
        # 转换为概率
        probabilities = F.softmax(output, dim=1)[0]
        
        # 获取预测结果
        predicted_digit = torch.argmax(probabilities).item()
        predicted_prob = probabilities[predicted_digit].item()
        
        # 获取Top-K预测
        top_probs, top_indices = torch.topk(probabilities, show_top_k)
        
    return {
        'predicted_digit': predicted_digit,
        'predicted_prob': predicted_prob,
        'top_k': {
            'digits': top_indices.tolist(),
            'probs': top_probs.tolist()
        },
        'all_probs': probabilities.tolist()
    }

def visualize_prediction(img, prediction_result):
    """可视化预测结果"""
    try:
        import matplotlib.pyplot as plt
        
        # 创建图表
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # 左侧：显示图片
        ax1.imshow(np.array(img), cmap='gray')
        ax1.set_title(f'Input Image\nPredicted: {prediction_result["predicted_digit"]} '
                     f'({prediction_result["predicted_prob"]:.1%})', fontsize=12)
        ax1.axis('off')
        
        # 右侧：显示概率分布
        digits = list(range(10))
        probs = prediction_result['all_probs']
        
        colors = ['red' if i == prediction_result['predicted_digit'] else 'blue' for i in range(10)]
        ax2.bar(digits, probs, color=colors, alpha=0.7)
        ax2.set_xlabel('Digit')
        ax2.set_ylabel('Probability')
        ax2.set_title('Probability Distribution')
        ax2.set_xticks(digits)
        ax2.set_ylim([0, 1.1])
        
        # 在柱子上方显示概率
        for i, prob in enumerate(probs):
            ax2.text(i, prob + 0.02, f'{prob:.1%}', 
                    ha='center', fontsize=9)
        
        # 显示Top-K预测
        top_text = "Top Predictions:\n"
        for digit, prob in zip(prediction_result['top_k']['digits'], 
                              prediction_result['top_k']['probs']):
            top_text += f"  {digit}: {prob:.1%}\n"
        
        ax2.text(0.02, 0.98, top_text, 
                transform=ax2.transAxes, 
                verticalalignment='top',
                fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig('prediction_result.png', dpi=100, bbox_inches='tight')
        plt.show()
        
        print(f"✓ 预测结果已保存到: prediction_result.png")
        
    except ImportError:
        print("  注意: 需要安装matplotlib来显示可视化结果")
    except Exception as e:
        print(f"  可视化失败: {e}")

def test_with_mnist_sample(model):
    """用MNIST测试集验证模型是否正确"""
    print("\n[验证] 用MNIST测试集验证模型...")
    try:
        from torchvision import datasets, transforms
        
        # 使用与训练时完全相同的transform
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        test_dataset = datasets.MNIST(
            root='./data',
            train=False,
            download=False,  # 不重新下载
            transform=transform
        )
        
        # 测试前5个样本
        correct = 0
        total = 5
        
        for i in range(total):
            image, true_label = test_dataset[i]
            image_batch = image.unsqueeze(0)  # 添加batch维度
            
            with torch.no_grad():
                output = model(image_batch)
                predicted = torch.argmax(output, dim=1).item()
            
            if predicted == true_label:
                correct += 1
            
            print(f"  样本{i}: 真实={true_label}, 预测={predicted}, {'✓' if predicted == true_label else '✗'}")
        
        accuracy = correct / total * 100
        print(f"  验证准确率: {accuracy:.1f}% ({correct}/{total})")
        
        if accuracy >= 80.0:
            print("  ✓ 模型验证通过")
        else:
            print("  ⚠️ 模型可能有问题")
            
    except Exception as e:
        print(f"  验证失败: {e}")

def main():
    """主函数"""
    print("=" * 60)
    print("MNIST手写数字识别 - 预测工具")
    print("=" * 60)
    
    # 1. 检查命令行参数
    if len(sys.argv) < 2:
        print("用法: python predict.py <图片路径> [模型路径]")
        print("示例: python predict.py my_digit.png")
        print("示例: python predict.py my_digit.png saved_models/model_weights.pth")
        return
    
    image_path = sys.argv[1]
    model_path = sys.argv[2] if len(sys.argv) > 2 else 'saved_models/model_weights.pth'
    
    # 2. 检查图片是否存在
    if not os.path.exists(image_path):
        print(f"✗ 图片不存在: {image_path}")
        return
    
    # 3. 加载模型
    print("\n[1] 加载模型...")
    try:
        model = load_trained_model(model_path)
        
        # 验证模型
        test_with_mnist_sample(model)
        
    except Exception as e:
        print(f"✗ 模型加载失败: {e}")
        print("请确保:")
        print("  1. 模型文件存在")
        print("  2. 运行过训练代码保存了模型")
        print("  3. 模型路径正确")
        return
    
    # 4. 预处理图片
    print("\n[2] 预处理图片...")
    image_tensor, original_img = preprocess_image(image_path)
    if image_tensor is None:
        return
    
    # 5. 进行预测
    print("\n[3] 进行预测...")
    result = predict_digit(model, image_tensor, show_top_k=3)
    
    # 6. 显示结果
    print("\n" + "=" * 40)
    print("预测结果:")
    print("=" * 40)
    print(f"  最可能数字: {result['predicted_digit']}")
    print(f"  置信度: {result['predicted_prob']:.1%}")
    print(f"\n  详细概率分布:")
    for digit, prob in enumerate(result['all_probs']):
        prefix = "→" if digit == result['predicted_digit'] else " "
        print(f"    {prefix} 数字 {digit}: {prob:6.2%}")
    
    print(f"\n  Top 3预测:")
    for digit, prob in zip(result['top_k']['digits'], result['top_k']['probs']):
        print(f"    数字 {digit}: {prob:6.2%}")
    
    # 7. 可视化结果
    print("\n[4] 生成可视化...")
    visualize_prediction(original_img, result)
    
    print("\n" + "=" * 60)
    print("预测完成！")
    print("=" * 60)

if __name__ == "__main__":
    main()
