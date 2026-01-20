"""
PyTorch Hello World - 使用真实MNIST数据集
需要先安装: pip install torchvision matplotlib
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import pickle
import json
import os
from datetime import datetime

def save_training_data(model, train_losses, test_losses, test_accuracies, 
                       optimizer, epoch_history, config, filename='training_data.pth'):
    """
    保存完整的训练数据
    """
    # 创建保存目录
    save_dir = 'saved_models'
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. 保存模型权重
    model_path = os.path.join(save_dir, 'model_weights.pth')
    torch.save(model.state_dict(), model_path)
    
    # 2. 保存完整模型（包含结构）
    full_model_path = os.path.join(save_dir, 'full_model.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'model_class': model.__class__.__name__,
        'model_config': {
            'layer_sizes': [28 * 28, 128, 64, 10],
            'dropout_rate': 0.2
        }
    }, full_model_path)
    
    # 3. 保存训练历史（pickle格式）
    history_path = os.path.join(save_dir, 'training_history.pkl')
    with open(history_path, 'wb') as f:
        pickle.dump({
            'train_losses': train_losses,
            'test_losses': test_losses,
            'test_accuracies': test_accuracies,
            'epoch_history': epoch_history,
            'save_timestamp': datetime.now().isoformat()
        }, f)
    
    # 4. 保存训练历史（JSON格式，便于查看）
    json_history_path = os.path.join(save_dir, 'training_history.json')
    with open(json_history_path, 'w') as f:
        json.dump({
            'train_losses': [float(loss) for loss in train_losses],
            'test_losses': [float(loss) for loss in test_losses],
            'test_accuracies': [float(acc) for acc in test_accuracies],
            'epoch_details': epoch_history,
            'final_accuracy': float(test_accuracies[-1]) if test_accuracies else 0.0,
            'training_config': config,
            'save_time': datetime.now().isoformat(),
            'model_summary': str(model)
        }, f, indent=2)
    
    # 5. 保存训练配置
    config_path = os.path.join(save_dir, 'training_config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    # 6. 保存模型结构定义
    model_code_path = os.path.join(save_dir, 'model_definition.py')
    with open(model_code_path, 'w') as f:
        f.write("""
import torch
import torch.nn as nn
import torch.nn.functional as F

class MNISTNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

def load_model(model_path):
    model = MNISTNet()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model
""")
    
    print(f"\n训练数据已保存到目录: {save_dir}")
    print(f"├── 模型权重: {model_path}")
    print(f"├── 完整模型: {full_model_path}")
    print(f"├── 训练历史(pkl): {history_path}")
    print(f"├── 训练历史(json): {json_history_path}")
    print(f"├── 训练配置: {config_path}")
    print(f"└── 模型定义: {model_code_path}")
    
    return save_dir

def load_training_data(save_dir='saved_models'):
    """
    加载保存的训练数据
    """
    if not os.path.exists(save_dir):
        print(f"保存目录不存在: {save_dir}")
        return None
    
    # 加载模型
    model_path = os.path.join(save_dir, 'full_model.pth')
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    
    # 重新创建模型
    model = MNISTNet()
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 加载训练历史
    history_path = os.path.join(save_dir, 'training_history.pkl')
    with open(history_path, 'rb') as f:
        history = pickle.load(f)
    
    # 加载配置
    config_path = os.path.join(save_dir, 'training_config.json')
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    return {
        'model': model,
        'history': history,
        'config': config,
        'checkpoint': checkpoint
    }

def main():
    # 0. 训练配置
    training_config = {
        'dataset': 'MNIST',
        'batch_size': 64,
        'test_batch_size': 1000,
        'learning_rate': 0.001,
        'num_epochs': 5,
        'optimizer': 'Adam',
        'loss_function': 'CrossEntropyLoss',
        'model_architecture': 'ThreeLayerFC',
        'dropout_rate': 0.2,
        'random_seed': 42,
        'start_time': datetime.now().isoformat()
    }
    
    # 设置随机种子
    torch.manual_seed(training_config['random_seed'])

    # 1. 数据预处理
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # 2. 下载并加载数据集
    train_dataset = datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = datasets.MNIST(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )

    # 3. 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=training_config['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=training_config['test_batch_size'], shuffle=False)

    print(f"训练集大小: {len(train_dataset)}")
    print(f"测试集大小: {len(test_dataset)}")

    # 4. 定义模型
    class MNISTNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(28 * 28, 128)
            self.fc2 = nn.Linear(128, 64)
            self.fc3 = nn.Linear(64, 10)
            self.dropout = nn.Dropout(0.2)

        def forward(self, x):
            x = x.view(-1, 28 * 28)
            x = torch.relu(self.fc1(x))
            x = self.dropout(x)
            x = torch.relu(self.fc2(x))
            x = self.dropout(x)
            x = self.fc3(x)
            return x

    # 实例化模型
    model = MNISTNet()

    # 5. 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=training_config['learning_rate'])

    # 6. 训练函数
    def train(epoch):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

            if batch_idx % 100 == 0:
                print(f'训练进度: [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                      f'({100. * batch_idx / len(train_loader):.0f}%)]')

        avg_loss = running_loss / len(train_loader)
        accuracy = 100. * correct / total
        return avg_loss, accuracy

    # 7. 测试函数
    def test():
        model.eval()
        test_loss = 0
        correct = 0

        with torch.no_grad():
            for data, target in test_loader:
                output = model(data)
                test_loss += criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader)
        accuracy = 100. * correct / len(test_loader.dataset)
        return test_loss, accuracy

    # 8. 开始训练
    print("\n开始训练...")
    num_epochs = training_config['num_epochs']
    train_losses = []
    test_losses = []
    test_accuracies = []
    epoch_history = []

    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = train(epoch)
        test_loss, test_acc = test()

        train_losses.append(train_loss)
        test_losses.append(test_loss)
        test_accuracies.append(test_acc)
        
        # 记录每个epoch的详细信息
        epoch_info = {
            'epoch': epoch,
            'train_loss': float(train_loss),
            'train_accuracy': float(train_acc),
            'test_loss': float(test_loss),
            'test_accuracy': float(test_acc),
            'timestamp': datetime.now().isoformat()
        }
        epoch_history.append(epoch_info)

        print(f'\nEpoch {epoch}:')
        print(f'训练集 - 平均损失: {train_loss:.4f}, 准确率: {train_acc:.2f}%')
        print(f'测试集 - 平均损失: {test_loss:.4f}, 准确率: {test_acc:.2f}%\n')

    # 更新训练配置
    training_config['end_time'] = datetime.now().isoformat()
    training_config['final_accuracy'] = float(test_accuracies[-1]) if test_accuracies else 0.0
    training_config['total_parameters'] = sum(p.numel() for p in model.parameters())
    training_config['trainable_parameters'] = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # 9. 保存训练数据
    save_dir = save_training_data(
        model=model,
        train_losses=train_losses,
        test_losses=test_losses,
        test_accuracies=test_accuracies,
        optimizer=optimizer,
        epoch_history=epoch_history,
        config=training_config
    )

    # 10. 可视化结果
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # 损失曲线
    ax1.plot(range(1, num_epochs + 1), train_losses, 'b-', label='train lost')
    ax1.plot(range(1, num_epochs + 1), test_losses, 'r-', label='test lost')
    ax1.set_xlabel('train time')
    ax1.set_ylabel('lost')
    ax1.set_title('train and test lost curve')
    ax1.legend()
    ax1.grid(True)

    # 准确率曲线
    ax2.plot(range(1, num_epochs + 1), test_accuracies, 'g-', label='test acu rate')
    ax2.set_xlabel('test time')
    ax2.set_ylabel('acurate %')
    ax2.set_title('test acu rate curv')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    
    # 保存图表
    plt.savefig(os.path.join(save_dir, 'training_curves.png'))
    plt.show()

    # 11. 显示测试集样本
    print("\ntest predict example")
    model.eval()
    with torch.no_grad():
        data, target = next(iter(test_loader))
        data, target = data[:6], target[:6]
        output = model(data)
        pred = output.argmax(dim=1)

        fig, axes = plt.subplots(2, 3, figsize=(8, 5))
        for i, ax in enumerate(axes.flat):
            ax.imshow(data[i].squeeze(), cmap='gray')
            ax.set_title(f'actual {target[i]}, predict {pred[i]}')
            ax.axis('off')
        plt.tight_layout()
        
        # 保存预测示例图
        plt.savefig(os.path.join(save_dir, 'prediction_examples.png'))
        plt.show()

    print(f"\nfinal test acu rate {test_accuracies[-1]:.2f}%")
    print(f"所有训练数据已保存到: {save_dir}")
    print("train complete")
    
    return model, save_dir

if __name__ == "__main__":
    model, save_dir = main()
    
    # 演示如何加载保存的数据
    print(f"\n你可以通过以下方式重新加载模型:")
    print(f"1. 加载完整训练数据:")
    print(f"   loaded_data = load_training_data('{save_dir}')")
    print(f"2. 只加载模型进行预测:")
    print(f"""   from saved_models.model_definition import load_model
   model = load_model('{save_dir}/model_weights.pth')
   # 现在可以使用model进行预测""")
