# MNIST数据以.pt或原始格式存储
# 可以直接从data/MNIST/raw读取

import os
import gzip
import numpy as np
from PIL import Image

def save_raw_mnist_images():
    """从原始.gz文件读取并保存"""
    raw_dir = './data/MNIST/raw'
    
    # MNIST文件路径
    train_images_gz = os.path.join(raw_dir, 'train-images-idx3-ubyte.gz')
    train_labels_gz = os.path.join(raw_dir, 'train-labels-idx1-ubyte.gz')
    
    if not os.path.exists(train_images_gz):
        print("需要先下载数据集")
        return
    
    # 读取图片
    with gzip.open(train_images_gz, 'rb') as f:
        # 跳过文件头
        f.read(16)
        # 读取图片数据
        buffer = f.read(28 * 28 * 60000)
        data = np.frombuffer(buffer, dtype=np.uint8)
        data = data.reshape(60000, 28, 28)
    
    # 读取标签
    with gzip.open(train_labels_gz, 'rb') as f:
        f.read(8)
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    
    # 保存图片
    save_dir = './mnist_raw_images'
    os.makedirs(save_dir, exist_ok=True)
    
    for i in range(100):  # 保存前100张
        img = Image.fromarray(data[i], mode='L')
        label = labels[i]
        
        # 按标签分类保存
        label_dir = os.path.join(save_dir, f'label_{label}')
        os.makedirs(label_dir, exist_ok=True)
        
        img.save(os.path.join(label_dir, f'raw_{i:04d}.png'))
    
    print(f"原始图片已保存到: {save_dir}")

save_raw_mnist_images()
