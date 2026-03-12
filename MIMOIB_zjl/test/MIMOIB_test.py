import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import torchvision
import utils.channel as channel
import json
import os
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ... [保留所有已有的函数定义，包括 save_test_results, MNISTDataLoader, Net, test 等] ...

if __name__ == "__main__":
    input_dim = 784
    latent_dim = 16
    batch_size = 512
    test_SNR = 20
    
    # ========== 第一部分：测试不同训练SNR的模型（固定κ=9）==========
    print("\n=== 测试不同训练SNR的模型（固定κ=9）===")
    
    model_weights_path = '/home/wen/project2/MIMO_IB_Compare/'
    
    # 左图需要的模型：不同训练SNR，但都是κ=9
    snr_models = [
        'MIMOIB_model/Complex_MIMOIB_infoNCE_SNR0_H9_20260312_112706.pth',
        'MIMOIB_model/Complex_MIMOIB_infoNCE_SNR5_H9_20260303_155117.pth',
        'MIMOIB_model/Complex_MIMOIB_infoNCE_SNR10_H9_20260312_101752.pth',
    ]
    
    # 初始化数据加载器
    mnist_loader = MNISTDataLoader(
        data_root='/home/wen/project2/data',
        train_batch_size=batch_size,
        test_batch_size=100
    )
    test_loader = mnist_loader.get_test_loader()
    
    # 存储左图数据
    left_results = {}  # key: 训练SNR, value: snr测试结果
    
    for model_name in snr_models:
        if not os.path.exists(model_weights_path + model_name):
            print(f"警告: 文件不存在 {model_name}")
            continue
            
        # 从文件名提取训练SNR
        if 'SNR0' in model_name:
            train_snr = 0
        elif 'SNR5' in model_name:
            train_snr = 5
        elif 'SNR10' in model_name:
            train_snr = 10
        else:
            continue
            
        print(f"\n测试模型: {model_name} (训练SNR={train_snr}dB)")
        
        model = Net(latent_dim).to(device)
        checkpoint = torch.load(model_weights_path + model_name)
        model.load_state_dict(checkpoint['model'] if 'model' in checkpoint else checkpoint)
        
        # 测试不同SNR（固定κ=9）
        snr_values = list(range(-10, 19, 2))
        snr_acc = []
        
        for snr in snr_values:
            acc = test(model, test_loader, SNR=snr, k_rician=9)
            snr_acc.append(acc)
            print(f"  SNR={snr}dB, 准确率={acc:.2f}%")
        
        left_results[train_snr] = {
            'snr_values': snr_values,
            'accuracy': snr_acc
        }
    
    # ========== 第二部分：测试不同训练κ的模型（固定SNR=5）==========
    print("\n=== 测试不同训练κ的模型（固定测试SNR=5）===")
    
    # 右图需要的模型：不同训练κ，但都是SNR=5
    # 注意：你需要有这些模型文件，如果没有需要先训练
    k_models = [
        'MIMOIB_model/Complex_MIMOIB_infoNCE_SNR5_H1_20260312_xxxxxx.pth',  # κ=1
        'MIMOIB_model/Complex_MIMOIB_infoNCE_SNR5_H5_20260312_xxxxxx.pth',  # κ=5
        'MIMOIB_model/Complex_MIMOIB_infoNCE_SNR5_H9_20260303_155117.pth',  # κ=9（已有）
    ]
    
    # 存储右图数据
    right_results = {}  # key: 训练κ, value: κ测试结果
    
    for model_name in k_models:
        if not os.path.exists(model_weights_path + model_name):
            print(f"警告: 文件不存在 {model_name}，跳过")
            continue
            
        # 从文件名提取训练κ
        if 'H1' in model_name:
            train_k = 1
        elif 'H5' in model_name:
            train_k = 5
        elif 'H9' in model_name:
            train_k = 9
        else:
            continue
            
        print(f"\n测试模型: {model_name} (训练κ={train_k})")
        
        model = Net(latent_dim).to(device)
        checkpoint = torch.load(model_weights_path + model_name)
        model.load_state_dict(checkpoint['model'] if 'model' in checkpoint else checkpoint)
        
        # 测试不同κ（固定SNR=5）
        k_values = list(range(1, 10))
        k_acc = []
        
        for k in k_values:
            acc = test(model, test_loader, SNR=5, k_rician=k)
            k_acc.append(acc)
            print(f"  κ={k}, 准确率={acc:.2f}%")
        
        right_results[train_k] = {
            'k_values': k_values,
            'accuracy': k_acc
        }
    
    # ========== 第三部分：画图 ==========
    plt.figure(figsize=(14, 5))
    
    # 左图：不同训练SNR（固定κ=9）
    plt.subplot(1, 2, 1)
    colors = {0: 'blue', 5: 'green', 10: 'orange'}
    markers = {0: 'o', 5: 's', 10: '^'}
    
    for train_snr, data in left_results.items():
        plt.plot(data['snr_values'], data['accuracy'], 
                color=colors[train_snr], 
                marker=markers[train_snr],
                linewidth=2, 
                markersize=6,
                label=f'Train SNR={train_snr}dB')
    
    plt.xlabel('Test SNR (dB)', fontsize=12)
    plt.ylabel('Classification Accuracy (%)', fontsize=12)
    plt.title('(a) Performance vs SNR (Fixed κ=9)', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.ylim(80, 100)
    plt.xlim(-10, 18)
    plt.legend(loc='lower right')
    
    # 右图：不同训练κ（固定SNR=5）
    plt.subplot(1, 2, 2)
    colors_k = {1: 'red', 5: 'purple', 9: 'brown'}
    markers_k = {1: 'o', 5: 's', 9: '^'}
    
    for train_k, data in right_results.items():
        plt.plot(data['k_values'], data['accuracy'],
                color=colors_k[train_k],
                marker=markers_k[train_k],
                linewidth=2,
                markersize=6,
                label=f'Train κ={train_k}')
    
    plt.xlabel('Test Rician Factor κ', fontsize=12)
    plt.ylabel('Classification Accuracy (%)', fontsize=12)
    plt.title('(b) Performance vs κ (Fixed SNR=5dB)', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.ylim(94, 100)
    plt.xlim(1, 9)
    plt.legend(loc='lower right')
    
    plt.suptitle('MNIST Classification over Rician Fading MIMO', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # 保存图片
    plt.savefig('snr_k_comparison.png', dpi=300, bbox_inches='tight')
    print("\n图片已保存: snr_k_comparison.png")
    plt.show()