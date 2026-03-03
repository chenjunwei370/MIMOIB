import torch
import json
import os
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np


def save_training_params(args, model_info, save_dir='./training_params'):
    """
    保存每次训练的参数到文件中
    
    Args:
        args: 训练参数配置对象
        model_info: 模型相关信息（如模型名称、架构等）
        save_dir: 保存目录
    """
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 获取当前时间作为文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"training_params_{timestamp}.json"
    filepath = os.path.join(save_dir, filename)
    
    # 将参数转换为字典格式
    params_dict = {
        'timestamp': datetime.now().isoformat(),
        'training_parameters': {
            'epochs': getattr(args, 'epochs', None),
            'batch_size': getattr(args, 'batch_size', None),
            'learning_rate': getattr(args, 'lr', None),
            'log_interval': getattr(args, 'log_interval', None),
            'train_SNR': getattr(args, 'train_SNR', None),
            'test_SNR': getattr(args, 'test_SNR', None),
            'Nr': getattr(args, 'Nr', None),
            'Nt': getattr(args, 'Nt', None),
            'K_rician': getattr(args, 'K_rician', None),
        },
        'model_info': model_info,
        'device': str(torch.cuda.get_device_name(0)) if torch.cuda.is_available() else 'CPU',
        'torch_version': torch.__version__,
    }
    
    history_params_dict = {
        'timestamp': datetime.now().isoformat(),
        'training_parameters': {
            'epochs': getattr(args, 'epochs', None),
            'batch_size': getattr(args, 'batch_size', None),
            'learning_rate': getattr(args, 'lr', None),
            'log_interval': getattr(args, 'log_interval', None),
            'train_SNR': getattr(args, 'train_SNR', None),
            'test_SNR': getattr(args, 'test_SNR', None),
            'Nr': getattr(args, 'Nr', None),
            'Nt': getattr(args, 'Nt', None),
            'K_rician': getattr(args, 'K_rician', None),
        },
    }

    # 保存到JSON文件
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(params_dict, f, indent=4, ensure_ascii=False)
    
    print(f"训练参数已保存到: {filepath}")
    
    # 同时保存一个汇总文件，追加到历史记录中
    summary_file = os.path.join(save_dir, "training_history.json")
    
    # 读取现有的历史记录
    history = []
    if os.path.exists(summary_file):
        try:
            with open(summary_file, 'r', encoding='utf-8') as f:
                history = json.load(f)
        except:
            history = []
    
    # 添加新的记录
    history.append(history_params_dict)
    
    # 保存更新后的历史记录
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=4, ensure_ascii=False)
    
    print(f"训练历史已更新: {summary_file}")
    
    return filepath

def load_training_history(save_dir='./training_params'):
    """
    加载并显示训练历史记录
    
    Args:
        save_dir: 保存目录
        
    Returns:
        训练历史记录列表
    """
    summary_file = os.path.join(save_dir, "training_history.json")
    
    if not os.path.exists(summary_file):
        print("没有找到训练历史记录文件")
        return []
    
    try:
        with open(summary_file, 'r', encoding='utf-8') as f:
            history = json.load(f)
        
        print(f"\n=== 训练历史记录 (共{len(history)}次训练) ===")
        for i, record in enumerate(history, 1):
            print(f"\n训练记录 #{i}:")
            print(f"  时间: {record['timestamp']}")
            params = record['training_parameters']
            print(f"  参数: epochs={params['epochs']}, batch_size={params['batch_size']}, lr={params['learning_rate']}")
            print(f"       train_SNR={params['train_SNR']}, K_rician={params['K_rician']}")
            print(f"       Nr={params['Nr']}, Nt={params['Nt']}")
        
        return history
    
    except Exception as e:
        print(f"加载训练历史时出错: {e}")
        return []

def compare_training_params(file1, file2, save_dir='./training_params'):
    """
    比较两次训练的参数差异
    
    Args:
        file1: 第一个参数文件名
        file2: 第二个参数文件名
        save_dir: 保存目录
    """
    try:
        with open(os.path.join(save_dir, file1), 'r', encoding='utf-8') as f:
            params1 = json.load(f)
        
        with open(os.path.join(save_dir, file2), 'r', encoding='utf-8') as f:
            params2 = json.load(f)
        
        print(f"\n=== 参数对比 ===")
        print(f"文件1: {file1} ({params1['timestamp']})")
        print(f"文件2: {file2} ({params2['timestamp']})")
        
        params1_train = params1['training_parameters']
        params2_train = params2['training_parameters']
        
        print(f"\n参数差异:")
        for key in params1_train:
            if params1_train[key] != params2_train.get(key):
                print(f"  {key}: {params1_train[key]} -> {params2_train.get(key)}")
        
    except Exception as e:
        print(f"比较参数时出错: {e}")


def plot_loss_curves(loss_history, save_path='./loss_curves.png'):
    """
    绘制训练过程中的损失曲线
    
    Args:
        loss_history: 包含损失记录的字典
        save_path: 保存图片的路径
    """
    plt.figure(figsize=(15, 10))
    
    # 创建2x2的子图布局
    plt.subplot(2, 2, 1)
    plt.plot(loss_history['epochs'], loss_history['total_loss'], 'b-', linewidth=2, label='Total Loss')
    plt.title('Total Loss vs Epochs', fontsize=14, fontweight='bold')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.subplot(2, 2, 2)
    plt.plot(loss_history['epochs'], loss_history['rec_loss'], 'r-', linewidth=2, label='Reconstruction Loss')
    plt.title('Reconstruction Loss vs Epochs', fontsize=14, fontweight='bold')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.subplot(2, 2, 3)
    plt.plot(loss_history['epochs'], loss_history['loss_InfoNCE'], 'g-', linewidth=2, label='InfoNCE Loss')
    plt.title('InfoNCE Loss vs Epochs', fontsize=14, fontweight='bold')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.subplot(2, 2, 4)
    plt.plot(loss_history['epochs'], loss_history['loss_kl'], 'm-', linewidth=2, label='KL Divergence Loss')
    plt.title('KL Divergence Loss vs Epochs', fontsize=14, fontweight='bold')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Loss curves saved to: {save_path}")
    
    # 也创建一个所有损失在一个图中的版本
    # plt.figure(figsize=(12, 6))
    # plt.plot(loss_history['epochs'], loss_history['total_loss'], 'b-', linewidth=2, label='Total Loss')
    # plt.plot(loss_history['epochs'], loss_history['rec_loss'], 'r-', linewidth=2, label='Reconstruction Loss')
    # plt.plot(loss_history['epochs'], loss_history['loss_InfoNCE'], 'g-', linewidth=2, label='InfoNCE Loss')
    # plt.plot(loss_history['epochs'], loss_history['loss_kl'], 'm-', linewidth=2, label='KL Divergence Loss')
    
    # plt.title('All Loss Curves vs Epochs', fontsize=16, fontweight='bold')
    # plt.xlabel('Epochs', fontsize=12)
    # plt.ylabel('Loss', fontsize=12)
    # plt.legend(fontsize=12)
    # plt.grid(True, alpha=0.3)
    # plt.tight_layout()
    
    # combined_save_path = save_path.replace('.png', '_combined.png')
    # plt.savefig(combined_save_path, dpi=300, bbox_inches='tight')
    # plt.show()
    # print(f"Combined loss curves saved to: {combined_save_path}")
