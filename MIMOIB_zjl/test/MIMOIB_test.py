import sys
sys.path.append("..")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import torchvision
import utils.channel as channel
import json
import os
from datetime import datetime

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def seed_torch(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def save_test_results(model_name, test_results, test_config, save_dir='./test/test_results'):
    """
    保存测试结果到文件中
    
    Args:
        model_name: 模型名称
        test_results: 测试结果字典
        test_config: 测试配置参数
        save_dir: 保存目录
    """
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 获取当前时间作为文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"test_results_{timestamp}.json"
    filepath = os.path.join(save_dir, filename)
    
    # 构建结果字典
    results_dict = {
        'test_info': {
            'model_name': model_name,
            'test_time': datetime.now().isoformat(),
            'timestamp': timestamp
        },
        'test_config': test_config,
        'test_results': test_results
    }
    
    # 保存到JSON文件
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(results_dict, f, indent=4, ensure_ascii=False)
    
    print(f"测试结果已保存到: {filepath}")
    
    # 同时保存一个汇总文件，追加到历史记录中
    summary_file = os.path.join(save_dir, "test_history.json")
    
    # 读取现有的历史记录
    history = []
    if os.path.exists(summary_file):
        try:
            with open(summary_file, 'r', encoding='utf-8') as f:
                history = json.load(f)
        except:
            history = []
    
    # 添加新的记录
    history.append(results_dict)
    
    # 保存更新后的历史记录
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=4, ensure_ascii=False)
    
    print(f"测试历史已更新: {summary_file}")
    
    return filepath

def load_test_history(save_dir='./test_results'):
    """
    加载并显示测试历史记录
    
    Args:
        save_dir: 保存目录
        
    Returns:
        测试历史记录列表
    """
    summary_file = os.path.join(save_dir, "test_history.json")
    
    if not os.path.exists(summary_file):
        print("没有找到测试历史记录文件")
        return []
    
    try:
        with open(summary_file, 'r', encoding='utf-8') as f:
            history = json.load(f)
        
        print(f"\n=== 测试历史记录 (共{len(history)}次测试) ===")
        for i, record in enumerate(history, 1):
            print(f"\n测试记录 #{i}:")
            test_info = record['test_info']
            print(f"  模型: {test_info['model_name']}")
            print(f"  时间: {test_info['test_time']}")
            print(f"  设备: {test_info['device']}")
            
            # 显示测试结果摘要
            results = record['test_results']
            if 'snr_accuracy' in results:
                avg_snr_acc = sum(results['snr_accuracy']) / len(results['snr_accuracy'])
                print(f"  SNR测试平均准确率: {avg_snr_acc:.2f}%")
            if 'rician_accuracy' in results:
                avg_rician_acc = sum(results['rician_accuracy']) / len(results['rician_accuracy'])
                print(f"  Rician测试平均准确率: {avg_rician_acc:.2f}%")
        
        return history
    
    except Exception as e:
        print(f"加载测试历史时出错: {e}")
        return []

def compare_test_results(model1_name, model2_name, save_dir='./test_results'):
    """
    比较两个模型的测试结果
    
    Args:
        model1_name: 第一个模型名称
        model2_name: 第二个模型名称
        save_dir: 保存目录
    """
    summary_file = os.path.join(save_dir, "test_history.json")
    
    if not os.path.exists(summary_file):
        print("没有找到测试历史记录文件")
        return
    
    try:
        with open(summary_file, 'r', encoding='utf-8') as f:
            history = json.load(f)
        
        # 查找对应模型的测试记录
        model1_record = None
        model2_record = None
        
        for record in history:
            if record['test_info']['model_name'] == model1_name:
                model1_record = record
            elif record['test_info']['model_name'] == model2_name:
                model2_record = record
        
        if not model1_record:
            print(f"没有找到模型 {model1_name} 的测试记录")
            return
        
        if not model2_record:
            print(f"没有找到模型 {model2_name} 的测试记录")
            return
        
        print(f"\n=== 模型对比 ===")
        print(f"模型1: {model1_name}")
        print(f"模型2: {model2_name}")
        
        results1 = model1_record['test_results']
        results2 = model2_record['test_results']
        
        # 比较SNR测试结果
        if 'snr_accuracy' in results1 and 'snr_accuracy' in results2:
            print(f"\nSNR测试结果对比:")
            avg1 = sum(results1['snr_accuracy']) / len(results1['snr_accuracy'])
            avg2 = sum(results2['snr_accuracy']) / len(results2['snr_accuracy'])
            print(f"  {model1_name} 平均准确率: {avg1:.2f}%")
            print(f"  {model2_name} 平均准确率: {avg2:.2f}%")
            print(f"  差值: {avg2 - avg1:+.2f}%")
        
        # 比较Rician测试结果
        if 'rician_accuracy' in results1 and 'rician_accuracy' in results2:
            print(f"\nRician测试结果对比:")
            avg1 = sum(results1['rician_accuracy']) / len(results1['rician_accuracy'])
            avg2 = sum(results2['rician_accuracy']) / len(results2['rician_accuracy'])
            print(f"  {model1_name} 平均准确率: {avg1:.2f}%")
            print(f"  {model2_name} 平均准确率: {avg2:.2f}%")
            print(f"  差值: {avg2 - avg1:+.2f}%")
        
    except Exception as e:
        print(f"比较测试结果时出错: {e}")

class MNISTDataLoader:
    """
    MNIST DataLoader class for handling data loading and preprocessing
    """
    def __init__(self, data_root='./data', train_batch_size=100, test_batch_size=100):
        """
        Initialize MNIST DataLoader
        
        Args:
            data_root: Root directory for data storage
            train_batch_size: Batch size for training data
            test_batch_size: Batch size for test data
        """
        self.data_root = data_root
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        
        # Define transforms
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        # Initialize datasets and loaders
        self._setup_datasets()
        self._setup_loaders()
    
    def _setup_datasets(self):
        """Setup MNIST datasets"""
        self.train_dataset = torchvision.datasets.MNIST(
            root=self.data_root,
            train=True,
            transform=self.transform,
            download=True
        )
        
        self.test_dataset = torchvision.datasets.MNIST(
            root=self.data_root,
            train=False,
            transform=self.transform,
            download=True
        )
    
    def _setup_loaders(self):
        """Setup data loaders"""
        self.train_loader = torch.utils.data.DataLoader(
            dataset=self.train_dataset,
            batch_size=self.train_batch_size,
            shuffle=True
        )
        
        self.test_loader = torch.utils.data.DataLoader(
            dataset=self.test_dataset,
            batch_size=self.test_batch_size,
            shuffle=False
        )
    
    def get_train_loader(self):
        """Return training data loader"""
        return self.train_loader
    
    def get_test_loader(self):
        """Return test data loader"""
        return self.test_loader

class Net(nn.Module):
    def __init__(self, K = 16):
        super(Net, self).__init__()

        self.encoder = nn.Linear(784, 512)
        self.fc_mu = nn.Linear(512, 2*K)
        self.fc_logvar = nn.Linear(512, 2*K)
        self.decoder = nn.Sequential(
                        nn.Linear(2*K,512),
                        nn.ReLU(),
                        nn.Linear(512,128),
                        nn.ReLU(),
                        nn.Linear(128,2*K),
                        nn.ReLU()
                        )
        self.classifier = nn.Sequential(
                        nn.Linear(2*K,1024),
                        nn.ReLU(),
                        nn.Linear(1024,256),
                        nn.ReLU(),
                        nn.Linear(256,10)
                        )

    def kl_divergence(self, mu, logvar):
        """
        KL(q(z|x) || p(z)), p(z) = N(0, I)
        """
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()

    
    def forward(self, x, SNR = 20, Nr = 16, Nt = 16, channel_type="rician", K_rician=10):
        h = F.relu(self.encoder(x))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std  # reparameterization trick

        kl_divergence = self.kl_divergence(mu, logvar)

        z_normalized= channel.power_normalize(z)

        if channel_type == "rician":
            r, H, noise_variance = channel.MIMO_channel_rician(z_normalized, SNR, Nr, Nt, K_rician=K_rician)
        elif channel_type == "awgn":
            r, noise_variance = channel.MIMO_channel_AWGN(z_normalized, SNR, Nr, Nt)
        else:
            raise ValueError(f"Unknown channel_type: {channel_type}")

        y = self.decoder(r)
        out = self.classifier(y)
        return F.log_softmax(out, dim=1), z, r, y, kl_divergence

def test(model, data_loader, SNR = 20, Nr = 16, Nt = 16, epoch = 1, stage1_epoch = 0, k_rician = 10):
    model.eval()
    accuracy = 0
    correct = 0
    global_epoch = 0
    with torch.no_grad():
        for idx, (images, labels) in enumerate(data_loader):

            global_epoch += 1
            x, y = images.to(device), labels.to(device)
            x = x.view(x.size(0), -1).to(torch.float32)

            # Forward pass through decoder
            logit, z, r, z_hat, loss_kl = model(x, SNR, Nr, Nt, K_rician=k_rician)
            pred = logit.argmax(dim=1, keepdim=True)
            correct += pred.eq(y.view_as(pred)).sum().item()
    accuracy = 100. * correct / len(data_loader.dataset)

    return accuracy

def main_test_SNR(model, test_loader, SNR):
    model.eval()
    acc_all = []

    PSNRs = list(range(-10, SNR, 2))

    for psnr in PSNRs:
        accuracy = test(model, test_loader, psnr)
        # print('Noise level:', psnr, 'Test Accuracy:', accuracy)
        acc_all.append(accuracy)
    print('[Acc vs SNR] Test All Accuracy:', acc_all)
    
    return {
        'snr_values': PSNRs,
        'snr_accuracy': acc_all
    }

def main_test_Rician(model, test_loader, SNR):
    model.eval()  # 设置为评估模式
    acc_all = []

    K = list(range(1, 10))

    for k in K:
        accuracy = test(model, test_loader, 10, k_rician=k)
        # print('Rician factor k:', k, 'Test Accuracy:', accuracy)
        acc_all.append(accuracy)
    print('[Acc vs Rician factor k] Test All Accuracy:', acc_all)
    
    return {
        'rician_k_values': K,
        'rician_accuracy': acc_all
    }

if __name__ == "__main__":
    input_dim = 784  # Example for MNIST (28x28 images flattened)
    latent_dim = 16  # Latent dimension size
    output_dim = 784  # Same as input_dim for reconstruction
    batch_size = 512
    test_SNR = 20
    K_rician = 10
    # model_weights_path = '/root/zhujl/MIMOIB/MIMO_IB_Compare/MIMOIB_model/'  # MIMOIB model
    # model_weights_list = ['Complex_MIMOIB_infoNCE_SNR5_H1_20251023_205158.pth',
    #                       'Complex_MIMOIB_infoNCE_SNR5_H9_20251023_221212.pth']
    
    model_weights_path = '/home/wen/project2/MIMO_IB_Compare/'  # 修改这里
    model_weights_list = [
        'MIMOIB_model/Complex_MIMOIB_infoNCE_SNR5_H9_20260303_155117.pth',  # 使用你已有的文件
        # 如果有第二个文件可以再加，没有就只留一个
    ]

    # Initialize MNIST data loader
    # 改为
    mnist_loader = MNISTDataLoader(
        data_root='/home/wen/project2/data',  # 你的实际数据路径
        train_batch_size=batch_size,
        test_batch_size=100
    )


    # Get data loaders
    train_loader = mnist_loader.get_train_loader()
    test_loader = mnist_loader.get_test_loader()

    # test
    for model_name in model_weights_list:
        model = Net(latent_dim).to(device)
        model.load_state_dict(torch.load(model_weights_path + model_name)['model'])
        # model = nn.DataParallel(model)
        print('Testing model name:', model_name)

        # 运行测试并收集结果
        rician_results = main_test_Rician(model, test_loader, test_SNR)
        snr_results = main_test_SNR(model, test_loader, test_SNR)
        
        # 合并测试结果
        test_results = {
            **rician_results,
            **snr_results
        }
        
        # 测试配置信息
        test_config = {
            'latent_dim': latent_dim,
            'batch_size': batch_size,
            'test_SNR': test_SNR,
            'K_rician': K_rician,
            'model_path': model_weights_path + model_name
        }
        
        # 保存测试结果
        save_test_results(model_name, test_results, test_config)
        
        print(f"模型 {model_name} 测试完成")
    
