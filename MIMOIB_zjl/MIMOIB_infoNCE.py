import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
import torchvision
import utils.channel as channel
import utils.saving as saving
import copy
import time
import os
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def seed_torch(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


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
    

class ProjectionHead(nn.Module):
    def __init__(self, latent_dim=64, proj_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, 128)
        self.fc2 = nn.Linear(128, proj_dim)

    def forward(self, z):
        return F.normalize(self.fc2(F.relu(self.fc1(z))), dim=-1)
    

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
    

def info_nce(z1, z2, temperature=0.1):
    """
    InfoNCE 对比损失
    z1, z2: [batch_size, proj_dim]
    """
    batch_size = z1.size(0)
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)

    logits = torch.matmul(z1, z2.T) / temperature  # [B, B]
    labels = torch.arange(batch_size).long().to(z1.device)  
    loss = F.cross_entropy(logits, labels)
    return loss

def info_nce_lower(z1, z2, temperature=0.05):
    """
    InfoNCE 下界（用于最大化互信息 I_lower)
    z1, z2: [batch_size, proj_dim]
    """
    batch_size = z1.size(0)
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)

    logits = torch.matmul(z1, z2.T) / temperature  # [B, B]
    labels = torch.arange(batch_size, device=z1.device)
    loss = F.cross_entropy(logits, labels)
    return loss


def info_upper(z1, z2, temperature=0.05):
    """
    Leave-one-out 上界 (Eq.13)
    用于最小化互信息 I_upper
    z1, z2: [batch_size, proj_dim]
    """
    batch_size = z1.size(0)
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)

    logits = torch.matmul(z1, z2.T) / temperature  # [B, B]
    
    
    mask = torch.eye(batch_size, dtype=torch.bool, device=z1.device)
    logits_wo_diag = logits.masked_fill(mask, float('-inf'))  # mask正样本

    
    log_num = torch.diag(logits)  # 正样本 logit
    log_denom = torch.logsumexp(logits_wo_diag, dim=1)  # 负样本 (K-1)
    
    i_upper = (log_num - log_denom).mean()
    
    return -i_upper  

# Training the model
def train(model, data_loader, test_loader, args):
    # 保存训练参数
    model_info = {
        'model_type': 'MIMO_IB_InfoNCE',
        'architecture': 'VAE with MIMO channel',
        'latent_dim': 32,
        'proj_dim': 128,
        'encoder_hidden': 512,
        'decoder_layers': [512, 128, 32],
        'classifier_layers': [1024, 256, 10]
    }
    
    params_file = saving.save_training_params(args, model_info)

    latent_dim = 32
    proj_dim = 128

    proj_x = ProjectionHead(latent_dim=784, proj_dim=proj_dim).to(device)
    proj_z = ProjectionHead(latent_dim=latent_dim, proj_dim=proj_dim).to(device)
    proj_z_hat = ProjectionHead(latent_dim=latent_dim, proj_dim=proj_dim).to(device)
    proj_z_prime = ProjectionHead(latent_dim=latent_dim, proj_dim=proj_dim).to(device)

    optimizer_model = torch.optim.Adam(
    list(model.parameters()) + list(proj_x.parameters()) + list(proj_z.parameters()) +
    list(proj_z_hat.parameters()) + list(proj_z_prime.parameters()),
    lr=1e-3
    )
    # optimizer_model = optim.Adam(model.parameters(), lr=args.lr, weight_decay = 5e-5)
    # scheduler = StepLR(optimizer, step_size=30, gamma=0.5)

    loss_history = {
        'total_loss': [],
        'rec_loss': [],
        'loss_InfoNCE': [],
        'loss_kl': [],
        'epochs': []
    }

    for epoch in range(0, args.epochs):
        model.train()

        epoch_total_loss = 0
        epoch_rec_loss = 0
        epoch_InfoNCE_loss = 0
        epoch_kl_loss = 0
        num_batches = 0

        for idx, (images, labels) in enumerate(data_loader):
            x, y = images.to(device), labels.to(device)
            x = x.view(x.size(0), -1).to(torch.float32)


            logit, z, r, z_hat, loss_kl = model(x, args.train_SNR, args.Nr, args.Nt, K_rician=args.K_rician)

            # Compute reconstruction loss
            rec_loss = F.cross_entropy(logit, y)

            x_proj = proj_x(x)
            z_proj = proj_z(z)
            z_hat_proj = proj_z_hat(r)
            z_prime_proj = proj_z_prime(z_hat)

            #infoNCE loss
            loss_kl = info_upper(x_proj, z_hat_proj)
            loss_pos = info_nce_lower(z_proj, z_hat_proj)
            loss_neg = info_upper(z_hat_proj, z_prime_proj)
            loss_InfoNCE = loss_pos + loss_neg

            if epoch < 5:
                total_loss = rec_loss
            else:
                anneal_ratio = min(1,(epoch - 5)/10)
                total_loss = rec_loss + (1e-3 * loss_InfoNCE + 1e-3 * loss_kl) * anneal_ratio  # MIMOIB
            

            if torch.isnan(total_loss):
                print(f"NaN detected in total_loss at epoch {epoch + 1}, batch {idx + 1}. Exiting training.")
                return loss_history
            
            # Backpropagate and update parameters
            optimizer_model.zero_grad()

            total_loss.backward()
            optimizer_model.step()

            # 累积这个epoch的损失
            epoch_total_loss += total_loss.item()
            epoch_rec_loss += rec_loss.item()
            epoch_InfoNCE_loss += loss_InfoNCE.item()
            epoch_kl_loss += loss_kl.item()
            num_batches += 1

        # 计算平均损失并记录
        avg_total_loss = epoch_total_loss / num_batches
        avg_rec_loss = epoch_rec_loss / num_batches
        avg_InfoNCE_loss = epoch_InfoNCE_loss / num_batches
        avg_kl_loss = epoch_kl_loss / num_batches

        loss_history['epochs'].append(epoch + 1)
        loss_history['total_loss'].append(avg_total_loss)
        loss_history['rec_loss'].append(avg_rec_loss)
        loss_history['loss_InfoNCE'].append(avg_InfoNCE_loss)
        loss_history['loss_kl'].append(avg_kl_loss)

        if (epoch + 1) % args.log_interval == 0:
            print(f"Testing at epoch {epoch + 1}...")
            test(model, test_loader, 30, args.Nr, args.Nt, args.K_rician)


        nowtime = time.strftime('%Y-%m-%d %H:%M:%S')
        print('Time:' + str(nowtime))
        print(f'Epoch {epoch + 1}/{args.epochs}, SNR:{args.train_SNR}, H: {args.K_rician}, Total loss: {avg_total_loss:.4f}, Rec loss: {avg_rec_loss:.4f}, InfoNCE loss: {avg_InfoNCE_loss:.4f}, KL loss: {avg_kl_loss:.4f}')
    

    saved_model = copy.deepcopy(model.state_dict())
    
    # 保存模型文件名包含时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f'Complex_MIMOIB_infoNCE_SNR{args.train_SNR}_H{args.K_rician}_{timestamp}.pth'
    model_path = f'./MIMO_IB_Compare/MIMOIB_model/{model_filename}'
    
    # 确保保存目录存在
    os.makedirs('./MIMO_IB_Compare/MIMOIB_model', exist_ok=True)
    
    # 保存模型和训练信息
    torch.save({
        'model': saved_model,
        'training_params': {
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'lr': args.lr,
            'train_SNR': args.train_SNR,
            'test_SNR': args.test_SNR,
            'Nr': args.Nr,
            'Nt': args.Nt,
            'K_rician': args.K_rician,
        },
        'loss_history': loss_history,
        'timestamp': timestamp,
        'params_file': params_file
    }, model_path)
    
    print(f"模型已保存到: {model_path}")

    fig_filename = f'Complex_MIMOIB_infoNCE_SNR{args.train_SNR}_H{args.K_rician}_{timestamp}.png'
    fig_path = f'./MIMO_IB_Compare/MIMOIB_model/{fig_filename}'
    saving.plot_loss_curves(loss_history, fig_path)



def test(model, data_loader, SNR = 20, Nr = 16, Nt = 16, k_rician = 10):
    model.eval()  # 设置为评估模式
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
    print('accuracy:', accuracy)

    return accuracy

def main_test(model, test_loader, SNR):
    model.eval()  # 设置为评估模式
    acc_all = []

    PSNRs = list(range(-10, SNR, 2))

    for psnr in PSNRs:
        accuracy = test(model, test_loader, psnr)
        print('Noise level:', psnr, 'Test Accuracy:', accuracy)
        acc_all.append(accuracy)
    print('Test All Accuracy:', acc_all)

def main_test_k(model, test_loader, SNR):
    model.eval()  # 设置为评估模式
    acc_all = []

    K = list(range(0, 15))

    for k in K:
        accuracy = test(model, test_loader, 10, k_rician=k)
        print('Rician factor k:', k, 'Test Accuracy:', accuracy)
        acc_all.append(accuracy)
    print('Test All Accuracy:', acc_all)


if __name__ == "__main__":
    seed_torch(42)

    input_dim = 784  # Example for MNIST (28x28 images flattened)
    latent_dim = 16  # Latent dimension size
    output_dim = 784  # Same as input_dim for reconstruction

    # Create configuration class to replace wandb.config
    class Config:
        def __init__(self):
            self.epochs = 200
            self.batch_size = 512  
            self.lr = 1e-3
            self.log_interval = 10
            self.train_SNR = 5
            self.test_SNR = 20
            self.Nr = 16
            self.Nt = 16
            self.K_rician = 1
    
    config = Config()

    # Initialize MNIST data loader
    mnist_loader = MNISTDataLoader(
        data_root='./data',
        train_batch_size=config.batch_size,
        test_batch_size=100
    )
    
    # Get data loaders
    train_loader = mnist_loader.get_train_loader()
    test_loader = mnist_loader.get_test_loader()


    # Initialize models
    model = Net(latent_dim).to(device)
    
    # model.load_state_dict(torch.load('./MIMO_IB_Compare/Complex_MIMOIB_OneStep_Rician_SNR10_H1_1e-2.pth')['model'])

    # Train the model
    train(model, train_loader, test_loader, config)

    # Test the model
    main_test(model, test_loader, config.test_SNR)

    main_test_k(model, test_loader, config.test_SNR)
