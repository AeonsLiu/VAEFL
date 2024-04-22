import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from config import args, logger, device
from models import Extractor, Classifier, VAE_Encoder, VAE_Decoder  # 引入VAE相关类
from utils import AvgMeter, set_seed, add_gaussian_noise


class Client():

    def __init__(self, client_id, trainset, valset, testset):
        set_seed(args.seed)
        self.id = client_id
        self.trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=0,
                                      pin_memory=True)
        self.valloader = DataLoader(valset, batch_size=args.batch_size * 10, shuffle=False, num_workers=0,
                                    pin_memory=False)
        self.testloader = DataLoader(testset, batch_size=args.batch_size * 10, shuffle=False, num_workers=0,
                                     pin_memory=False)
        self.train_size = len(trainset)
        self.val_size = len(valset)
        self.test_size = len(testset)
        if args.add_noise:
            self.noise_std = args.noise_std
            logger.info("client:%2d, train_size:%4d, val_size:%4d, test_size:%4d, noise_std:%2.6f"
                        % (self.id, self.train_size, self.val_size, self.test_size, self.noise_std))
        else:
            logger.info("client:%2d, train_size:%4d, val_size:%4d, test_size:%4d"
                        % (self.id, self.train_size, self.val_size, self.test_size))
        self.init_net()

    def get_params(self, models):
        params = []
        for model in models:
            params.append({"params": self.net[model].parameters()})
        return params

    def frozen_net(self, models, frozen):
        for model in models:
            for param in self.net[model].parameters():
                param.requires_grad = not frozen
            if frozen:
                self.net[model].eval()
            else:
                self.net[model].train()

    def save_client(self):
        optim_dict = {
            "net": self.net.state_dict(),
            "EC_optimizer": self.EC_optimizer.state_dict(),
            "D_optimizer": self.D_optimizer.state_dict(),
            "G_optimizer": self.G_optimizer.state_dict(),
        }
        torch.save(optim_dict, args.checkpoint_dir + "/client" + str(self.id) + ".pkl")

    def load_client(self):
        checkpoint = torch.load(args.checkpoint_dir + "/client" + str(self.id) + ".pkl")
        self.net.load_state_dict(checkpoint["net"])
        self.EC_optimizer.load_state_dict(checkpoint["EC_optimizer"])
        self.D_optimizer.load_state_dict(checkpoint["D_optimizer"])
        self.G_optimizer.load_state_dict(checkpoint["G_optimizer"])

    def init_net(self):
        set_seed(args.seed)
        ##############################################################
        # frozen all models' parameters, unfrozen when need to train #
        ##############################################################
        self.net = nn.ModuleDict()

        self.net["extractor"] = Extractor()  # E
        self.net["classifier"] = Classifier()  # C
        self.net["VAE_Encoder"] = VAE_Encoder()  # VAE 编码器
        self.net["VAE_Decoder"] = VAE_Decoder()  # VAE 解码器
        self.frozen_net(["extractor", "classifier", "VAE_Encoder", "VAE_Decoder"], True)
        self.EC_optimizer = optim.Adam(self.get_params(["extractor", "classifier"]), lr=args.lr, weight_decay=args.weight_decay)
        # 添加VAE的优化器
        self.VAE_optimizer = optim.Adam(self.get_params(["VAE_Encoder", "VAE_Decoder"]), lr=args.lr, weight_decay=args.weight_decay)

        self.net.to(device)

        self.BCE_criterion = nn.BCELoss().to(device)
        self.CE_criterion = nn.CrossEntropyLoss().to(device)
        self.MSE_criterion = nn.MSELoss().to(device)
        self.COS_criterion = nn.CosineSimilarity().to(device)

    # def compute_gan_acc(self):
    #     correct, total = 0, 0
    #     with torch.no_grad():
    #         for batch in range(100):
    #             y = torch.randint(0, args.num_classes, (args.batch_size,)).to(device)
    #             z = torch.randn(args.batch_size, args.noise_dim, 1, 1).to(device)
    #             feat = self.net["generator"](z, y)
    #             pred = self.net["classifier"](feat)
    #             correct += torch.sum((torch.argmax(pred, dim=1) == y).float())
    #             total += args.batch_size
    #     return (correct / total).item()

    def compute_vae_reconstruction_error(self):
        mse_loss = nn.MSELoss()
        total_loss = 0.0
        total_count = 0

        with torch.no_grad():
            for x, _ in self.testloader:
                x = x.to(device)
                encoded, _ = self.net["VAE_Encoder"](x)
                reconstructed = self.net["VAE_Decoder"](encoded)
                loss = mse_loss(reconstructed, x)
                total_loss += loss.item() * x.size(0)
                total_count += x.size(0)

        average_loss = total_loss / total_count
        return average_loss
    
    def local_test(self):
        correct, total = 0, 0
        with torch.no_grad():
            for batch, (x, y) in enumerate(self.testloader):
                if args.add_noise:
                    x += add_gaussian_noise(x, mean=0., std=self.noise_std)
                x = x.to(device)
                y = y.to(device)
                feat = self.net["extractor"](x)
                pred = self.net["classifier"](feat)
                correct += torch.sum((torch.argmax(pred, dim=1) == y).float())
                total += x.size(0)
        return (correct / total).item()

    def local_val(self):
        correct, total = 0, 0
        with torch.no_grad():
            for batch, (x, y) in enumerate(self.valloader):
                if args.add_noise:
                    x += add_gaussian_noise(x, mean=0., std=self.noise_std)
                x = x.to(device)
                y = y.to(device)
                feat = self.net["extractor"](x)
                pred = self.net["classifier"](feat)
                correct += torch.sum((torch.argmax(pred, dim=1) == y).float())
                total += x.size(0)
        return (correct / total).item()

    def local_train(self, current_round):
        set_seed(args.seed)
        logger.info("Training Client %2d's Network Start!" % self.id)
        EC_loss_meter = AvgMeter()
        VAE_loss_meter = AvgMeter()

        # 解冻所有需要训练的网络部分
        self.frozen_net(["extractor", "classifier"], False)

        for epoch in range(args.local_epoch):
            EC_loss_meter.reset()
            VAE_loss_meter.reset()

            # 特征提取器和分类器的训练
            for batch, (x, y) in enumerate(self.trainloader):
                if args.add_noise:
                    x += add_gaussian_noise(x, mean=0., std=self.noise_std)
                x = x.to(device)
                y = y.to(device)

                self.EC_optimizer.zero_grad()
                E = self.net["extractor"](x)
                EC = self.net["classifier"](E)
                EC_loss = self.CE_criterion(EC, y)
                EC_loss.backward()
                self.EC_optimizer.step()
                EC_loss_meter.update(EC_loss.item())
        self.frozen_net(["extractor", "classifier"], True)
        
        
        
        self.frozen_net(["VAE_Encoder", "VAE_Decoder"], False)
        for epoch in range(args.local_epoch):
            VAE_loss_meter.reset()
            # VAE的训练
            for batch, (x, _) in enumerate(self.trainloader):
                x = x.to(device)
                E = self.net["extractor"](x)

                self.VAE_optimizer.zero_grad()
                encoded_mu, encoded_logvar = self.net["VAE_Encoder"](E)
                std = torch.exp(0.5 * encoded_logvar)
                eps = torch.randn_like(std)
                encoded = encoded_mu + eps * std
                reconstructed = self.net["VAE_Decoder"](encoded)
                recon_loss = self.MSE_criterion(reconstructed, E)
                kl_loss = -0.5 * torch.sum(1 + encoded_logvar - encoded_mu.pow(2) - encoded_logvar.exp())
                VAE_loss = recon_loss + kl_loss
                VAE_loss.backward()
                self.VAE_optimizer.step()
                VAE_loss_meter.update(VAE_loss.item())

        self.frozen_net(["VAE_Encoder", "VAE_Decoder"], True)
        
        EC_loss = EC_loss_meter.get()
        VAE_loss = VAE_loss_meter.get()
        EC_acc = self.local_val()
        logger.info("Client:[%2d], Epoch:[%2d], EC_loss:%2.6f, VAE_loss:%2.6f, EC_acc:%2.6f" % (
                self.id, epoch, EC_loss, VAE_loss, EC_acc))
