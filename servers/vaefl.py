import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from config import args, logger, device
from models import Classifier, VAE_Encoder, VAE_Decoder  # 引入VAE相关类
from utils import AvgMeter, add_gaussian_noise, set_seed


class Server():

    def __init__(self, clients):
        self.clients = clients

        self.weights = []
        for client in clients:
            self.weights.append(client.train_size)
        self.weights = np.array(self.weights) / np.sum(self.weights)
        logger.info("client weights: %s" % str(self.weights.tolist()))

        self.init_net()

    def get_params(self, models):
        params = []
        for model in models:
            params.append({"params": self.global_net[model].parameters()})
        return params

    def frozen_net(self, models, frozen):
        for model in models:
            for param in self.global_net[model].parameters():
                param.requires_grad = not frozen
            if frozen:
                self.global_net[model].eval()
            else:
                self.global_net[model].train()

    def init_net(self):
        ##############################################################
        # frozen all models' parameters, unfrozen when need to train #
        ##############################################################
        set_seed(args.seed)
        self.global_net = nn.ModuleDict()

        self.global_net["VAE_Encoder"] = VAE_Encoder()  # VAE 编码器
        self.global_net["VAE_Decoder"] = VAE_Decoder()  # VAE 解码器
        self.global_net["classifier"] = Classifier()
        self.frozen_net(["VAE_Encoder", "VAE_Decoder", "classifier"], True)
        self.VAE_optimizer = optim.Adam(self.get_params(["VAE_Encoder", "VAE_Decoder", "classifier"]), lr=args.lr)

        self.global_net.to(device)

        self.KL_criterion = nn.KLDivLoss(reduction="batchmean").to(device)
        self.CE_criterion = nn.CrossEntropyLoss().to(device)
        
    def load_client(self):
        for i in range(len(self.clients)):
            checkpoint = torch.load(args.checkpoint_dir + "/client" + str(i) + ".pkl")
            self.clients[i].net.load_state_dict(checkpoint["net"])
            self.clients[i].EC_optimizer.load_state_dict(checkpoint["EC_optimizer"])
            # 更新为使用 VAE 相关的优化器
            self.clients[i].VAE_optimizer.load_state_dict(checkpoint["VAE_optimizer"])

    def save_client(self):
        for i in range(len(self.clients)):
            optim_dict = {
                "net": self.clients[i].net.state_dict(),
                "EC_optimizer": self.clients[i].EC_optimizer.state_dict(),
                # 更新为保存 VAE 相关的优化器状态
                "VAE_optimizer": self.clients[i].VAE_optimizer.state_dict()
            }
            torch.save(optim_dict, args.checkpoint_dir + "/client" + str(i) + ".pkl")


    def receive(self, models):

        for model in models:
            avg_param = {}
            params = []
            for client in self.clients:
                params.append(client.net[model].state_dict())

            for key in params[0].keys():
                avg_param[key] = params[0][key] * self.weights[0]
                for idx in range(1, len(self.clients)):
                    avg_param[key] += params[idx][key] * self.weights[idx]
            self.global_net[model].load_state_dict(avg_param)

    def send(self, models):
        for model in models:
            global_param = self.global_net[model].state_dict()
            for client in self.clients:
                client.net[model].load_state_dict(global_param)

    def compute_aggregate_acc(self):
        correct, total = 0, 0
        with torch.no_grad():
            for client in self.clients:
                for x, y in client.valloader:
                    if args.add_noise:
                        x = add_gaussian_noise(x, mean=0., std=client.noise_std)
                    x = x.to(device)
                    y = y.to(device)
                    feat = client.net["extractor"](x)
                    pred = self.global_net["classifier"](feat)
                    correct += torch.sum((torch.argmax(pred, dim=1) == y).float())
                    total += x.size(0)
        return (correct / total).item()

    def global_train(self):
        set_seed(args.seed)

        logger.info("Training Server's Network Start!")
        distill_loss_meter = AvgMeter()

        after_average_acc = self.compute_aggregate_acc()
        logger.info("after average client acc:%2.6f" % after_average_acc)

        # 冻结网络时，应包括VAE组件
        self.frozen_net(["VAE_Encoder", "VAE_Decoder", "classifier"], False)

        for epoch in range(args.global_epoch):
            distill_loss_meter.reset()

            for batch in range(args.global_iter_per_epoch):
                # y = torch.randint(0, args.num_classes, (args.batch_size,)).to(device)
                latent_dim = 20
                num_samples = 8
                x = torch.randn(num_samples, latent_dim).to(device)
                # x = torch.randn(args.batch_size, args.noise_dim, 1, 1).to(device)

                self.VAE_optimizer.zero_grad()

                # 使用VAE进行特征提取和重构
                # global_encoded, _ = self.global_net["VAE_Encoder"](x)
                global_reconstructed = self.global_net["VAE_Decoder"](x)

                global_pred = self.global_net["classifier"](global_reconstructed)
                q = torch.log_softmax(global_pred, -1)
                p = 0
                for i, client in enumerate(self.clients):
                    # local_encoded, _ = client.net["VAE_Encoder"](x)
                    local_reconstructed = client.net["VAE_Decoder"](x)
                    local_pred = client.net["classifier"](local_reconstructed)
                    p += self.weights[i] * local_pred
                p = torch.softmax(p, -1).detach()
                distill_loss = self.KL_criterion(q, p)

                distill_loss.backward()
                self.VAE_optimizer.step()
                distill_loss_meter.update(distill_loss.item())

            distill_loss = distill_loss_meter.get()
            logger.info("Server Epoch:[%2d], distill_loss:%2.6f" % (epoch, distill_loss))

        self.frozen_net(["VAE_Encoder", "VAE_Decoder", "classifier"], True)

        after_distill_acc = self.compute_aggregate_acc()
        logger.info("after distill client acc:%2.6f" % after_distill_acc)
