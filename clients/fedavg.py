import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import time
import numpy as np
from matplotlib import pyplot as plt

from torch.utils.data import DataLoader
from torchvision import transforms

from config import args, logger, device
from models import Extractor, Classifier
from utils import AvgMeter, set_seed, add_gaussian_noise


class Client():

    def __init__(self, client_id, trainset, valset, testset):
        set_seed(args.seed)
        self.id = client_id
        self.testset = testset
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
            self.noise_std = args.noise_std * self.id / (args.n_clients - 1)
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
    
    def get_params1(self, models):
        params_list = []
        for model in models:
            params_list.extend(self.net[model].parameters())
        return params_list
    
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
            "EC_optimizer": self.EC_optimizer.state_dict()}
        torch.save(optim_dict, args.checkpoint_dir + "/client" + str(self.id) + ".pkl")

    def load_client(self):
        checkpoint = torch.load(args.checkpoint_dir + "/client" + str(self.id) + ".pkl")
        self.net.load_state_dict(checkpoint["net"])
        self.EC_optimizer.load_state_dict(checkpoint["EC_optimizer"])

    def init_net(self):
        set_seed(args.seed)
        ##############################################################
        # frozen all models' parameters, unfrozen when need to train #
        ##############################################################
        self.net = nn.ModuleDict()

        self.net["extractor"] = Extractor()  # E
        self.net["classifier"] = Classifier()  # C
        self.frozen_net(["extractor", "classifier"], True)
        self.EC_optimizer = optim.Adam(self.get_params(["extractor", "classifier"]), lr=args.lr,
                                       weight_decay=args.weight_decay)

        self.net.to(device)

        self.BCE_criterion = nn.BCELoss().to(device)
        self.CE_criterion = nn.CrossEntropyLoss().to(device)
        self.MSE_criterion = nn.MSELoss().to(device)
        self.COS_criterion = nn.CosineSimilarity().to(device)

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
        logger.info("Training Client %2d's EC Network Start!" % self.id)
        EC_loss_meter = AvgMeter()

        for epoch in range(args.local_epoch):
            EC_loss_meter.reset()

            self.frozen_net(["extractor", "classifier"], False)

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

            EC_loss = EC_loss_meter.get()
            EC_acc = self.local_val()
            logger.info("Client:[%2d], Epoch:[%2d], EC_loss:%2.6f, EC_acc:%2.6f" % (self.id, epoch, EC_loss, EC_acc))


    def reconstruction(self, dataset):
        dst = self.testset
        root_path = '/home/lyc'
        rel_path = os.path.join(root_path, args.checkpoint_dir + '/results')
        save_path = os.path.join(root_path, args.checkpoint_dir + '/results/test_%s'%dataset).replace('\\', '/')

        lr = 1.0
        num_dummy = 1
        Iteration = 300
        num_exp = 10

        use_cuda = torch.cuda.is_available()
        device = 'cuda' if use_cuda else 'cpu'

        tt = transforms.Compose([transforms.ToTensor()])
        tp = transforms.Compose([transforms.ToPILImage()])

        if not os.path.exists(rel_path):
            os.mkdir(rel_path)
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        for idx_net in range(num_exp):
            print('running %d|%d experiment'%(idx_net, num_exp))
            idx_shuffle = np.random.permutation(len(dst))

            for method in ['DLG']: # for method in ['DLG', 'iDLG']:
                print('%s, Try to generate %d images' % (method, num_dummy))

                criterion = nn.CrossEntropyLoss().to(device)  # 计算交叉熵损失函数
                imidx_list = []

                for imidx in range(num_dummy):
                    idx = idx_shuffle[imidx]
                    imidx_list.append(idx)
                    tmp_datum = dst[idx][0].float().to(device)
                    tmp_datum = tmp_datum.view(1, *tmp_datum.size())
                    tmp_label = torch.Tensor([dst[idx][1]]).long().to(device)
                    tmp_label = tmp_label.view(1, )
                    if imidx == 0:
                        gt_data = tmp_datum
                        gt_label = tmp_label
                    else:
                        gt_data = torch.cat((gt_data, tmp_datum), dim=0)  # tensor拼接
                        gt_label = torch.cat((gt_label, tmp_label), dim=0)


                # compute original gradient
                E = self.net["extractor"](gt_data)
                out = self.net["classifier"](E)


                # to test:
                # y = criterion(out, gt_label).requires_grad_(True)  # 计算loss
                # tensors_list = self.get_params1(["extractor", "classifier"])
                # tensors_list = [tensor.requires_grad_() for tensor in tensors_list]
                # dy_dx = torch.autograd.grad(y, tensors_list, allow_unused=True)  # 通过自动求微分得到真实梯度
                # 这一步是一个列表推导式，先从dy_dx这个Tensor中一步一步取元素出来，对原有的tensor进行克隆， 放在一个list中
                # https://blog.csdn.net/Answer3664/article/details/104417013
                # print("___________________________________________________________________")
                # print(dy_dx)
                # print("___________________________________________________________________")
                # original_dy_dx = list((_.detach().clone() for _ in dy_dx if _ is not None))

                y = criterion(out, gt_label).requires_grad_(True)
                params_list = self.get_params1(["extractor", "classifier"])
                
                print("=========================================================")
                print(params_list)
                print("=========================================================")

                for param in params_list:
                    param.requires_grad_(True)
                
                dy_dx = torch.autograd.grad(y, params_list)  # 通过自动求微分得到真实梯度
                original_dy_dx = list((_.detach().clone() for _ in dy_dx))

                # generate dummy data and label
                dummy_data = torch.randn(gt_data.size()).to(device).requires_grad_(True)   # 初始化虚拟参数
                dummy_label = torch.randn((gt_data.shape[0], 10)).to(device).requires_grad_(True)

                if method == 'DLG':
                    # LBFGS具有收敛速度快、内存开销少等优点？？？
                    optimizer = torch.optim.LBFGS([dummy_data, dummy_label], lr=lr)  # 设置优化器为拟牛顿法

                # elif method == 'iDLG':
                    # optimizer = torch.optim.LBFGS([dummy_data, ], lr=lr)
                    # predict the ground-truth label
                    # label_pred = torch.argmin(torch.sum(original_dy_dx[-2], dim=-1), dim=-1).detach().reshape((1,)).requires_grad_(False)

                history = []
                history_iters = []
                losses = []
                mses = []
                train_iters = []

                print('lr =', lr)
                for iters in range(Iteration):

                    def closure():
                        # 清空过往梯度
                        optimizer.zero_grad()
                        E1 = self.net["extractor"](dummy_data)
                        pred = self.net["classifier"](E1)
                        if method == 'DLG':
                            # 将假的预测进行softmax归一化，转换为概率
                            dummy_loss = - torch.mean(torch.sum(torch.softmax(dummy_label, -1) * torch.log(torch.softmax(pred, -1)), dim=-1))
                            # dummy_loss = criterion(pred, gt_label)
                        # elif method == 'iDLG':
                        #     dummy_loss = criterion(pred, label_pred)
                    
                        dummy_dy_dx = torch.autograd.grad(dummy_loss.requires_grad_(True), params_list, create_graph=True)

                        grad_diff = 0
                        for gx, gy in zip(dummy_dy_dx, original_dy_dx):
                            grad_diff += ((gx - gy) ** 2).sum()

                        grad_diff.backward()
                        return grad_diff                      

                    optimizer.step(closure)
                    current_loss = closure().item()
                    train_iters.append(iters)
                    losses.append(current_loss)
                    mses.append(torch.mean((dummy_data-gt_data)**2).item())


                    if iters % int(Iteration / 30) == 0:
                        current_time = str(time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime()))
                        print(current_time, iters, 'loss = %.8f, mse = %.8f' %(current_loss, mses[-1]))
                        history.append([tp(dummy_data[imidx].cpu()) for imidx in range(num_dummy)])
                        history_iters.append(iters)

                        for imidx in range(num_dummy):
                            plt.figure(figsize=(12, 8))
                            plt.subplot(3, 10, 1)
                            # plt.imshow(tp(gt_data[imidx].cpu()))
                            # 得到灰度图像
                            plt.imshow(tp(gt_data[imidx].cpu()), cmap='gray')
                            plt.title('Truth image')
                            for i in range(min(len(history), 29)):
                                plt.subplot(3, 10, i + 2)

                                if dataset == 'fmnist':
                                    plt.imshow(history[i][imidx], cmap='gray')  # 这一行是显示灰度图片的意思, 如果不是mnist数据集，将这一行改为如下
                                else:
                                    plt.imshow(history[i][imidx])
                                plt.title('iter=%d' % (history_iters[i]))
                                plt.axis('off')

                            if method == 'DLG':
                                plt.savefig('%s/client%d_DLG_on_%s_%05d.png' % (save_path, self.id, imidx_list, imidx_list[imidx]))
                                plt.close()
                            # elif method == 'iDLG':
                            #     plt.savefig('%s/iDLG_on_%s_%05d.png' % (save_path, imidx_list, imidx_list[imidx]))
                            #     plt.close()

                        if current_loss < 0.000001:  # 收敛阈值
                            break

                    if method == 'DLG':
                        if dataset == 'fmnist':
                            plt.imshow(tp(gt_data[0].cpu()), cmap='gray')
                            plt.axis('off')
                            plt.savefig('%s/client%d_origin_DLG_on_%s_%05d.png' % (save_path, self.id, imidx_list, imidx_list[0]))
                            plt.close()
                            plt.imshow(history[-1][0], cmap='gray')
                            plt.axis('off')
                            plt.savefig('%s/client%d_recovered_DLG_on_%s_%05d.png' % (save_path, self.id, imidx_list, imidx_list[0]))
                            plt.close()




                if method == 'DLG':
                    loss_DLG = losses
                    label_DLG = torch.argmax(dummy_label, dim=-1).detach().item()
                    mse_DLG = mses
                # elif method == 'iDLG':
                #    loss_iDLG = losses
                #    label_iDLG = label_pred.item()
                #    mse_iDLG = mses



            print('imidx_list:', imidx_list)
            print('loss_DLG:', loss_DLG[-1])# , 'loss_iDLG:', loss_iDLG[-1])
            print('mse_DLG:', mse_DLG[-1]) # , 'mse_iDLG:', mse_iDLG[-1])
            print('gt_label:', gt_label.detach().cpu().data.numpy(), 'lab_DLG:', label_DLG) # , 'lab_iDLG:', label_iDLG)

            print('----------------------\n\n')

