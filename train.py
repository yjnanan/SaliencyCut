import numpy as np
import torch
import torch.nn.functional as F
import argparse
import os

from dataloaders.dataloader import initDataloader
from modeling.net import Model
from tqdm import tqdm
from sklearn.metrics import average_precision_score, roc_auc_score
from modeling.layers import build_criterion
import random
from torch.autograd import Variable
from modeling.saliency import get_bspline_kernel, rescale_intensity
import torchvision.transforms.functional as TF

import matplotlib.pyplot as plt

class Rotation:
    def __init__(self, angle):
        self.angle = angle

    def __call__(self, x):
        return TF.rotate(x, self.angle)

class Trainer(object):

    def __init__(self, args):
        self.args = args
        # Define Dataloader
        kwargs = {'num_workers': args.workers}
        self.train_loader, self.test_loader = initDataloader.build(args, **kwargs)

        self.model = Model(args, backbone=self.args.backbone)

        if self.args.pretrain_dir != None:
            self.model.load_state_dict(torch.load(self.args.pretrain_dir))
            print('Load pretrain weight from: ' + self.args.pretrain_dir)

        self.criterion = build_criterion(args.criterion)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0002, weight_decay=1e-5)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)

    def generate_target(self, target, eval=False):
        targets = list()
        if eval:
            targets.append(target == 0)
            targets.append(target)
            return targets
        else:
            targets.append(target == 0)
            targets.append(target != 0)
        return targets

    def training(self, epoch):
        train_loss = 0.0
        class_loss = list()
        for j in range(self.args.total_heads):
            class_loss.append(0.0)
        self.model.train()
        tbar = tqdm(self.train_loader)
        for idx, sample in enumerate(tbar):
            image, target = sample['image'], sample['label']
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()

            input_image = image.clone()
            input_var = Variable(input_image, requires_grad=True)
            self.model.eval()
            self.optimizer.zero_grad()
            output_var = self.model(input_var)
            saliency_target = target.clone()
            targets = self.generate_target(saliency_target)

            losses = list()
            for i in range(self.args.total_heads):
                losses.append(self.criterion(output_var[i], targets[i].float()).view(-1, 1))

            loss_clean = torch.cat(losses)
            loss_clean = torch.sum(loss_clean)
            loss_clean.backward(retain_graph=True)
            self.optimizer.zero_grad()
            self.model.train()

            h, w = input_image.shape[2], input_image.shape[3]
            saliency = torch.sqrt(torch.mean(input_var.grad ** 2, dim=1, keepdim=True)).detach()
            saliency = F.adaptive_avg_pool2d(saliency, self.args.grid_size)
            bs_kernel, bs_pad = get_bspline_kernel(spacing=[h // self.args.grid_size, h // self.args.grid_size], order=2)
            saliency = F.conv_transpose2d(saliency, bs_kernel, padding=bs_pad, stride=h // self.args.grid_size)
            saliency = F.interpolate(saliency, size=(h, w), mode='bilinear', align_corners=True)
            saliency = rescale_intensity(saliency).detach()

            pseudo_indexes = np.argwhere(target.detach().cpu().numpy() == 2).reshape(-1)
            normal_indexes = np.argwhere(target.detach().cpu().numpy() == 0).reshape(-1)
            normal_list = list(normal_indexes)
            pseudo_image = image.clone()[pseudo_indexes]

            for i in range(pseudo_indexes.size):
                anomlay_idx = random.choice(normal_list)
                anomaly_mask = saliency[anomlay_idx, :, :, :]
                anomaly_mask[anomaly_mask > 0.4] = 1
                anomaly_mask[anomaly_mask <= 0.4] = 0
                anomlay_image = image.clone()[anomlay_idx]
                anomlay_aug = (1 - anomaly_mask) * anomlay_image
                pseudo_image[i, :, :, :][anomlay_aug != 0] = anomlay_aug[anomlay_aug != 0]
                image[pseudo_indexes[i], :, :, :] = pseudo_image[i, :, :, :]

            outputs = self.model(image)
            targets = self.generate_target(target)

            losses = list()
            for i in range(self.args.total_heads):
                if self.args.criterion == 'CE':
                    prob = F.softmax(outputs[i], dim=1)
                    losses.append(self.criterion(prob, targets[i].long()).view(-1, 1))
                else:
                    losses.append(self.criterion(outputs[i], targets[i].float()).view(-1, 1))

            loss = torch.cat(losses)
            loss = torch.sum(loss)

            self.optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            train_loss += loss.item()
            for i in range(self.args.total_heads):
                class_loss[i] += losses[i].item()

            tbar.set_description('Epoch:%d, Train loss: %.3f' % (epoch, train_loss / (idx + 1)))
        self.scheduler.step()


    def normalization(self, data):
        return data

    def eval(self):
        self.model.eval()
        tbar = tqdm(self.test_loader, desc='\r')
        test_loss = 0.0
        class_pred = list()
        for i in range(self.args.total_heads):
            class_pred.append(np.array([]))
        total_target = np.array([])
        for i, sample in enumerate(tbar):
            image, target = sample['image'], sample['label']
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()


            with torch.no_grad():
                outputs = self.model(image)
                targets = self.generate_target(target, eval=True)

                losses = list()
                for i in range(self.args.total_heads):
                    if self.args.criterion == 'CE':
                        prob = F.softmax(outputs[i], dim=1)
                        losses.append(self.criterion(prob, targets[i].long()))
                    else:
                        losses.append(self.criterion(outputs[i], targets[i].float()))

                loss = losses[0]
                for i in range(1, self.args.total_heads):
                    loss += losses[i]

            test_loss += loss.item()
            tbar.set_description('Test loss: %.3f' % (test_loss / (i + 1)))

            for i in range(self.args.total_heads):
                if i == 0:
                    data = -1 * outputs[i].data.cpu().numpy()
                else:
                    data = outputs[i].data.cpu().numpy()
                class_pred[i] = np.append(class_pred[i], data)
            total_target = np.append(total_target, target.cpu().numpy())

        total_pred = self.normalization(class_pred[0])
        for i in range(1, self.args.total_heads):
            total_pred = total_pred + self.normalization(class_pred[i])

        with open(self.args.experiment_dir + '/result.txt', mode='a+', encoding="utf-8") as w:
            for label, score in zip(total_target, total_pred):
                w.write(str(label) + '   ' + str(score) + "\n")

        total_roc, total_pr = aucPerformance(total_pred, total_target)

        normal_mask = total_target == 0
        outlier_mask = total_target == 1
        plt.clf()
        plt.bar(np.arange(total_pred.size)[normal_mask], total_pred[normal_mask], color='green')
        plt.bar(np.arange(total_pred.size)[outlier_mask], total_pred[outlier_mask], color='red')
        plt.ylabel("Anomaly score")
        plt.savefig(args.experiment_dir + "/vis.png")
        return total_roc, total_pr

    def save_weights(self, filename):
        torch.save(self.model.state_dict(), os.path.join(args.experiment_dir, filename))

    def init_network_weights_from_pretraining(self):

        net_dict = self.model.state_dict()
        ae_net_dict = self.ae_model.state_dict()

        ae_net_dict = {k: v for k, v in ae_net_dict.items() if k in net_dict}
        net_dict.update(ae_net_dict)
        self.model.load_state_dict(net_dict)

def aucPerformance(mse, labels, prt=True):
    roc_auc = roc_auc_score(labels, mse)
    ap = average_precision_score(labels, mse)
    if prt:
        print("AUC-ROC: %.4f, AUC-PR: %.4f" % (roc_auc, ap))
    return roc_auc, ap;

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=48, help="batch size used in SGD")
    parser.add_argument("--steps_per_epoch", type=int, default=20, help="the number of batches per epoch")
    parser.add_argument("--epochs", type=int, default=30, help="the number of epochs")
    parser.add_argument("--cont_rate", type=float, default=0.0, help="the outlier contamination rate in the training data")
    parser.add_argument("--test_threshold", type=int, default=0,
                        help="the outlier contamination rate in the training data")
    parser.add_argument("--test_rate", type=float, default=0.0,
                        help="the outlier contamination rate in the training data")
    parser.add_argument("--dataset", type=str, default='mvtecad', help="a list of data set names")
    parser.add_argument("--ramdn_seed", type=int, default=42, help="the random seed number")
    parser.add_argument('--workers', type=int, default=4, metavar='N', help='dataloader threads')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--savename', type=str, default='model.pkl', help="save modeling")
    parser.add_argument('--dataset_root', type=str, default='', help="dataset root")
    parser.add_argument('--experiment_dir', type=str, default='./experiments/carpet_anomaly', help="dataset root")
    parser.add_argument('--classname', type=str, default='carpet', help="dataset class")
    parser.add_argument('--img_size', type=int, default=448, help="dataset root")
    parser.add_argument("--nAnomaly", type=int, default=0, help="the number of anomaly data in training set")
    parser.add_argument("--n_scales", type=int, default=2, help="number of scales at which features are extracted")
    parser.add_argument('--backbone', type=str, default='resnet18', help="backbone")
    parser.add_argument('--criterion', type=str, default='deviation', help="loss")
    parser.add_argument("--topk", type=float, default=0.1, help="topk in MIL")
    parser.add_argument('--know_class', type=str, default=None, help="set the know class for hard setting")
    parser.add_argument('--pretrain_dir', type=str, default=None, help="root of pretrain weight")
    parser.add_argument("--total_heads", type=int, default=2, help="number of head in training")
    parser.add_argument('--outlier_root', type=str, default=None, help="OOD dataset root")
    parser.add_argument('--grid_size', type=int, default=18)
    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    trainer = Trainer(args)


    argsDict = args.__dict__
    if not os.path.exists(args.experiment_dir):
        os.makedirs(args.experiment_dir)
    with open(args.experiment_dir + '/setting.txt', 'w') as f:
        f.writelines('------------------ start ------------------' + '\n')
        for eachArg, value in argsDict.items():
            f.writelines(eachArg + ' : ' + str(value) + '\n')
        f.writelines('------------------- end -------------------')

    print('Seed:', trainer.args.ramdn_seed)
    print('Dataset root:', trainer.args.dataset_root)
    print('Dataset:', trainer.args.classname)
    print('Konwn class:', trainer.args.know_class)
    print('Total Epoches:', trainer.args.epochs)
    print('nAnomaly:', trainer.args.nAnomaly)
    trainer.model = trainer.model.to('cuda')
    trainer.criterion = trainer.criterion.to('cuda')
    for epoch in range(0, trainer.args.epochs):
        trainer.training(epoch)
    trainer.eval()
    trainer.save_weights(args.savename)

