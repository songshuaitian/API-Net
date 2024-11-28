
import os
import argparse
import json

import math
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from tqdm import tqdm
from setproctitle import setproctitle
import models
from datasets import build_dataset, build_dataloader
from datasets.data_sampler import EnlargedSampler
from pytorch_ssim import ssim
from utils import AverageMeter,ContrastLoss_vgg
from utils.options import parse_options
from models import *
setproctitle('API-Net')

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='model-s', type=str, help='model name')
parser.add_argument('--premodel', default='premodel-s', type=str, help='model name')
parser.add_argument('--num_workers', default=4, type=int, help='number of workers')
parser.add_argument('--no_autocast', action='store_false', default=True, help='disable autocast')
parser.add_argument('--save_dir', default='./saved_models/', type=str, help='path to models saving')
parser.add_argument('--data_dir', default='/home/a/data/sst/data/', type=str, help='path to dataset')
parser.add_argument('--log_dir', default='./logs/', type=str, help='path to logs')
parser.add_argument('--dataset', default='rgb_500', type=str, help='dataset name')
parser.add_argument('--exp', default='rgb500', type=str, help='experiment setting')
parser.add_argument('--exp_save', default='RTTS', type=str, help='experiment setting')
parser.add_argument('--cpk', default='cpk', type=str, help='experiment setting')
parser.add_argument('--gpu', default='0', type=str, help='GPUs used for training')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu



def train(train_loader, network,network2,pert_model,  criterion,  criterion_cr, optimizer,optimizer_pert, scaler,scaler_pert):
    losses = AverageMeter()
    losses_e = AverageMeter()
    losses_pert = AverageMeter()
    torch.cuda.empty_cache()
    network.train()
    pert_model.train()

    for batch in train_loader:
        source_img = batch['lq'].cuda()
        target_img = batch['gt'].cuda()
        seg_img = batch['seg'].cuda()

        pert_img = pert_model(source_img)
        org_output = network2(pert_img, seg_img)
        loss_pert1 = -criterion(org_output,target_img)

        losses_pert.update(loss_pert1.item())
        optimizer_pert.zero_grad()
        scaler_pert.scale(loss_pert1).backward()
        scaler_pert.step(optimizer_pert)
        scaler_pert.update()

        new_pert_img = pert_model(source_img)
        output = network(new_pert_img.detach(),seg_img)#新扰动防御
        loss_p = criterion(output,target_img)

        loss_e = criterion(source_img,target_img)
        loss = loss_p + 0.1*criterion_cr(output,target_img,source_img)


        losses_e.update(loss_e.item())

        losses.update(loss.item())
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()


    return losses.avg,losses_e.avg,losses_pert.avg

def valid(val_loader, network):
    PSNR = AverageMeter()
    SSIM = AverageMeter()
    torch.cuda.empty_cache()

    network.eval()

    for batch in val_loader:
        source_img = batch['lq'].cuda()
        target_img = batch['gt'].cuda()
        seg_img = batch['seg'].cuda()

        with torch.no_grad():  # torch.no_grad() may cause warning
            output = network(source_img,seg_img).clamp_(-1, 1)

        mse_loss = F.mse_loss(output, target_img, reduction='none').mean((1, 2, 3))
        psnr = 10 * torch.log10(1 / mse_loss).mean()
        PSNR.update(psnr.item(), source_img.size(0))
        ssim_value = ssim(output, target_img, size_average=True)
        SSIM.update(ssim_value.item(), source_img.size(0))

    return PSNR.avg,SSIM.avg


def create_train_val_dataloader(opt):
    # create train and val dataloaders
    train_loader, val_loaders = None, []
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':

            dataset_enlarge_ratio = dataset_opt.get('dataset_enlarge_ratio', 1)
            train_set = build_dataset(dataset_opt)
            train_sampler = EnlargedSampler(train_set, opt['world_size'], opt['rank'], dataset_enlarge_ratio)
            train_loader = build_dataloader(
                train_set,
                dataset_opt,
                num_gpu=opt['num_gpu'],
                dist=opt['dist'],
                sampler=train_sampler,
                seed=opt['manual_seed'])

            num_iter_per_epoch = math.ceil(
                len(train_set) * dataset_enlarge_ratio / (dataset_opt['batch_size_per_gpu'] * opt['world_size']))
            total_iters = int(opt['train']['total_iter'])
            total_epochs = math.ceil(total_iters / (num_iter_per_epoch))

        elif phase.split('_')[0] == 'val':
            val_set = build_dataset(dataset_opt)
            val_loader = build_dataloader(
                val_set, dataset_opt, num_gpu=opt['num_gpu'], dist=opt['dist'], sampler=None, seed=opt['manual_seed'])

        else:
            raise ValueError(f'Dataset phase {phase} is not recognized.')
    return train_loader, train_sampler, val_loader, total_epochs, total_iters

if __name__ == '__main__':
    setting_filename = os.path.join('configs', args.exp, args.model + '.json')
    if not os.path.exists(setting_filename):
        setting_filename = os.path.join('configs', args.exp, 'default.json')
    with open(setting_filename, 'r') as f:
        setting = json.load(f)

    root_path = '/mnt/data/user/sst/code/API-Net'#当前项目路径
    opt, args_configs1 = parse_options(root_path, is_train=True)
    network = eval(args.model.replace('-', '_'))()
    network2 = eval(args.premodel.replace('-', '_'))()#预训练模型
    pert_model = Perturbation()

    network = nn.DataParallel(network).cuda()
    pert_model = nn.DataParallel(pert_model).cuda()
    network2 = nn.DataParallel(network2).cuda()


    save_dir = os.path.join(args.save_dir, args.exp_save)
    save_dir2 = os.path.join(args.save_dir, args.exp)
    save_dir1 = os.path.join(args.save_dir, args.cpk)

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(save_dir1, exist_ok=True)

    if os.path.exists(os.path.join(save_dir1, 'premodel.pth')):
        ################################################network2##########################################
        print('network2 is load state_dict')
        checkpoint2 = torch.load(os.path.join(save_dir1, 'premodel.pth'))
        network2.load_state_dict(checkpoint2['state_dict'])
        for param in network2.parameters():
            param.requires_grad = False


    criterion = nn.L1Loss()
    criterion_cr = ContrastLoss_vgg()
    if setting['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(network.parameters(), lr=setting['lr'])
        optimizer_pert = torch.optim.Adam(pert_model.parameters(), lr=setting['lr'])
    elif setting['optimizer'] == 'adamw':
        optimizer = torch.optim.AdamW(network.parameters(), lr=setting['lr'])
        optimizer_pert = torch.optim.AdamW(pert_model.parameters(), lr=setting['lr'])
    else:
        raise Exception("ERROR: unsupported optimizer")

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=setting['epochs'],
                                                           eta_min=setting['lr'] * 1e-2)

    scheduler_pert = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_pert, T_max=setting['epochs'],
                                                            eta_min=setting['lr'] * 1e-2)
    scaler = GradScaler()
    scaler_pert = GradScaler()



    if os.path.exists(os.path.join(save_dir,'API-Net.pth')):
        print('==> Continue training from existing model')
        checkpoint = torch.load(os.path.join(save_dir, 'API-Net.pth'))
        network.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['lr_scheduler'])
        scaler.load_state_dict(checkpoint['scaler'])
        start_epoch = checkpoint['epoch'] + 1
    if os.path.exists(os.path.join(save_dir,  'pertmodel.pth')):
        checkpoint_pert = torch.load(os.path.join(save_dir, 'pertmodel.pth'))
        pert_model.load_state_dict(checkpoint_pert['state_dict'])
        optimizer_pert.load_state_dict(checkpoint_pert['optimizer'])
        scheduler_pert.load_state_dict(checkpoint_pert['lr_scheduler'])
        scaler_pert.load_state_dict(checkpoint_pert['scaler'])


    else:
        print('==> Starting training, current model name: ' + args.model)
        best_psnr = 0
        start_epoch = 0

    result = create_train_val_dataloader(opt)
    train_loader, train_sampler, val_loader, total_epochs, total_iters = result


    for epoch in tqdm(range(start_epoch, setting['epochs'] + 1), initial=start_epoch, total=setting['epochs'] + 1):
        loss,loss_e,loss_pert = train(train_loader, network,network2, pert_model,criterion, criterion_cr, optimizer,optimizer_pert,scaler,scaler_pert)  # loss=去雾模型，loss1 =扰动模型

        scheduler.step()
        scheduler_pert.step()
        if epoch % setting['eval_freq'] == 0:
            avg_psnr,avg_ssim = valid(val_loader, network)

            if avg_psnr > best_psnr:
                best_psnr = avg_psnr
                torch.save({'state_dict': network.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'lr_scheduler': scheduler.state_dict(),
                            'scaler': scaler.state_dict(),
                            'epoch': epoch,
                            },
                           os.path.join(save_dir, 'API-Net.pth'.format(epoch)))
                torch.save({'state_dict': pert_model.state_dict(),
                            'optimizer': optimizer_pert.state_dict(),
                            'lr_scheduler': scheduler_pert.state_dict(),
                            'scaler': scaler_pert.state_dict(),
                            },
                           os.path.join(save_dir,'pertmodel.pth'.format(epoch)))

            os.makedirs('./checkpoint', exist_ok=True)
            with open('./checkpoint/loss.txt', 'a') as file:
                file.write('Epoch [{}/{}], Loss: {:.4f},loss_e: {:.4f},loss_pert: {:.4f}'
                           .format(epoch + 1, setting['epochs'], loss,loss_e,loss_pert))
                file.write('Best PSNR: {:.4f}\n'.format(best_psnr))
                file.write('Val PSNR: {:.4f}\n'.format(avg_psnr))
                file.write('Val SSIM: {:.4f}\n'.format(avg_ssim))
                file.write('\n')

            print('loss:', loss,'loss_e:',loss_e,'loss_pert:',loss_pert, 'best_psnr', best_psnr,'avg_ssim:',avg_ssim)

