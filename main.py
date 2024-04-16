#!/usr/bin/env python
# -*- coding: utf-8 -*-



from __future__ import print_function
import os
import sys
import gc
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from model import RIENET, multimodal_RIENET, RPMNet
from util import npmat2euler
import numpy as np
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from tqdm import tqdm
import yaml
from easydict import EasyDict

from common.math import se3
from common.math_torch import se3

from data import dataset

class IOStream:
    def __init__(self, path):
        self.f = open(path, 'a')

    def cprint(self, text):
        print(text)
        self.f.write(text + '\n')
        self.f.flush()

    def close(self):
        self.f.close()


def _init_(args):
    if not os.path.exists('/root/autodl-tmp/MFDA-Net/checkpoints'):
        os.makedirs('/root/autodl-tmp/MFDA-Net/checkpoints')
    if not os.path.exists('/root/autodl-tmp/MFDA-Net/checkpoints/' + args.exp_name):
        os.makedirs('/root/autodl-tmp/MFDA-Net/checkpoints/' + args.exp_name)
    if not os.path.exists('/root/autodl-tmp/MFDA-Net/checkpoints/' + args.exp_name + '/' + 'models'):
        os.makedirs('/root/autodl-tmp/MFDA-Net/checkpoints/' + args.exp_name + '/' + 'models')
    if not args.eval:
        os.system('cp main.py checkpoints' + '/' + args.exp_name + '/' + 'main.py.backup')
        os.system('cp model.py checkpoints' + '/' + args.exp_name + '/' + 'model.py.backup')
        os.system('cp utils.py checkpoints' + '/' + args.exp_name + '/' + 'utils.py.backup')

def get_grad_norm(model, prefix, norm_type: float = 2.0):
    parameters = [p for n, p in model.named_parameters() if n.startswith(prefix)]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    param_norm = [torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]
    total_norm = torch.norm(torch.stack(param_norm), norm_type)
    return total_norm

def test_one_epoch(args, net, test_loader):
    net.eval()

    total_loss = 0
    num_examples = 0
    rotations_ab = []
    translations_ab = []
    rotations_ab_pred = []
    translations_ab_pred = []
    emd_list = []
    
    for src, target, rotation_ab, translation_ab, whole_src, whole_tgt, img in tqdm(test_loader):
        src = src.cuda()
        target = target.cuda()
        rotation_ab = rotation_ab.cuda()
        translation_ab = translation_ab.cuda()
        whole_src = whole_src.cuda()
        whole_tgt = whole_tgt.cuda()
        img = img.cuda()

        batch_size = src.size(0)
        num_examples += batch_size
        
        rotation_ab_pred, translation_ab_pred, emd,\
            loss1, loss2, loss3 = net(src, target, whole_src, whole_tgt, img)
        rotations_ab.append(rotation_ab.detach().cpu().numpy())
        translations_ab.append(translation_ab.detach().cpu().numpy())
        rotations_ab_pred.append(rotation_ab_pred.detach().cpu().numpy())
        translations_ab_pred.append(translation_ab_pred.detach().cpu().numpy())
        emd_list.append(emd.detach().cpu().numpy())
        loss = loss1.sum() + loss2.sum() + loss3.sum()
        total_loss += loss.item()

    rotations_ab = np.concatenate(rotations_ab, axis=0)
    translations_ab = np.concatenate(translations_ab, axis=0)
    rotations_ab_pred = np.concatenate(rotations_ab_pred, axis=0)
    translations_ab_pred = np.concatenate(translations_ab_pred, axis=0)

    return total_loss * 1.0 / num_examples, rotations_ab, \
           translations_ab, rotations_ab_pred, translations_ab_pred, emd_list

def train_one_epoch(args, net, train_loader, opt):
    net.train()
    total_loss = 0
    num_examples = 0
    rotations_ab = []
    translations_ab = []
    rotations_ab_pred = []
    translations_ab_pred = []

    for src, target, rotation_ab, translation_ab, whole_src, whole_tgt, img in tqdm(train_loader):
        src = src.cuda()
        target = target.cuda()
        rotation_ab = rotation_ab.cuda()
        translation_ab = translation_ab.cuda()
        whole_src = whole_src.cuda()
        whole_tgt = whole_tgt.cuda()
        img = img.cuda()

        batch_size = src.size(0)
        opt.zero_grad()
        num_examples += batch_size

        rotation_ab_pred, translation_ab_pred, emd,\
            loss1, loss2, loss3 = net(src, target, whole_src, whole_tgt, img)
        rotations_ab.append(rotation_ab.detach().cpu().numpy())
        translations_ab.append(translation_ab.detach().cpu().numpy())
        rotations_ab_pred.append(rotation_ab_pred.detach().cpu().numpy())
        translations_ab_pred.append(translation_ab_pred.detach().cpu().numpy())
        loss = loss1.sum() + loss2.sum() + loss3.sum()

        print('Loss1: %f, Loss2: %f, Loss3: %f'
                  % (loss1.sum(), loss2.sum(), loss3.sum()))
        loss.backward()

        #计算2d和3d分支的梯度
        grad_norm_3d = get_grad_norm(net, prefix='emb_nn')
        grad_norm_2d = get_grad_norm(net, prefix='img_emb_nn')
        grad_norm_fusion = get_grad_norm(net, prefix='fusion')
        grad_norm_forwards = get_grad_norm(net, prefix='forwards')

        print('norm:','3d: %f, 2d: %f, fusion: %f, forwards: %f'
                  % (grad_norm_3d, grad_norm_2d, grad_norm_fusion, grad_norm_forwards))
        opt.step()
        total_loss += loss.item()

    rotations_ab = np.concatenate(rotations_ab, axis=0)
    translations_ab = np.concatenate(translations_ab, axis=0)
    rotations_ab_pred = np.concatenate(rotations_ab_pred, axis=0)
    translations_ab_pred = np.concatenate(translations_ab_pred, axis=0)

    return total_loss * 1.0 / num_examples, rotations_ab, \
           translations_ab, rotations_ab_pred, translations_ab_pred, grad_norm_3d, grad_norm_2d, grad_norm_fusion, \
            grad_norm_forwards


def test(args, net, test_loader, boardio, textio):
    with torch.no_grad():
        test_loss, test_rotations_ab, test_translations_ab, \
        test_rotations_ab_pred, \
        test_translations_ab_pred, emd = test_one_epoch(args, net, test_loader)

    pred_transforms = torch.from_numpy(np.concatenate([test_rotations_ab_pred,test_translations_ab_pred.reshape(-1,3,1)], axis=-1))
    gt_transforms = torch.from_numpy(np.concatenate([test_rotations_ab,test_translations_ab.reshape(-1,3,1)], axis=-1))
    concatenated = se3.concatenate(se3.inverse(gt_transforms), pred_transforms)
    rot_trace = concatenated[:, 0, 0] + concatenated[:, 1, 1] + concatenated[:, 2, 2]
    residual_rotdeg = (torch.acos(torch.clamp(0.5 * (rot_trace - 1), min=-1.0, max=1.0)) * 180.0 / np.pi).detach().cpu().numpy()
    residual_transmag = concatenated[:, :, 3].norm(dim=-1).detach().cpu().numpy()

    deg_mean = np.mean(residual_rotdeg) #/.////
    deg_rmse = np.sqrt(np.mean(residual_rotdeg**2))

    trans_mean = np.mean(residual_transmag) #/.////
    trans_rmse = np.sqrt(np.mean(residual_transmag**2))

    test_rotations_ab_pred_euler = npmat2euler(test_rotations_ab_pred)
    test_eulers_ab = npmat2euler(test_rotations_ab)
    test_r_mse_ab = np.mean((test_rotations_ab_pred_euler - test_eulers_ab) ** 2)
    test_r_rmse_ab = np.sqrt(test_r_mse_ab)
    test_r_mae_ab = np.mean(np.abs(test_rotations_ab_pred_euler - test_eulers_ab))
    test_t_mse_ab = np.mean((test_translations_ab - test_translations_ab_pred) ** 2)
    # from sklearn.metrics import r2_score
    # r_ab_r2_score = r2_score(test_eulers_ab, test_rotations_ab_pred_euler)
    # t_ab_r2_score = r2_score(test_translations_ab, test_translations_ab_pred)

    test_t_rmse_ab = np.sqrt(test_t_mse_ab)
    test_t_mae_ab = np.mean(np.abs(test_translations_ab - test_translations_ab_pred))
    
    emd = np.mean(emd)

    textio.cprint('==FINAL TEST==')
    textio.cprint('A--------->B')
    textio.cprint('EPOCH:: %d, Loss: %f, rot_MSE: %f, rot_RMSE: %f,'
                  'rot_MAE: %f, trans_MSE: %f, trans_RMSE: %f, trans_MAE: %f, deg_mean: %f, deg_rmse: %f, \
                  trans_mean: %f, trans_rmse: %f, emd: %f : '
                  % (-1, test_loss,
                     test_r_mse_ab, test_r_rmse_ab,
                     test_r_mae_ab, test_t_mse_ab, test_t_rmse_ab, test_t_mae_ab,
                     deg_mean,deg_rmse,trans_mean,trans_rmse, emd))

def train(args, net, train_loader, test_loader, boardio, textio):
    checkpoint = None
    if args.resume:
        textio.cprint("start resume from checkpoint...........")
        if args.model_path is '':
            model_path = '/root/autodl-tmp/MFDA-Net/checkpoints' + '/' + args.exp_name + '/models/model.best.t7'
            print(model_path)
        else:
            model_path = args.model_path
            print(model_path)
        if not os.path.exists(model_path):
            print("can't find pretrained model")
            return
        checkpoint = torch.load(model_path)
        args.start_epoch = checkpoint['epoch']
        net.load_state_dict(checkpoint['model'], strict=False)
        textio.cprint("end resume from checkpoint!!!!!!!!!!!!!!")
    best_test_r_mse_ab = np.inf
    if args.use_sgd:
        print("Use SGD")
        opt = optim.SGD(net.parameters(), lr=args.lr * 100, momentum=args.momentum, weight_decay=1e-4)
    else:
        print("Use Adam")
        opt = optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-4)

    scheduler = MultiStepLR(opt,
                            milestones=[int(i) for i in args.lr_step],
                            gamma=0.3)

    if checkpoint is not None:
        best_test_r_mse_ab = checkpoint['best_result']
        print(best_test_r_mse_ab)
        opt.load_state_dict(checkpoint['optimizer'])
    best_test_loss = np.inf

    best_test_r_mse_ab = np.inf
    best_test_r_rmse_ab = np.inf
    best_test_r_mae_ab = np.inf
    best_test_t_mse_ab = np.inf
    best_test_t_rmse_ab = np.inf
    best_test_t_mae_ab = np.inf

    best_deg_mean = np.inf
    best_deg_rmse = np.inf
    best_trans_mean = np.inf
    best_trans_rmse = np.inf

    for epoch in range(args.epochs):
        scheduler.step()
        train_loss, train_rotations_ab, train_translations_ab, \
        train_rotations_ab_pred, train_translations_ab_pred,\
             grad_norm_3d, grad_norm_2d, grad_norm_fusion, \
                grad_norm_forwards = train_one_epoch(args, net, train_loader, opt)
        
        with torch.no_grad():
            test_loss, test_rotations_ab, test_translations_ab, \
            test_rotations_ab_pred, \
            test_translations_ab_pred ,emd = test_one_epoch(args, net, test_loader)

        pred_transforms = torch.from_numpy(np.concatenate([test_rotations_ab_pred,test_translations_ab_pred.reshape(-1,3,1)], axis=-1))
        gt_transforms = torch.from_numpy(np.concatenate([test_rotations_ab,test_translations_ab.reshape(-1,3,1)], axis=-1))
        concatenated = se3.concatenate(se3.inverse(gt_transforms), pred_transforms)
        rot_trace = concatenated[:, 0, 0] + concatenated[:, 1, 1] + concatenated[:, 2, 2]
        residual_rotdeg = (torch.acos(torch.clamp(0.5 * (rot_trace - 1), min=-1.0, max=1.0)) * 180.0 / np.pi).detach().cpu().numpy()
        residual_transmag = concatenated[:, :, 3].norm(dim=-1).detach().cpu().numpy()

        deg_mean = np.mean(residual_rotdeg) #/.////
        deg_rmse = np.sqrt(np.mean(residual_rotdeg**2))

        trans_mean = np.mean(residual_transmag) #/.////
        trans_rmse = np.sqrt(np.mean(residual_transmag**2))

        train_rotations_ab_pred_euler = npmat2euler(train_rotations_ab_pred)
        train_eulers_ab = npmat2euler(train_rotations_ab)
        train_r_mse_ab = np.mean((train_rotations_ab_pred_euler - train_eulers_ab) ** 2)
        train_r_rmse_ab = np.sqrt(train_r_mse_ab)
        train_r_mae_ab = np.mean(np.abs(train_rotations_ab_pred_euler - train_eulers_ab))
        train_t_mse_ab = np.mean((train_translations_ab - train_translations_ab_pred) ** 2)
        train_t_rmse_ab = np.sqrt(train_t_mse_ab)
        train_t_mae_ab = np.mean(np.abs(train_translations_ab - train_translations_ab_pred))

        test_rotations_ab_pred_euler = npmat2euler(test_rotations_ab_pred)
        test_eulers_ab = npmat2euler(test_rotations_ab)
        test_r_mse_ab = np.mean((test_rotations_ab_pred_euler - test_eulers_ab) ** 2)
        test_r_rmse_ab = np.sqrt(test_r_mse_ab)
        test_r_mae_ab = np.mean(np.abs(test_rotations_ab_pred_euler - test_eulers_ab))
        test_t_mse_ab = np.mean((test_translations_ab - test_translations_ab_pred) ** 2)
        test_t_rmse_ab = np.sqrt(test_t_mse_ab)
        test_t_mae_ab = np.mean(np.abs(test_translations_ab - test_translations_ab_pred))


        if best_test_loss >= test_loss:
            best_test_loss = test_loss
    
            best_test_r_mse_ab = test_r_mse_ab
            best_test_r_rmse_ab = test_r_rmse_ab
            best_test_r_mae_ab = test_r_mae_ab

            best_test_t_mse_ab = test_t_mse_ab
            best_test_t_rmse_ab = test_t_rmse_ab
            best_test_t_mae_ab = test_t_mae_ab

            best_deg_mean = deg_mean
            best_deg_rmse = deg_rmse
            best_trans_mean = trans_mean
            best_trans_rmse = trans_rmse

            if torch.cuda.device_count() > 1:
                state = {'model':net.module.state_dict(),'optimizer':opt.state_dict(),'epoch':epoch+1,'best_result':best_test_r_mse_ab}
                torch.save(state, '/root/autodl-tmp/MFDA-Net/checkpoints/%s/models/model.best.t7' % args.exp_name)
            else:
                state = {'model':net.state_dict(),'optimizer':opt.state_dict(),'epoch':epoch+1,'best_result':best_test_r_mse_ab}
                torch.save(state, '/root/autodl-tmp/MFDA-Net/checkpoints/%s/models/model.best.t7' % args.exp_name)

        textio.cprint('==TRAIN==')
        textio.cprint('A--------->B')
        textio.cprint('EPOCH:: %d, Loss: %f, rot_MSE: %f, rot_RMSE: %f, '
                      'rot_MAE: %f, trans_MSE: %f, trans_RMSE: %f, trans_MAE: %f'
                      % (epoch, train_loss, train_r_mse_ab,
                         train_r_rmse_ab, train_r_mae_ab, train_t_mse_ab, train_t_rmse_ab, train_t_mae_ab))

        textio.cprint('==TEST==')
        textio.cprint('A--------->B')
        textio.cprint('EPOCH:: %d, Loss: %f, rot_MSE: %f, rot_RMSE: %f, '
                      'rot_MAE: %f, trans_MSE: %f, trans_RMSE: %f, trans_MAE: %f , deg_mean: %f, deg_rmse: %f, trans_mean: %f, trans_rmse: %f: '
                      % (epoch, test_loss, test_r_mse_ab,
                         test_r_rmse_ab, test_r_mae_ab, test_t_mse_ab, test_t_rmse_ab, test_t_mae_ab, deg_mean,deg_rmse,trans_mean,trans_rmse))

        textio.cprint('==BEST TEST==')
        textio.cprint('A--------->B')
        textio.cprint('EPOCH:: %d, Loss: %f, rot_MSE: %f, rot_RMSE: %f, '
                      'rot_MAE: %f, trans_MSE: %f, trans_RMSE: %f, trans_MAE: %f , deg_mean: %f, deg_rmse: %f, trans_mean: %f, trans_rmse: %f: '
                      % (epoch, best_test_loss, best_test_r_mse_ab, best_test_r_rmse_ab,
                         best_test_r_mae_ab, best_test_t_mse_ab, best_test_t_rmse_ab, best_test_t_mae_ab, best_deg_mean,best_deg_rmse,best_trans_mean,best_trans_rmse))
        
        boardio.add_scalar('A->B/train/loss', train_loss, epoch)
        boardio.add_scalar('A->B/train/rotation/MSE', train_r_mse_ab, epoch)
        boardio.add_scalar('A->B/train/rotation/RMSE', train_r_rmse_ab, epoch)
        boardio.add_scalar('A->B/train/rotation/MAE', train_r_mae_ab, epoch)
        boardio.add_scalar('A->B/train/translation/MSE', train_t_mse_ab, epoch)
        boardio.add_scalar('A->B/train/translation/RMSE', train_t_rmse_ab, epoch)
        boardio.add_scalar('A->B/train/translation/MAE', train_t_mae_ab, epoch)
        ############GRAD NORM
        boardio.add_scalar('grad_norm_3d', grad_norm_3d, epoch)
        boardio.add_scalar('grad_norm_2d', grad_norm_2d, epoch)
        boardio.add_scalar('grad_norm_fusion', grad_norm_fusion, epoch)
        boardio.add_scalar('grad_norm_forwards', grad_norm_forwards, epoch)
        ############TEST
        boardio.add_scalar('A->B/test/loss', test_loss, epoch)
        boardio.add_scalar('A->B/test/rotation/MSE', test_r_mse_ab, epoch)
        boardio.add_scalar('A->B/test/rotation/RMSE', test_r_rmse_ab, epoch)
        boardio.add_scalar('A->B/test/rotation/MAE', test_r_mae_ab, epoch)
        boardio.add_scalar('A->B/test/translation/MSE', test_t_mse_ab, epoch)
        boardio.add_scalar('A->B/test/translation/RMSE', test_t_rmse_ab, epoch)
        boardio.add_scalar('A->B/test/translation/MAE', test_t_mae_ab, epoch)

        ############BEST TEST
        boardio.add_scalar('A->B/best_test/loss', best_test_loss, epoch)
        boardio.add_scalar('A->B/best_test/rotation/MSE', best_test_r_mse_ab, epoch)
        boardio.add_scalar('A->B/best_test/rotation/RMSE', best_test_r_rmse_ab, epoch)
        boardio.add_scalar('A->B/best_test/rotation/MAE', best_test_r_mae_ab, epoch)
        boardio.add_scalar('A->B/best_test/translation/MSE', best_test_t_mse_ab, epoch)
        boardio.add_scalar('A->B/best_test/translation/RMSE', best_test_t_rmse_ab, epoch)
        boardio.add_scalar('A->B/best_test/translation/MAE', best_test_t_mae_ab, epoch)

        if torch.cuda.device_count() > 1:
            state ={'model':net.module.state_dict(),'optimizer':opt.state_dict(),'epoch':epoch+1,'best_result':best_test_r_mse_ab}
            torch.save(state, '/root/autodl-tmp/MFDA-Net/checkpoints/%s/models/model.%d.t7' % (args.exp_name, epoch))
        else:
            state ={'model':net.state_dict(),'optimizer':opt.state_dict(),'epoch':epoch+1,'best_result':best_test_r_mse_ab}
            torch.save(state, '/root/autodl-tmp/MFDA-Net/checkpoints/%s/models/model.%d.t7' % (args.exp_name, epoch))
        gc.collect()

def parse_args_from_yaml(yaml_path):
    with open(yaml_path, 'r',encoding='utf-8') as fd:
        args = yaml.safe_load(fd)
        args = EasyDict(d=args)
    return args

def main():
    #args = parse_args_from_yaml(sys.argv[1])
    args = parse_args_from_yaml('/root/autodl-tmp/MFDA-Net/config/mytrain7.yaml')
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    boardio = SummaryWriter(log_dir='/root/autodl-tmp/MFDA-Net/checkpoints/' + args.exp_name)
    _init_(args)

    textio = IOStream('/root/autodl-tmp/MFDA-Net/checkpoints/' + args.exp_name + '/run.log')
    textio.cprint(str(args))
    '''
    if args.dataset == 'modelnet40':
        train_loader = DataLoader(ModelNet40(num_points=args.n_points,
                                             num_subsampled_points=args.n_subsampled_points,
                                             partition='train', gaussian_noise=args.gaussian_noise,
                                             unseen=args.unseen, rot_factor=args.rot_factor),
                                  batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=6)
        test_loader = DataLoader(ModelNet40(num_points=args.n_points,
                                            num_subsampled_points=args.n_subsampled_points,
                                            partition='test', gaussian_noise=args.gaussian_noise,
                                            unseen=args.unseen, rot_factor=args.rot_factor),
                                 batch_size=args.test_batch_size, shuffle=False, drop_last=False, num_workers=6)
    '''
    

    if args.dataset == '7scenes':
        trainset, testset = dataset.get_datasets(args)
        test_loader = torch.utils.data.DataLoader(
        testset,
        batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
        train_loader = torch.utils.data.DataLoader(
        trainset,
        batch_size=args.batch_size, shuffle=True, num_workers=args.workers)

    elif args.dataset == 'part_pc':
        trainset, testset = dataset.get_part_pc_datasets(args)
        test_loader = torch.utils.data.DataLoader(
        testset,
        batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
        train_loader = torch.utils.data.DataLoader(
        trainset,
        batch_size=args.batch_size, shuffle=True, num_workers=args.workers)

    else:
        raise Exception("not implemented")



    if args.model == 'multimodal_RIENET':
        net = multimodal_RIENET(args).cuda()
        if args.eval:
            if args.model_path is '':
                model_path = '/root/autodl-tmp/MFDA-Net/pretrained' + '/' + args.exp_name + '/model.best.t7'
            else:
                model_path = args.model_path
                print(model_path)
            if not os.path.exists(model_path):
                print("can't find pretrained model")
                return
            
            checkpoint = torch.load(model_path)
            print(checkpoint['epoch'],checkpoint['best_result'])
            net.load_state_dict(checkpoint['model'], strict=False)
            textio.cprint("end resume from checkpoint!!!!!!!!!!!!!!")
        if torch.cuda.device_count() > 1:
            net = nn.DataParallel(net)
            print("Let's use", torch.cuda.device_count(), "GPUs!")

    elif args.model == 'RIENET':
        net = RIENET(args).cuda()
        if args.eval:
            model_path = args.model_path
            print(model_path)
            if not os.path.exists(model_path):
                print("can't find pretrained model")
                return
            
            checkpoint = torch.load(model_path)
            print(checkpoint['epoch'],checkpoint['best_result'])
            net.load_state_dict(checkpoint['model'], strict=False)
            textio.cprint("end resume from checkpoint!!!!!!!!!!!!!!")
        if torch.cuda.device_count() > 1:
            net = nn.DataParallel(net)
            print("Let's use", torch.cuda.device_count(), "GPUs!")

    elif args.model == 'rpmnet':
        net = RPMNet(args).cuda()
        if args.eval:
            model_path = args.model_path
            print(model_path)
            if not os.path.exists(model_path):
                print("can't find pretrained model")
                return
            
            checkpoint = torch.load(model_path)
            print(checkpoint['epoch'],checkpoint['best_result'])
            net.load_state_dict(checkpoint['model'], strict=False)
            textio.cprint("end resume from checkpoint!!!!!!!!!!!!!!")
        if torch.cuda.device_count() > 1:
            net = nn.DataParallel(net)
            print("Let's use", torch.cuda.device_count(), "GPUs!")

    if args.eval:
        test(args, net, test_loader, boardio, textio)
    else:
        train(args, net, train_loader, test_loader, boardio, textio)


    print('FINISH')
    boardio.close()


if __name__ == '__main__':
    main()
