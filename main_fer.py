# Copyright (c) 2018, Curious AI Ltd. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
import pdb
import re
import argparse
import os
import shutil
import time
import math
import logging

from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import torchvision.datasets


from mean_teacher import architectures, datasets, data, losses, ramps, cli
from mean_teacher.run_context import RunContext
from mean_teacher.data import NO_LABEL
from mean_teacher.utils import *
from mean_teacher.argutils import *
import random, cv2
import pandas as pd

import image_utils
import torch
from torch.utils.tensorboard import SummaryWriter
import notifyhub
import pdb
from PIL import ImageFile
import pickle
from torchvision import transforms
import csv
ImageFile.LOAD_TRUNCATED_IMAGES = True
CONFIG_FP = '/home/tohar/PycharmProjects/TSCN/config.json'


writer = SummaryWriter()

LOG = logging.getLogger('main')

args = None
best_prec1 = 0
global_step = 0

best_prec1_stdt = 0
best_prec1_stdt_epoch = 0
best_prec1_tchr_epoch = 0

TRAIN_DATA_SIZE = 0
TEST_DATA_SIZE = 0

def main(context):
    global global_step
    global best_prec1

    global best_prec1_stdt
    global best_prec1_stdt_epoch
    global best_prec1_tchr_epoch

    global TRAIN_DATA_SIZE
    global TEST_DATA_SIZE

    # Initialize randoms seeds
    torch.cuda.manual_seed_all(args.manual_seed)
    torch.manual_seed(args.manual_seed)
    np.random.seed(args.manual_seed)
    random.seed(args.manual_seed)

    checkpoint_path = context.transient_dir
    save_args(args, context.result_dir, "opt")

    training_log = context.create_train_log("training")
    validation_log = context.create_train_log("validation")
    ema_validation_log = context.create_train_log("ema_validation")

    dataset_config = datasets.__dict__[args.dataset]()
    num_classes = dataset_config.pop('num_classes')

    train_loader, eval_loader, test_loader = create_data_loaders(**dataset_config, args=args)

    def create_model(ema=False, attention=False):
        LOG.info("=> creating {pretrained}{ema}model '{arch}'".format(
            pretrained='pre-trained ' if args.pretrained else '',
            ema='EMA ' if ema else '',
            arch=args.arch))
        model_factory = architectures.__dict__[args.arch]
        model_params = dict(pretrained_facedb=args.pretrained_facedb, num_classes=num_classes, attention=attention )
        print(model_params)
        model = model_factory(**model_params)

        if ema:
            for param in model.parameters():
                param.detach_()

        return model

    if 'baseline_mt' in args.exp: #todo
        s_attention = False
        print("========= MT (without uncertainty module)")
    else:
        s_attention = True

    model = create_model(ema=False, attention=s_attention)
    ema_model = create_model(ema=True, attention=False)

    if args.opt == 'adam':
        optimizer  = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)
    elif args.opt == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    else:
        raise('unimplemented optimizer')

    # optionally resume from a checkpoint
    if args.pretrained:
        print("Loading pretrained weights...", args.pretrained)

        pretrained = torch.load(args.pretrained)
        pretrained_state_dict = pretrained['model_state_dict']

        # update model params
        model_state_dict = model.state_dict()
        loaded_keys = 0
        total_keys = 0
        for key in pretrained_state_dict:
            model_state_dict[key] = pretrained_state_dict[key]
            total_keys += 1
            if key in model_state_dict:
                loaded_keys += 1
        print("Loaded params num:", loaded_keys)
        print("Total params num:", total_keys)
        model.load_state_dict(model_state_dict, strict=True)

        # update ema model params. note that ema model does not have attention layer
        model_state_dict = ema_model.state_dict()
        loaded_keys = 0
        total_keys = 0
        for key in pretrained_state_dict:
            if ((key == 'alpha2.0.weight') | (key == 'alpha2.0.bias') | (key == 'alpha.0.weight')| (key == 'alpha.0.bias')):
                pass
            else:
                model_state_dict[key] = pretrained_state_dict[key]
                total_keys += 1
                if key in model_state_dict:
                    loaded_keys += 1
        print("EMA model:")
        print("Loaded params num:", loaded_keys)
        print("Total params num:", total_keys)
        ema_model.load_state_dict(model_state_dict, strict=True)

    model = nn.DataParallel(model).cuda()
    ema_model = nn.DataParallel(ema_model).cuda()

    if args.resume: #init our model with mt model
        assert os.path.isfile(args.resume), "=> no checkpoint found at '{}'".format(args.resume)
        LOG.info("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        #pdb.set_trace()
        if args.resume_from_branch != 'none':
            model.load_state_dict(checkpoint[args.resume_from_branch], strict=False)
            ema_model.load_state_dict(checkpoint[args.resume_from_branch], strict=False)
        else:

            model.load_state_dict(checkpoint['state_dict'], strict=False)
            ema_model.load_state_dict(checkpoint['ema_state_dict'], strict=False)
        LOG.info("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))

    if args.resume_exp: # resume training with same architecture and parameters
        assert os.path.isfile(args.resume_exp), "=> no checkpoint found at '{}'".format(args.resume_exp)
        LOG.info("=> loading checkpoint '{}'".format(args.resume_exp))
        checkpoint = torch.load(args.resume_exp)
        args.start_epoch = checkpoint['epoch']
        global_step = checkpoint['global_step']
        best_prec1 = checkpoint['best_prec1']
        model.load_state_dict(checkpoint['state_dict'])
        ema_model.load_state_dict(checkpoint['ema_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        LOG.info("=> loaded checkpoint '{}' (epoch {})".format(args.resume_exp, checkpoint['epoch']))

    cudnn.benchmark = True
    model = model.cuda()
    ema_model = ema_model.cuda()

    if args.evaluate:
        LOG.info("Evaluating the primary model on test set:")
        validate(test_loader, model, validation_log, global_step, args.start_epoch)
        LOG.info("Evaluating the EMA model on test set:")
        validate(test_loader, ema_model, ema_validation_log, global_step, args.start_epoch)

        LOG.info("Evaluating the primary model on eval set:")
        validate(eval_loader, model, validation_log, global_step, args.start_epoch)
        LOG.info("Evaluating the EMA model on eval set:")
        validate(eval_loader, ema_model, ema_validation_log, global_step, args.start_epoch)
        return

    LOG.info("Begin Training Phase:")
    for epoch in range(args.start_epoch, args.epochs):
        start_time = time.time()
        # train for one epoch
        train(train_loader, model, ema_model,  optimizer, epoch, training_log)

        LOG.info("--- training epoch in %s seconds ---" % (time.time() - start_time))

        if args.evaluation_epochs and (epoch + 1) % args.evaluation_epochs == 0:
            start_time = time.time()
            LOG.info("Evaluating the primary model:")
            prec1 = validate(eval_loader, model, validation_log, global_step, epoch + 1, 'val')
            LOG.info("Evaluating the EMA model:")
            ema_prec1 = validate(eval_loader, ema_model, ema_validation_log, global_step, epoch + 1, 'val_ema')
            LOG.info("--- validation in %s seconds ---" % (time.time() - start_time))
            is_best = ema_prec1 > best_prec1
            best_prec1 = max(ema_prec1, best_prec1)
            if is_best:
                best_prec1_tchr_epoch = epoch + 1
            if prec1 > best_prec1_stdt:
                best_prec1_stdt = prec1
                best_prec1_stdt_epoch = epoch + 1
                is_best_stdt = True
            else:
                is_best_stdt = False

        else:
            is_best = False

        if args.checkpoint_epochs and (epoch + 1) % args.checkpoint_epochs == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'global_step': global_step,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'ema_state_dict': ema_model.state_dict(),
                'best_prec1': best_prec1,
                'optimizer' : optimizer.state_dict(),
            }, is_best, checkpoint_path, epoch + 1)
        writer.flush()

        print('Best Acc Student: {:.3f}(@{})'.format(best_prec1_stdt, best_prec1_stdt_epoch))
        print('Best Acc Teacher: {:.3f}(@{})'.format(best_prec1, best_prec1_tchr_epoch))

    writer.close()
    print(args)
    send_final(checkpoint_path)


class RafDataSet(Dataset):
    def __init__(self, raf_path, phase, transform=None, transform2=None, noise=False, noise_type='sym'):
        self.phase = phase
        self.transform = transform
        self.transform2 = transform2
        self.raf_path = raf_path
        self.noise = noise

        if self.noise>0:
            print("===== use noisy dataset of " + str(self.noise) + " ratio ======")

            if noise_type != 'sym':
                df = pd.read_csv('../rafdb_noisy/inject' + str(self.noise) + 'noise_' + str(noise_type) +'.txt', sep=' ', header=None,
                                 names=['name', 'label'])
            else:
                df = pd.read_csv('../rafdb_noisy/inject' + str(self.noise) + 'noise.txt', sep=' ', header=None,
                                 names=['name', 'label'])

        else:
            df = pd.read_csv(os.path.join(self.raf_path, 'EmoLabel/list_patition_label.txt'), sep=' ', header=None,
                         names=['name', 'label'])

        if phase == 'train':
            self.data = df[df['name'].str.startswith('train')]
        else:
            self.data = df[df['name'].str.startswith('test')]

        file_names = self.data.loc[:, 'name'].values
        self.label = self.data.loc[:,
                     'label'].values - 1  # 0:Surprise, 1:Fear, 2:Disgust, 3:Happiness, 4:Sadness, 5:Anger, 6:Neutral


        _, self.sample_counts = np.unique(self.label, return_counts=True)
        # print(f' distribution of {phase} samples: {self.sample_counts}')

        self.file_paths = []
        for f in file_names:
            f = f.split(".")[0]
            f = f + "_aligned.jpg"
            path = os.path.join(self.raf_path, 'Image/aligned', f)
            self.file_paths.append(path)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        image = Image.open(path).convert('RGB')
        label = self.label[idx]

        if self.transform is not None:
            image1 = self.transform(image)

        if self.transform2 is not None:
            image2 = self.transform(image)
        else:
            image2 = image1

        return (image1, image2), label


def create_data_loaders(train_transformation,
                        train_transformation2,
                        eval_transformation,
                        datadir,
                        args):


    assert_exactly_one([args.exclude_unlabeled, args.labeled_batch_size])
    test_loader = None

    if args.dataset == 'raf':
        train_dataset = RafDataSet(datadir, phase='train', transform=train_transformation, transform2=train_transformation2, noise=args.noise, noise_type=args.noise_type)

        print('Train set size:', train_dataset.__len__())
        val_dataset = RafDataSet(datadir, phase='test', transform=eval_transformation)
        print('Validation set size:', val_dataset.__len__())


        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=args.batch_size,
                                                   num_workers=args.workers,
                                                   shuffle=True,
                                                   pin_memory=True)

        eval_loader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=args.batch_size,
                                                 num_workers=2 * args.workers,
                                                 shuffle=False,
                                                 pin_memory=True,
                                                 drop_last=False)

    if not test_loader:
        test_loader = eval_loader
    return train_loader, eval_loader, test_loader


def update_ema_variables(model, ema_model, alpha, global_step):

    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

def train(train_loader, model, ema_model, optimizer, epoch, log):
    global global_step

    class_criterion = nn.CrossEntropyLoss(ignore_index=NO_LABEL, reduction='mean').cuda()

    if args.consistency_type == 'mse':
        consistency_criterion = losses.softmax_mse_loss
    elif args.consistency_type == 'kl':
        consistency_criterion = losses.softmax_kl_loss
    else:
        assert False, args.consistency_type
    if args.ls2 :
        consistency_criterion = losses.softmax_kl_loss_sl2

    meters = AverageMeterSet()

    # switch to train mode
    model.train()
    ema_model.train()

    end = time.time()

    running_loss = 0.0
    correct_sum = 0
    correct_sum_ema = 0
    iter_cnt = 0
    sample_cnt = 0

    for i, sample in enumerate(train_loader):
        ((input_var, ema_input_var), target_var) = sample
        iter_cnt += 1
        # measure data loading time
        meters.update('data_time', time.time() - end)

        adjust_learning_rate(optimizer, epoch, i, len(train_loader))
        meters.update('lr', optimizer.param_groups[0]['lr'])

        target_var = target_var.cuda()
        labeled_minibatch_size = target_var.data.ne(NO_LABEL).sum()
        assert labeled_minibatch_size > 0
        meters.update('labeled_minibatch_size', labeled_minibatch_size)

        input_var = input_var.cuda()
        sample_cnt += labeled_minibatch_size

        if not 'baseline_mt' in args.exp: #todo
            dict, class_logit = model(input_var, return_features=True, return_w=True)
            cons_logit = dict['pure_logits']

        else: #baseline MT without attention
            dict, class_logit = model(input_var, return_features=True)
            cons_logit = class_logit

        ema_logit = Variable(ema_logit.detach().data, requires_grad=False)

        class_loss = class_criterion(class_logit, target_var)

        _, predicts = torch.max(cons_logit, 1)
        class_loss = class_loss * args.class_gamma

        meters.update('class_loss', class_loss.data)

        # Calculate EMA class loss:
        ema_class_loss = class_criterion(ema_logit, target_var)
        meters.update('ema_class_loss', ema_class_loss.data)
        eps2 = args.eps2

        if args.consistency:
            consistency_weight = get_current_consistency_weight(epoch)
            meters.update('cons_weight', consistency_weight)
            consistency_loss = consistency_weight * consistency_criterion(cons_logit, ema_logit, eps2, tau=args.tau, k=args.k, function=args.function)
            meters.update('cons_loss', consistency_loss.data)
        else:
            consistency_loss = 0
            meters.update('cons_loss', 0)

        loss = class_loss + consistency_loss

        assert not (np.isnan(loss.cpu().data) or loss.data > 1e5), 'Loss explosion: {}'.format(loss.data)
        meters.update('loss', loss.data)

        _, predicts = torch.max(cons_logit, 1)
        correct_num = torch.eq(predicts, target_var).sum()
        correct_sum += correct_num


        _, ema_predicts = torch.max(ema_logit, 1)
        correct_num_ema = torch.eq(ema_predicts, target_var).sum()
        correct_sum_ema += correct_num_ema

        # compute gradient and do SGD step

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        global_step += 1
        update_ema_variables(model, ema_model, args.ema_decay, global_step)

        # measure elapsed time
        meters.update('batch_time', time.time() - end)
        end = time.time()


    myacc = correct_sum.float() / float(sample_cnt) * 100
    myacc_ema = correct_sum_ema.float() / float(sample_cnt) * 100

    writer.add_scalar("class_loss/train", meters.averages()['class_loss/avg'], epoch)

    writer.add_scalar("class_loss/ema_train", meters.averages()['ema_class_loss/avg'], epoch)
    writer.add_scalar("cons_loss/train", meters.averages()['cons_loss/avg'], epoch)
    writer.add_scalar("loss/train", meters.averages()['loss/avg'], epoch)
    writer.add_scalar("acc/train", myacc, epoch)
    writer.add_scalar("acc/train_ema", myacc_ema, epoch)


    LOG.info(
        'Train: [{}]\t'
        'Class {:.4f}\t'
        'Cons {:.4f}\t'
        'EMA Class {:.4f}\t'
        'Acc {myacc:.3f}\t'
        'Acc-EMA {myacc_ema:.3f}\t'.format(
            epoch, meters.averages()['class_loss/avg'], meters.averages()['cons_loss/avg'],  meters.averages()['ema_class_loss/avg'] , meters=meters, myacc=myacc, myacc_ema=myacc_ema))

def validate(eval_loader, model, log, global_step, epoch, val='val'):
    class_criterion = nn.CrossEntropyLoss(reduction='sum')
    meters = AverageMeterSet()

    running_loss = 0.0
    correct_sum = 0
    iter_cnt = 0
    sample_cnt = 0

    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, data in enumerate(eval_loader):
            ((input_var, _), target_var) = data
            iter_cnt += 1
            meters.update('data_time', time.time() - end)

            target_var = target_var.cuda()

            minibatch_size = len(target_var)
            labeled_minibatch_size = target_var.data.ne(NO_LABEL).sum()
            assert labeled_minibatch_size > 0
            meters.update('labeled_minibatch_size', labeled_minibatch_size)

            # compute output
            input_var = input_var.cuda()
            output1 = model(input_var, mode_val=True)

            class_loss_sum = class_criterion(output1, target_var)
            class_loss = class_loss_sum / minibatch_size

            # measure accuracy and record loss
            running_loss += class_loss
            _, predicts = torch.max(output1, 1)
            correct_num = torch.eq(predicts, target_var).sum()
            correct_sum += correct_num
            sample_cnt += output1.size(0)


            meters.update('class_loss', class_loss.data, labeled_minibatch_size)


            # measure elapsed time
            meters.update('batch_time', time.time() - end)
            end = time.time()

    myacc = correct_sum.float() / float(sample_cnt) * 100

    running_loss = running_loss / iter_cnt #todo: print


    LOG.info(
        'Test: [{epoch}]\t'
        'Class {meters[class_loss].avg:.4f}\t'
        'Acc {myacc:.3f}'.format(
            epoch=epoch, meters=meters, myacc=myacc))

    writer.add_scalar("class_loss/"+val, meters.averages()['class_loss/avg'], epoch)
    writer.add_scalar("acc/"+val, myacc, epoch)
    return myacc


def save_checkpoint(state, is_best, dirpath, epoch):
    filename = 'checkpoint.{}.ckpt'.format(epoch)
    checkpoint_path = os.path.join(dirpath, filename)
    best_filename = 'best_{:.3f}.{}.ckpt'.format(state['best_prec1'], epoch)
    best_path = os.path.join(dirpath, best_filename)
    torch.save(state, checkpoint_path)
    LOG.info("--- checkpoint saved to %s ---" % checkpoint_path)
    if is_best:
        shutil.copyfile(checkpoint_path, best_path)
        LOG.info("--- checkpoint copied to %s ---" % best_path)


def adjust_learning_rate(optimizer, epoch, step_in_epoch, total_steps_in_epoch):
    lr = args.lr
    epoch = epoch + step_in_epoch / total_steps_in_epoch

    # LR warm-up to handle large minibatch sizes from https://arxiv.org/abs/1706.02677
    lr = ramps.linear_rampup(epoch, args.lr_rampup) * (args.lr - args.initial_lr) + args.initial_lr

    # Cosine LR rampdown from https://arxiv.org/abs/1608.03983 (but one cycle only)
    if args.lr_rampdown_epochs:
        assert args.lr_rampdown_epochs >= args.epochs
        lr *= ramps.cosine_rampdown(epoch, args.lr_rampdown_epochs)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


def send_final(exp_name):
    msg = 'Best Acc Student: {:.3f}(@{}), Teacher: {:.3f}(@{}). Saved to: {}. Args: {}'.format(best_prec1_stdt, best_prec1_stdt_epoch,best_prec1, best_prec1_tchr_epoch, exp_name, args)
    notifyhub.send(message=msg, config_fp=CONFIG_FP)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    args = cli.parse_commandline_args()
    main(RunContext(__file__, 0))
