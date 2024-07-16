import os
import time
import random
import argparse
import json
import itertools
import copy

import matplotlib.pyplot as plt
import numpy as np
from typing import List
from tqdm import tqdm
import torch
import torch.nn as nn
from torchvision import transforms, datasets, models, utils
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
import shutil
from torch.utils.tensorboard import SummaryWriter
# from utils import config, set_seed, Wandb, generate_exp_directory
from nats_bench import create
from xautodl.models.cell_operations import NAS_BENCH_201
from xautodl.models import get_cell_based_tiny_net, get_search_spaces
from xautodl.log_utils import AverageMeter, time_string
from xautodl.utils import count_parameters_in_MB, obtain_accuracy
from xautodl.procedures import get_optim_scheduler, prepare_logger, save_checkpoint
from xautodl.datasets import get_datasets, get_nas_search_loaders
from xautodl.config_utils import dict2config
from torch.utils.data import random_split


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 最好的那个模型的idx是9930


def get_model_for_chestXray(idx=9930, dataset = 'ImageNet16-120'):
    api = create('NATS-tss-v1_0-3ffb9-full', "topology", fast_mode=True, verbose=False)
    config = api.get_net_config(idx, dataset)
    network = get_cell_based_tiny_net(config)
    weights = api.get_net_param(9930, 'ImageNet16-120', seed=None, hp='200')
    network.load_state_dict(weights[777])
    num_ftrs = network.classifier.in_features
    network.classifier = nn.Sequential(
        nn.Linear(num_ftrs, 2, bias=True),
        nn.LogSoftmax(dim=1)
    )
    return network


def get_model_for_BUSI(idx=9930, dataset = 'ImageNet16-120'):
    api = create('NATS-tss-v1_0-3ffb9-full', "topology", fast_mode=True, verbose=False)
    config = api.get_net_config(idx, dataset)
    network = get_cell_based_tiny_net(config)
    weights = api.get_net_param(9930, 'ImageNet16-120', seed=None, hp='200')
    network.load_state_dict(weights[777])
    num_ftrs = network.classifier.in_features
    network.classifier = nn.Sequential(
        nn.Linear(num_ftrs, 3, bias=True),
        nn.LogSoftmax(dim=1)
    )
    # network.classifier = nn.Linear(num_ftrs, 3, bias=True)
    return network


def get_model_for_ISIC(idx=9930, dataset = 'ImageNet16-120'):
    print('creating model for ISIC...')
    api = create('NATS-tss-v1_0-3ffb9-full', "topology", fast_mode=True, verbose=False)
    config = api.get_net_config(idx, dataset)
    network = get_cell_based_tiny_net(config)
    weights = api.get_net_param(9930, 'ImageNet16-120', seed=None, hp='200')
    network.load_state_dict(weights[777])
    num_ftrs = network.classifier.in_features
    network.classifier = nn.Sequential(
        nn.Linear(num_ftrs, 7, bias=True),
        nn.LogSoftmax(dim=1)
    )
    # network.classifier = nn.Linear(num_ftrs, 3, bias=True)
    print('done')
    return network


def load_data_chest():
    compose = transforms.Compose([
        transforms.Resize(300),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],  # mean
                             [0.229, 0.224, 0.225])  # std
    ])

    image_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(size=300, scale=(0.8, 1.1)),  # 功能：随机长宽比裁剪原始图片, 表示随机crop出来的图片会在的0.08倍至1.1倍之间
            transforms.RandomRotation(degrees=10),  # 功能：根据degrees随机旋转一定角度, 则表示在（-10，+10）度之间随机旋转
            transforms.ColorJitter(0.4, 0.4, 0.4),  # 功能：修改亮度、对比度和饱和度
            transforms.RandomHorizontalFlip(),  # 功能：水平翻转

            transforms.CenterCrop(size=256),  # 功能：根据给定的size从中心裁剪，size - 若为sequence,则为(h,w)，若为int，则(size,size)
            transforms.ToTensor(),  # numpy --> tensor
            # 功能：对数据按通道进行标准化（RGB），即先减均值，再除以标准差
            transforms.Normalize([0.485, 0.456, 0.406],  # mean
                                 [0.229, 0.224, 0.225])  # std
        ]),
        'test': compose
    }
    data_dir = 'ChestXRay2017/chest_xray/'
    train_dir = data_dir + 'train/'
    test_dir = data_dir + 'test/'

    print('Loading data from:', data_dir)
    train_set = datasets.ImageFolder(train_dir, transform=image_transforms['train'])
    test_set = datasets.ImageFolder(test_dir, transform=image_transforms['test'])
    BATCH_SIZE = 10

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                              pin_memory=True, num_workers=1)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True,
                             pin_memory=True, num_workers=1)
    train_label = dict((v, k) for k, v in train_set.class_to_idx.items())

    return train_set, train_loader, test_set, test_loader, train_label


def load_data_BUSI():
    from BUSI_loader import train_loader, train_set, test_loader, test_set
    return train_set, train_loader, test_set, test_loader

def load_data_ISIC():
    from ISIC_loader import train_ds, train_dl_i, test_ds, test_dl_i
    return train_ds, train_dl_i, test_ds, test_dl_i

def test_create_model():
    api = create('NATS-tss-v1_0-3ffb9-full', "topology", fast_mode=True, verbose=False)
    size = len(api)
    # for i, arch_str in enumerate(api):
    #     print('{:5d}/{:5d} : {:}'.format(i, len(api), arch_str))
    best_arch = api.query_meta_info_by_index(9930)
    res_metrics = best_arch.get_metrics('cifar100', 'train')
    print(best_arch)
    print(res_metrics)
    # config = api.get_net_config(9930, 'cifar100')
    # network = get_cell_based_tiny_net(config)
    # torch.save(network.state_dict(), 'best_cifar100_model.pth')
    
    # searched best network's id is 9930
    idx = 9930
    target_dataset = 'ImageNet16-120'
    config = api.get_net_config(idx, target_dataset)
    network = get_cell_based_tiny_net(config)
    weights = api.get_net_param(9930, target_dataset, seed=None, hp='200')
    # print('weights.keys:', weights.keys())
    network.load_state_dict(weights[777])
    # torch.save(network.state_dict(), 'best_ImageNet_model.pth')

    # target_dataset = 'cifar100'
    # config = api.get_net_config(idx, target_dataset)
    # network = get_cell_based_tiny_net(config)
    # weights = api.get_net_param(idx, target_dataset, seed=None, hp='200')
    # # print('weights.keys:', weights.keys())
    # network.load_state_dict(weights[777])
    # torch.save(network.state_dict(), 'best_cifar100_model.pth')
    #
    # target_dataset = 'cifar10'
    # config = api.get_net_config(idx, target_dataset)
    # network = get_cell_based_tiny_net(config)
    # weights = api.get_net_param(idx, target_dataset, seed=None, hp='200')
    # # print('weights.keys:', weights.keys())
    # network.load_state_dict(weights[777])
    # torch.save(network.state_dict(), 'best_cifar10_model.pth')

    # num_ftrs = network.classifier.in_features
    # network.classifier = nn.Linear(num_ftrs, 3, bias=True)
    # print(network)


def test_chest(model, test_loader, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0.0  # 正确数
    print('Testing...')
    with torch.no_grad():
        for batch_id, (images, labels) in tqdm(enumerate(test_loader), desc='Test',
                                               total=len(test_loader)):
            images, labels = images.to(device), labels.to(device)
            # 输出
            _, outputs = model(images)
            # 损失
            loss = criterion(outputs, labels)
            # 累计损失
            total_loss += loss.item()
            # 获取预测概率最大值的索引
            _, predicted = torch.max(outputs, dim=1)
            # 累计正确预测的数
            correct += predicted.eq(labels.view_as(predicted)).sum().item()
            # 错误分类的图片
            # misclassified_images(predicted, writer, labels, images, outputs, epoch)

        # 平均损失
        # avg_loss = total_loss / len(test_loader)
        # 计算正确率
        accuracy = 100 * correct / len(test_loader.dataset)
        # 将test的结果写入write
        # writer.add_scalar("Test Loss", total_loss, epoch)
        # writer.add_scalar("Accuracy", accuracy, epoch)
        # writer.flush()
        return total_loss, accuracy


def train_chest():
    model = get_model_for_chestXray().to(device)
    train_set, train_loader, test_set, test_loader, train_label = load_data_chest()
    criterion = nn.NLLLoss()
    optimizer = Adam(params=model.parameters(), lr=0.001, weight_decay=0.)
    # keras lr decay equivalent
    lr_decay = 3e-3
    fcn = lambda step: 1. / (1. + lr_decay * step)
    scheduler = LambdaLR(optimizer, lr_lambda=fcn)

    epochs = 100
    best_loss = 1e10
    best_acc = 0
    train_loss_list = []
    test_loss_list = []
    # train_acc_list = []
    test_acc_list = []

    print('Start training...')
    for epoch in range(epochs):
        print(f'\nEpoch {epoch} / {epochs}')
        model.train()
        total_loss = 0.0
        for batch_id, (images, labels) in tqdm(enumerate(train_loader),
                            desc='[TRAIN] Epoch ' + str(epoch + 1) + '/' + str(epochs),
                            total=len(train_loader)):
            images, labels = images.to(device), labels.to(device)
            # 梯度置0
            optimizer.zero_grad()
            # 模型输出
            _, outputs = model(images)
            # print('model output:', outputs)

            # 计算损失
            # print('output type:', outputs.dtype)
            # print('label type:', labels.dtype)
            # exit(-1)
            print('outputs: ', outputs)
            print('labels:', labels)
            loss = criterion(outputs, labels)
            # 反向传播
            loss.backward()
            # 更新参数
            optimizer.step()
            # 累计损失
            print('loss.item:', loss.item())
            total_loss += loss.item() * images.size(0)

        scheduler.step()

        # 平均训练损失
        print(f'\nEpoch {epoch} total loss: {total_loss}')
        train_loss = total_loss / len(train_loader)
        train_loss_list.append(train_loss)

        # test
        test_loss, test_acc = test_chest(model, test_loader, criterion)
        test_loss_list.append(test_loss)
        test_acc_list.append(test_acc)
        if test_acc > best_acc:
            best_acc = test_acc

        if test_loss < best_loss:
            best_loss = test_loss
            torch.save(model.state_dict(), 'chestXray_model.pth')

        print('Epoch: ' + str(epoch + 1) + '/' + str(epochs),
              'train_loss:', train_loss,
              'test_loss:', test_loss,
              'test_acc:', test_acc)

    print('Train finished')
    print('Best acc:', best_acc)
    print('Best loss:', best_loss)
    plt.figure(figsize=(10, 6))
    X = [x for x in range(epochs)]
    plt.plot(X, train_loss_list, label = 'train_loss', color = 'red')
    plt.plot(X, test_loss_list, label = 'test_loss', color = 'green')
    plt.xlabel('Epoch')
    plt.ylabel('Losses')
    plt.legend(loc='best')
    plt.title('Train and test losses')
    plt.savefig('loss_train_test.png')

    plt.figure(figsize=(10, 6))
    plt.plot(X, test_acc_list, label='test_accuracy', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('test acc')
    plt.legend(loc='best')
    plt.title('test_accuracy')
    plt.savefig('test_acc.png')


def train_BUSI():
    model = get_model_for_BUSI().to(device)
    train_set, train_loader, test_set, test_loader = load_data_BUSI()
    criterion = nn.NLLLoss()
    optimizer = Adam(params=model.parameters(), lr=0.001, weight_decay=0.)
    # keras lr decay equivalent
    lr_decay = 3e-3
    fcn = lambda step: 1. / (1. + lr_decay * step)
    scheduler = LambdaLR(optimizer, lr_lambda=fcn)

    epochs = 20
    best_loss = 1e10
    best_acc = 0
    train_loss_list = []
    test_loss_list = []
    # train_acc_list = []
    test_acc_list = []

    print('BUSI .. Start training...')
    for epoch in range(epochs):
        print(f'\nEpoch {epoch} / {epochs}')
        model.train()
        total_loss = 0.0
        for batch_id, (images, labels) in tqdm(enumerate(train_loader),
                            desc='[TRAIN] Epoch ' + str(epoch + 1) + '/' + str(epochs),
                            total=len(train_loader)):

            images, labels = images.to(device), labels.to(device, torch.int64)

            # 梯度置0
            optimizer.zero_grad()
            # 模型输出
            _, outputs = model(images)
            # print('model output:', outputs)
            # 计算损失
            # print('output type:', outputs.dtype)
            # print('label type:', labels.dtype)
            # print('outputs: ', outputs)
            # print('labels:', labels)
            loss = criterion(outputs, labels)
            # 反向传播
            loss.backward()
            # 更新参数
            optimizer.step()
            # 累计损失
            # print('loss.item:', loss.item())
            loss_item = loss.item()
            total_loss += loss.item()

        scheduler.step()

        # 平均训练损失
        print(f'\nEpoch {epoch} total loss: {total_loss}')
        train_loss = total_loss / len(train_loader)
        train_loss_list.append(train_loss)

        # test
        test_loss, test_acc = test_BUSI(model, test_loader, criterion)
        test_loss_list.append(test_loss)
        test_acc_list.append(test_acc)
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), 'BUSI_model.pth')

        if test_loss < best_loss:
            best_loss = test_loss


        print('Epoch: ' + str(epoch + 1) + '/' + str(epochs),
              'train_loss:', train_loss,
              'test_loss:', test_loss,
              'test_acc:', test_acc)

    print('Train finished')
    print('Best acc:', best_acc)
    print('Best loss:', best_loss)
    plt.figure(figsize=(10, 6))
    X = [x for x in range(epochs)]
    plt.plot(X, train_loss_list, label = 'train_loss', color = 'red')
    plt.plot(X, test_loss_list, label = 'test_loss', color = 'green')
    plt.xlabel('Epoch')
    plt.ylabel('Losses')
    plt.legend(loc='best')
    plt.title('Train and test losses')
    plt.savefig('loss_train_test_BUSI.png')

    plt.figure(figsize=(10, 6))
    plt.plot(X, test_acc_list, label='test_accuracy', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('test acc')
    plt.legend(loc='best')
    plt.title('test_accuracy')
    plt.savefig('test_acc_BUSI.png')


def test_BUSI(model, test_loader, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0.0  # 正确数
    print('Testing...')
    with torch.no_grad():
        for batch_id, (images, labels) in tqdm(enumerate(test_loader), desc='Test',
                                               total=len(test_loader)):
            images, labels = images.to(device), labels.to(device, torch.int64)
            # 输出
            _, outputs = model(images)
            # 损失
            loss = criterion(outputs, labels)
            # 累计损失
            loss_item = loss.item()
            total_loss += loss_item
            # 获取预测概率最大值的索引
            _, predicted = torch.max(outputs, dim=1)
            # 累计正确预测的数
            correct += predicted.eq(labels.view_as(predicted)).sum().item()
            # 错误分类的图片
            # misclassified_images(predicted, writer, labels, images, outputs, epoch)

        # 平均损失
        # avg_loss = total_loss / len(test_loader)
        # 计算正确率
        accuracy = 100 * correct / len(test_loader.dataset)
        # 将test的结果写入write
        # writer.add_scalar("Test Loss", total_loss, epoch)
        # writer.add_scalar("Accuracy", accuracy, epoch)
        # writer.flush()
        return total_loss, accuracy


def train_ISIC():
    model = get_model_for_ISIC().to(device)
    train_set, train_loader, test_set, test_loader = load_data_ISIC()
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(params=model.parameters(), lr=0.001, weight_decay=0.)
    # keras lr decay equivalent
    lr_decay = 3e-3
    fcn = lambda step: 1. / (1. + lr_decay * step)
    scheduler = LambdaLR(optimizer, lr_lambda=fcn)

    epochs = 40
    best_loss = 1e10
    best_acc = 0
    train_loss_list = []
    test_loss_list = []
    # train_acc_list = []
    test_acc_list = []

    print('ISIC .. Start training...')
    for epoch in range(epochs):
        print(f'\nEpoch {epoch} / {epochs}')
        model.train()
        total_loss = 0.0
        for batch_id, (images, labels) in tqdm(enumerate(train_loader),
                            desc='[TRAIN] Epoch ' + str(epoch + 1) + '/' + str(epochs),
                            total=len(train_loader)):

            images, labels = images.to(device), labels.to(device)

            # 梯度置0
            optimizer.zero_grad()
            # 模型输出
            _, outputs = model(images)
            # print('model output:', outputs)
            # print('output type:', outputs.dtype)
            # print('label type:', labels.dtype)
            # print('outputs: ', outputs)
            # print('labels:', labels)
            loss = criterion(outputs, labels)
            # 反向传播
            loss.backward()
            # 更新参数
            optimizer.step()
            # 累计损失
            # print('loss.item:', loss.item())
            # loss_item = loss.item()
            total_loss += loss.item()

        scheduler.step()

        # 平均训练损失
        print(f'\nEpoch {epoch} total loss: {total_loss}')
        train_loss = total_loss / len(train_loader)
        train_loss_list.append(train_loss)

        # test
        test_loss, test_acc = test_ISIC(model, test_loader, criterion)
        test_loss_list.append(test_loss)
        test_acc_list.append(test_acc)
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), 'ISIC_model_new.pth')

        if test_loss < best_loss:
            best_loss = test_loss


        print('Epoch: ' + str(epoch + 1) + '/' + str(epochs),
              'train_loss:', train_loss,
              'test_loss:', test_loss,
              'test_acc:', test_acc)

    print('Train finished')
    print('Best acc:', best_acc)
    with open('train_ISIC.txt', 'w') as f:
        f.write(f'Best acc {best_acc}')
    print('Best loss:', best_loss)
    plt.figure(figsize=(10, 6))
    X = [x for x in range(epochs)]
    plt.plot(X, train_loss_list, label = 'train_loss', color = 'red')
    plt.plot(X, test_loss_list, label = 'test_loss', color = 'green')
    plt.xlabel('Epoch')
    plt.ylabel('Losses')
    plt.legend(loc='best')
    plt.title('Train and test losses')
    plt.savefig('loss_train_test_ISIC.png')

    plt.figure(figsize=(10, 6))
    plt.plot(X, test_acc_list, label='test_accuracy', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('test acc')
    plt.legend(loc='best')
    plt.title('test_accuracy')
    plt.savefig('test_acc_ISIC.png')


def test_ISIC(model, test_loader, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0.0  # 正确数
    print('Testing...')
    with torch.no_grad():
        for batch_id, (images, labels) in tqdm(enumerate(test_loader), desc='Test',
                                               total=len(test_loader)):
            images, labels = images.to(device), labels.to(device)
            # 输出
            _, outputs = model(images)
            # 损失
            loss = criterion(outputs, labels)
            # 累计损失
            loss_item = loss.item()
            total_loss += loss_item
            # 获取预测概率最大值的索引
            _, predicted = torch.max(outputs, dim=1)
            # 累计正确预测的数
            correct += predicted.eq(labels.view_as(predicted)).sum().item()
            # 错误分类的图片
            # misclassified_images(predicted, writer, labels, images, outputs, epoch)

        # 平均损失
        avg_loss = total_loss / len(test_loader)
        # 计算正确率
        accuracy = 100 * correct / len(test_loader.dataset)
        # 将test的结果写入write
        # writer.add_scalar("Test Loss", total_loss, epoch)
        # writer.add_scalar("Accuracy", accuracy, epoch)
        # writer.flush()
        return avg_loss, accuracy



if __name__ == '__main__':
    # train_set, train_loader, test_set, test_loader, train_label = load_data_chest()
    # print(train_loader.dataset)
    # print(test_loader.dataset)
    # print(train_label)
    train_ISIC()