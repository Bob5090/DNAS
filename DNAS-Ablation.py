import __init__
import os
import time
import random
import argparse
import json
import itertools
import copy
import numpy as np
from typing import List
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from utils import config, set_seed, Wandb, generate_exp_directory
from nats_bench import create
from xautodl.models.cell_operations import NAS_BENCH_201
from xautodl.models import get_cell_based_tiny_net, get_search_spaces
from xautodl.log_utils import AverageMeter, time_string
from xautodl.utils import count_parameters_in_MB, obtain_accuracy
from xautodl.procedures import get_optim_scheduler, prepare_logger, save_checkpoint
from xautodl.datasets import get_datasets, get_nas_search_loaders
from xautodl.config_utils import dict2config

NONE_ENCODING = [1, 0, 0, 0, 0]


def train_func(
        train_loader,
        model,
        criterion,
        scheduler,
        w_optimizer,
        epoch_str,
        print_freq,
        logger,
):
    data_time, batch_time = AverageMeter(), AverageMeter()
    base_losses = AverageMeter()
    end = time.time()
    model.train()
    # four inputs: train, train_label, test, test_label
    for step, (base_inputs, base_targets) in enumerate(
            train_loader
    ):
        scheduler.update(None, 1.0 * step / len(train_loader))
        base_inputs = base_inputs.cuda(non_blocking=True)
        base_targets = base_targets.cuda(non_blocking=True)

        # measure data loading time
        data_time.update(time.time() - end)
        model.zero_grad()
        _, logits = model(base_inputs)
        base_loss = criterion(logits, base_targets)
        base_loss.backward()
        w_optimizer.step()
        base_losses.update(base_loss.item(), base_inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if step % print_freq == 0 or step + 1 == len(train_loader):
            Sstr = (
                "*EPOCH* "
                + " [{:}][{:03d}/{:03d}]".format(epoch_str, step, len(train_loader))
            )
            Tstr = "Time {batch_time.val:.2f} ({batch_time.avg:.2f}) Data {data_time.val:.2f} ({data_time.avg:.2f})".format(
                batch_time=batch_time, data_time=data_time
            )
            Wstr = "Base [Loss {loss.val:.3f} ({loss.avg:.3f}) ".format(
                loss=base_losses
            )
            strs = Sstr + " " + Tstr + " " + Wstr
            logger.log(strs)
    return base_losses.avg

def train_func_ablation(
        train_loader,
        model,
        criterion,
        scheduler,
        w_optimizer,
        epoch_str,
        print_freq,
        logger,
        portion
):
    data_time, batch_time = AverageMeter(), AverageMeter()
    base_losses = AverageMeter()
    end = time.time()
    model.train()
    # batch_size = train_loader.batch_size
    # four inputs: train, train_label, test, test_label
    for step, (base_inputs, base_targets) in enumerate(
            train_loader
    ):
        scheduler.update(None, 1.0 * step / len(train_loader))
        base_inputs = base_inputs.cuda(non_blocking=True)
        base_targets = base_targets.cuda(non_blocking=True)

        # sample
        batch_size = base_inputs.size(0)
        sample_size = int(portion * batch_size)
        indices = random.sample(range(batch_size), sample_size)
        sampled_inputs = base_inputs[indices]
        sampled_targets = base_targets[indices]

        # measure data loading time
        data_time.update(time.time() - end)
        model.zero_grad()
        # use sampled inputs
        _, logits = model(sampled_inputs)
        base_loss = criterion(logits, sampled_targets)
        base_loss.backward()
        w_optimizer.step()
        base_losses.update(base_loss.item(), sampled_inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if step % print_freq == 0 or step + 1 == len(train_loader):
            Sstr = (
                "*EPOCH* "
                + " [{:}][{:03d}/{:03d}]".format(epoch_str, step, len(train_loader))
            )
            Tstr = "Time {batch_time.val:.2f} ({batch_time.avg:.2f}) Data {data_time.val:.2f} ({data_time.avg:.2f})".format(
                batch_time=batch_time, data_time=data_time
            )
            Wstr = "Base [Loss {loss.val:.3f} ({loss.avg:.3f}) ".format(
                loss=base_losses
            )
            strs = Sstr + " " + Tstr + " " + Wstr
            logger.log(strs)
    return base_losses.avg



def valid_func(xloader, model):
    data_time, batch_time = AverageMeter(), AverageMeter()
    arch_top1, arch_top5 = AverageMeter(), AverageMeter()
    end = time.time()
    with torch.no_grad():
        model.eval()
        for step, (arch_inputs, arch_targets) in enumerate(xloader):
            arch_targets = arch_targets.cuda(non_blocking=True)
            # measure data loading time
            data_time.update(time.time() - end)
            # prediction
            _, logits = model(arch_inputs.cuda(non_blocking=True))
            # record
            arch_prec1, arch_prec5 = obtain_accuracy(
                logits.data, arch_targets.data, topk=(1, 5)
            )
            arch_top1.update(arch_prec1.item(), arch_inputs.size(0))
            arch_top5.update(arch_prec5.item(), arch_inputs.size(0))
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
    return arch_top1.avg, arch_top5.avg


def train_val_epochs(
                     train_epochs,
                     train_loader, valid_loader,
                     model,
                     portion,
                     criterion, w_scheduler, w_optimizer,
                     show_alpha=False,
                     enable_valid=True
                     ):
    best_a_top1 = 0.
    best_w_loss = np.inf
    for epoch in range(train_epochs):
        # w_scheduler.update(epoch, 0.0)
        epoch_str = "{:03d}-{:03d}".format(epoch + 1, train_epochs)
        search_w_loss = train_func_ablation(train_loader, model, criterion,
                                   w_scheduler, w_optimizer,
                                   epoch_str, config.print_freq, logger,
                                   portion
                                   )
        strs = "[{:}] search [base] : loss={:.2f}".format(
            epoch_str, search_w_loss)
        if search_w_loss < best_w_loss:
            best_w_loss = search_w_loss

        # only validate at the last training epoch
        if enable_valid and epoch == train_epochs - 1:
            search_a_top1, _ = valid_func(valid_loader, model)
            if search_a_top1 > best_a_top1:
                best_a_top1 = search_a_top1
            strs += "search [arch] : accuracy@1={:.2f}%".format(search_a_top1)
        logger.log(strs)
        if show_alpha:
            with torch.no_grad():
                logger.log("{:}".format(model.show_alphas()))
    return best_w_loss, best_a_top1


def check_model_valid(arch_mask, edge2index, max_nodes=4):
    for i in range(1, max_nodes):
        none_flag = True
        for j in range(i):
            node_str = "{:}<-{:}".format(i, j)
            with torch.no_grad():
                weights = arch_mask[edge2index[node_str]]
                none_flag = none_flag and torch.all(
                    weights == torch.FloatTensor(NONE_ENCODING))
        # a node at least chooses a not-none edge
        if none_flag:
            return False
    return True
    

def single_path_sample_models(model,
                              edge_indices,
                              group_list):
    """
        edge_indicies: list of edge idx. indicate which edges to split
            (if only one edge, then Greedy Search;
            if multiple edges, then Tree Search with depth >1;
            if all edges, then global search at the current decision stage;
        group list: indicate how to group (one list indicates one group, list of groups) for each edge in edge_indicies
                    e.g. [[[0], [1, 2], [3, 4]], [[0], [1, 2], [3, 4]]
    """
    # give the edge indices for spliting.
    model = model.to('cpu')
    arch_mask = model.arch_mask.cpu()
    group_models = []

    # group_list should have the group for each layer
    assert len(edge_indices) == len(group_list)

    # possible sub group idx for each layer.
    n_group_list = [list(range(len(g))) for g in group_list]
    model_group_indicies = [i for i in itertools.product(*n_group_list)]
    model_group_list = []
    for model_i_indicies in model_group_indicies:
        arch_mask_copy = copy.deepcopy(arch_mask)
        model_i_group_list = []
        for idx, edge_i in enumerate(edge_indices):
            arch_mask_copy[edge_i, :] = 0
            group_idx = model_i_indicies[idx]
            op_indicies = group_list[idx][group_idx]
            model_i_group_list.append(op_indicies)

            # check op_indicies is list or not? if list means a mixed op (group)
            if isinstance(op_indicies, list):
                # check if it is a nested list (if True, then hierachical grouping)
                if isinstance(op_indicies[0], list):  # merge nested list.
                    op_indicies = list(itertools.chain(*op_indicies))
                for op_idx in op_indicies:
                    arch_mask_copy[edge_i, op_idx] = 1
            else:
                arch_mask_copy[edge_i, op_indicies] = 1
        valid_model = check_model_valid(
            arch_mask_copy, model.edge2index, model._max_nodes)
        if valid_model:
            model_copy = copy.deepcopy(model)
            model_copy.arch_mask = arch_mask_copy
            model_group_list.append(model_i_group_list)
            group_models.append(model_copy)
    return group_models, model_group_list


def check_single_path_model(model):
    """check wheter each edge choose a single operation.
    """
    n_edges = len(model.arch_mask)
    edge_flag = [False] * n_edges
    for i in range(n_edges):
        edge_flag[i] = model.arch_mask[i].sum() == 1
    model_flag = all(edge_flag)
    return model_flag, edge_flag


def get_groups_from_alphas(supernet):
    arch_parameters = supernet.arch_parameters.cpu()
    valid_flag = arch_parameters>0
    sorted_idx = (torch.argsort(arch_parameters[:, 1:], descending=True) + 1).tolist()
    valid_idx = [[idx for idx in sorted_idx[i] if valid_flag[i, idx]] for i in range(len(sorted_idx))]
    for i, item in enumerate(valid_idx):
        if len(item) == 0:
           valid_idx[i] = [0] 
    # grouping based on alpha values 
    group_lists = []
    for i in range(len(valid_idx)):
        if len(valid_idx[i])>2:    
            group_lists.append([valid_idx[i][:2], valid_idx[i][2:]])
        else:
            group_lists.append(valid_idx[i])
    
    supernet.arch_mask *= valid_flag.to(supernet.arch_mask.dtype).to(supernet.arch_mask.device)
    return group_lists


def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.reset_parameters()
    if isinstance(m, nn.BatchNorm2d):
        m.reset_parameters()


def dfs_search(supernet, edge_flag, group_lists, stage):
    if all(edge_flag):
        return supernet

    edge_to_decide = [i for i, x in enumerate(edge_flag) if not x]
    np.random.shuffle(edge_to_decide)
    for i in range(len(edge_to_decide)):
        edge_idx = edge_to_decide[i]
        group_list = group_lists[edge_idx]

        for group_idx in group_list:
            supernet_copy = copy.deepcopy(supernet)
            arch_mask_copy = supernet_copy.arch_mask
            arch_mask_copy[edge_idx, :] = 0

            if isinstance(group_idx, list):
                if isinstance(group_idx[0], list):
                    group_idx = list(itertools.chain(*group_idx))
                for op_idx in group_idx:
                    arch_mask_copy[edge_idx, op_idx] = 1
            else:
                arch_mask_copy[edge_idx, group_idx] = 1

            valid_model = check_model_valid(
                arch_mask_copy.to('cpu'), supernet_copy.edge2index, supernet_copy._max_nodes)

            if valid_model:
                supernet_copy.arch_mask = arch_mask_copy
                edge_flag[edge_idx] = True

                result = dfs_search(supernet_copy, edge_flag, group_lists, stage + 1)

                if result:
                    return result

                edge_flag[edge_idx] = False  # Backtrack
                supernet_copy = None

    return None

def main_dfs(config):
    logger.log("---------Start DFS Searching---------")
    train_data, valid_data, xshape, class_num = get_datasets(
        config.data.dataset, config.data.data_path, -1)
    config.xshape = xshape
    config.class_num = class_num

    # create nats-bench
    search_space = get_search_spaces(config.search_space, "nats-bench")
    model_config = dict2config(
        dict(
            num_classes=class_num,
            space=search_space,
            **config.model
        ),
        None,
    )
    logger.log("search space : {:}".format(search_space))
    logger.log("model config : {:}".format(model_config))
    supernet = get_cell_based_tiny_net(model_config)
    supernet.set_cal_mode(config.warmup_mode)
    logger.log("{:}".format(supernet))

    # warmup. default: False
    config.epochs = config.warmup_epochs
    config.LR = config.warmup_lr
    config.lr_min = config.warmup_lr_min
    w_optimizer, w_scheduler, criterion = get_optim_scheduler(
        supernet.weights, config, supernet.alphas
    )
    logger.log("w-optimizer : {:}".format(w_optimizer))
    logger.log("w-scheduler : {:}".format(w_scheduler))
    logger.log("criterion   : {:}".format(criterion))
    params = count_parameters_in_MB(supernet)
    logger.log("The parameters of the search model = {:.2f} MB".format(params))
    logger.log("search-space : {:}".format(search_space))
    if bool(config.use_api):
        api = create('DataSets/NATS-tss-v1_0-3ffb9-full', "topology", fast_mode=True, verbose=False)
    else:
        api = None
    logger.log("{:} create API = {:} done".format(time_string(), api))
    supernet, criterion = supernet.cuda(), criterion.cuda()  # use a single GPU

    if config.load_path is not None:
        logger.log(
            "=> loading checkpoint of the last-info '{:}' start".format(
                config.load_path)
        )
        checkpoint = torch.load(config.load_path, map_location='cpu')
        start_epoch = checkpoint["epoch"]
        supernet.load_state_dict(checkpoint["model"])
        w_scheduler.load_state_dict(checkpoint["w_scheduler"])
        w_optimizer.load_state_dict(checkpoint["w_optimizer"])
        logger.log(
            "=> loading checkpoint start with {:}-th epoch.".format(
                start_epoch
            )
        )
    else:
        start_epoch = 0

    _, train_loader, valid_loader = get_nas_search_loaders(
        train_data,
        valid_data,
        config.data.dataset,
        "configs/nas-benchmark/",
        (config.warmup_batch_size, config.test_batch_size),
        config.workers
    )

    # start training
    start_time = (time.time())
    supernet.apply(init_weights)
    # warm up epochs.
    train_val_epochs(config.epochs,
                     train_loader, valid_loader,
                     supernet, config.portion,
                     criterion, w_scheduler, w_optimizer,
                     enable_valid=False
                     )
    # save checkpoint of the warmup
    save_checkpoint(
        {
            "epoch": global_epoch,
            "config": copy.deepcopy(config),
            "model": supernet.state_dict(),
            "w_optimizer": w_optimizer.state_dict(),
            "w_scheduler": w_scheduler.state_dict(),
        },
        os.path.join(config.ckpt_dir, config.logname + '_supernet.pth'),
        logger,
    )

    logger.log("\n\n========== Start DFS Searching ============= ")
    supernet.set_cal_mode('joint')
    torch.cuda.empty_cache()
    _, train_loader, valid_loader = get_nas_search_loaders(
        train_data,
        valid_data,
        config.data.dataset,
        "configs/nas-benchmark/",
        (config.train_batch_size, config.test_batch_size),
        config.workers,
    )
    (test_val_idx, argbest) = (
        0, np.argmin) if 'loss' in config.metric else (1, np.argmax)

    epoch = 0
    config.epochs = config.train_epochs
    config.LR = config.train_lr
    config.LR_min = config.train_lr_min
    depth = config.d_a  # architecture tree expansion depth

    if config.grouping == 'alpha':
        group_lists = get_groups_from_alphas(supernet)
    else:
        assert isinstance(config.grouping, List)
        group_lists = config.grouping * len(supernet.edge2index)

    stages = int(np.ceil(np.log2(len(NAS_BENCH_201))))
    model_flag, edge_flag = check_single_path_model(supernet)
    stage = -1

    logger.log(f'the group list is: {group_lists}')
    stage = 0
    while not all(edge_flag):
        logger.log(f"\n======= Stage: [{stage}] ========")
        logger.log(f"current model alpha and reduce is: \n{supernet.arch_mask}")

        group_lists = get_groups_from_alphas(supernet)
        result = dfs_search(supernet, edge_flag, group_lists, stage)

        if result:
            supernet = result
            logger.log(f"Successfully found a valid architecture at stage {stage}")
            logger.log(f"Final architecture alpha and reduce is: \n{supernet.arch_mask}")
        else:
            logger.log(f"No valid architecture found at stage {stage}")

        logger.log(f"Finish the stage {stage}, current group list: \n{group_lists}\n"
                   f"current edge_flag: {edge_flag}")
        torch.cuda.empty_cache()
        stage += 1
    # the final post procedure : count the time
    genotype = supernet.genotype

    logger.log("\n" + "-" * 100)
    end_time = time.time()
    total_time = end_time - start_time
    logger.log(f"total time: {total_time}")
    if api is not None:
        logger.log("{:}".format(api.query_by_arch(genotype, "200")))
    # first, query the acc before training.
    arch_index = api.query_index_by_arch(genotype)
    cifar10_score = api.arch2infos_dict[arch_index]['200'].get_metrics(
        'cifar10', 'ori-test')['accuracy']
    cifar100_score = api.arch2infos_dict[arch_index]['200'].get_metrics(
        'cifar100', 'x-test')['accuracy']
    imagenet16_score = api.arch2infos_dict[arch_index]['200'].get_metrics(
        'ImageNet16-120', 'x-test')['accuracy']
    # log the acc and the weights.
    summary_writer.add_scalar(f'final/cifar10-test',
                              cifar10_score, global_epoch + 1)
    summary_writer.add_scalar(f'final/cifar100-test',
                              cifar100_score, global_epoch + 1)
    summary_writer.add_scalar(f'final/imagenet16-test',
                              imagenet16_score, global_epoch + 1)
    summary_writer.add_scalar(f'final/model_index',
                              arch_index, global_epoch + 1)

def parse_option():
    parser = argparse.ArgumentParser('search cell')
    parser.add_argument('--cfg', type=str, required=True, help='config file')
    parser.add_argument('--debug', action='store_true',
                        default=False, help='debug mode')
    parser.add_argument('--recursive', type=bool, default=True)
    parser.add_argument('--portion', type=float, require=True, default=0.5)
    args, opts = parser.parse_args()
    config.load(args.cfg, recursive=False)
    config.update(opts)
    config.debug = args.debug
    config.enable_valid = 'loss' not in config.metric
    if config.rand_seed is None or config.rand_seed < 0:
        config.rand_seed = random.randint(1, 100000)
    return args, config


def parse_option_plus():
    parser = argparse.ArgumentParser('Searching cell')
    parser.add_argument('--cfg', type=str, required=True, help='config file')
    parser.add_argument('--debug', action='store_true',
                        default=False, help='debug mode')
    parser.add_argument('--search_space', type=str, required=False, default='tss')
    # parser.add_argument('--epochs', type=int, default=50, required=False)
    # parser.add_argument('--batch_size', type=int, default=600, required=False)
    parser.add_argument('--warmup_epochs', type=int, default=50, required=False)
    parser.add_argument('--portion', type=float, required=True)
    args = parser.parse_args()
    # print('args', args.__dict__)
    opts = args.__dict__
    config.loadSingleFile(args.cfg)
    config.update(opts)
    config.debug = args.debug
    config.enable_valid = 'loss' not in config.metric
    # config.data.dataset = args.dataset
    # config.data.data_path = args.data_path

    if config.rand_seed is None or config.rand_seed < 0:
        config.rand_seed = random.randint(1, 100000)
    return args, config


if __name__ == "__main__":
    start = time.perf_counter()
    opt, config = parse_option_plus()
    # print(config)

    os.environ['TORCH_HOME'] = 'DataSets'
    device = torch.device('cuda')
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    set_seed(config.rand_seed)

    # if config.load_path is None:
    tags = [config.search_space,
            config.data.dataset,
            'DNAS-Ablaion',
            f'Portion{config.portion}',
            f'WE{config.warmup_epochs}', f'WBS{config.warmup_batch_size}',
            ]
    if config.get('note', False):
        tags.append(str(config.note))
    if config.re_init:
        tags.append('reinit')
    tags.append(f'Seed{config.rand_seed}')
    generate_exp_directory(config, tags)
    config.wandb.tags = tags
    # else:  # resume from the existing ckpt and reuse the folder.
    #    resume_exp_directory(config, config.load_path)
    #    config.wandb.tags = ['resume']
    logger = prepare_logger(config)
    # wandb and tensorboard
    cfg_path = os.path.join(config.log_dir, "config.json")
    with open(cfg_path, 'w') as f:
        json.dump(vars(opt), f, indent=2)
        json.dump(vars(config), f, indent=2)
        os.system('cp %s %s' % (opt.cfg, config.log_dir))
    config.cfg_path = cfg_path

    # wandb config
    config.wandb.name = config.logname
    Wandb.launch(config, config.wandb.use_wandb)

    # tensorboard
    summary_writer = SummaryWriter(log_dir=config.log_dir)

    print('output log:')
    logger.log(config)

    global_epoch = 0
    main_dfs(config)
