import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import argparse
import numpy as np
import os

from data.Datasets import build_train_dataset
from GMFlowRGBD.GMFlowRGBD import GMFlow
from Loss import flow_loss_func
from Evaluate import (validate_chairs, validate_things, validate_sintel, validate_kitti,
                      create_sintel_submission, create_kitti_submission, inference_on_dir)

from Utils.Logger import Logger
from Utils import Misc
from Utils.DistanceUtils import get_dist_info, init_dist, setup_for_distributed

# import PIL


def CheckEnv():
    # 在下面的代码行中使用断点来调试脚本。
    print("torch version: ", torch.__version__)  # 注意是双下划线
    # print("PIL Version: ", PIL.__version__)


def get_args_parser():
    parser = argparse.ArgumentParser()

    # 数据集参数
    parser.add_argument('--checkpoint_dir', default='CheckPointsDir', type=str,
                        help='where to save the training log and models')   # 存储训练日志和模型位置
    parser.add_argument('--stage', default='sintel', type=str,
                        help='training stage')                              # 训练阶段(目前是哪个数据集)
    parser.add_argument('--image_size', default=[320, 896], type=int, nargs='+',
                        help='image size for training')                     # 图像的大小, + 号表示 1 或多个参数
    parser.add_argument('--padding_factor', default=16, type=int,           # padding的因子
                        help='the input should be divisible by padding_factor, otherwise do padding')
    parser.add_argument('--max_flow', default=400, type=int,
                        help='exclude very large motions during training')  # 在训练时排除非常大的动作(1帧移动400个像素)
    parser.add_argument('--val_dataset', default=['sintel'], type=str, nargs='+',
                        help='validation dataset')                          # 验证数据集
    parser.add_argument('--with_speed_metric', action='store_true',
                        help='with speed metric when evaluation')           # 评价时采用速度指标

    # 训练参数
    parser.add_argument('--lr', default=4e-4, type=float)                   # 学习率
    parser.add_argument('--batch_size', default=12, type=int)               # Batch Size
    parser.add_argument('--num_workers', default=1, type=int)               # GPU数量
    parser.add_argument('--weight_decay', default=1e-4, type=float)         # 权重衰退(控制模型的复杂度并减少过拟合的风险)
    parser.add_argument('--grad_clip', default=1.0, type=float)             # 梯度剪裁(防止梯度爆炸的问题)
    parser.add_argument('--num_steps', default=100000, type=int)            # 迭代次数
    parser.add_argument('--seed', default=326, type=int)                    # 随机数种子
    parser.add_argument('--summary_freq', default=100, type=int)            # 生成摘要的频率(摘要是一种用于可视化和监控模型训练过程的工具)
    parser.add_argument('--val_freq', default=10000, type=int)              # 训练中进行验证的频率(迭代10000次,验证一次)
    parser.add_argument('--save_ckpt_freq', default=10000, type=int)        # 模型检查点的频率(模型检查点是保存模型的中间状态的文件)
    parser.add_argument('--save_latest_ckpt_freq', default=1000, type=int)  # 保存最新模型检查点频率(最新模型检查点是指保存最近一次训练状态的模型文件)

    # 恢复预训练模型 或者 恢复终端训练
    parser.add_argument('--resume', default=None, type=str,
                        help='resume from pretrain model for finetune or resume from terminated training')
    # strict_resume 参数用于控制在恢复训练时对模型参数的匹配要求。当 strict_resume 设置为 True（严格模式）时,
    # 恢复训练时要求加载的检查点中的模型参数与当前模型的参数完全匹配，包括参数的名称、形状和数据类型等。如果匹配不成功，将会引发错误并终止训练。
    parser.add_argument('--strict_resume', action='store_true')
    # 用于指定在恢复训练时是否重新创建优化器（optimizer）no_resume_optimizer .
    # 设置为 True 时，恢复训练时会忽略检查点中保存的优化器状态，而是重新创建一个新的优化器，并从头开始进行优化。这意味着之前的优化器状态和参数更新历史将被丢弃。
    parser.add_argument('--no_resume_optimizer', action='store_true')

    # GMFlow 模型参数
    parser.add_argument('--num_scales', default=1, type=int,                # 放缩尺度
                        help='basic gmflow model uses a single 1/8 feature, the refinement uses 1/4 feature')
    parser.add_argument('--feature_channels', default=128, type=int)        # 特征通道
    parser.add_argument('--upsample_factor', default=8, type=int)           # 上采用因子
    parser.add_argument('--num_transformer_layers', default=6, type=int)    # 编码器或解码器中使用的 Transformer 层的数量
    parser.add_argument('--num_head', default=1, type=int)                  # 多头注意力，MultiHead的数量
    parser.add_argument('--attention_type', default='swin', type=str)       # 注意力机制的类型:使用 Swin Transformer 模型中的注意力机制类型。
    parser.add_argument('--ffn_dim_expansion', default=4, type=int)         # 用于指定在 Transformer 模型中的前馈神经网络（FFN）层中的维度扩展倍数

    # 多头注意力机制中每个注意力头的分割数量, 在多头注意力机制中，输入特征通常会被分成多个子空间，每个子空间对应一个注意力头.
    # 每个注意力头独立地计算注意力权重，并将它们的结果进行合并以获得最终的注意力表示。
    parser.add_argument('--attn_splits_list', default=[2], type=int, nargs='+',
                        help='number of splits in attention')
    # 指定在自注意力机制中用于计算相关性的半径列表
    # 如果 corr_radius_list 设置为 [1, 2, 3]，则表示在自注意力机制中，每个位置与距离其1个单位、2个单位和3个单位的位置进行相关性计算
    # 当 corr_radius_list 设置为 [-1] 时，模型将使用全局相关性或全连接的方式计算注意力权重。
    # 这意味着每个位置都与所有其他位置进行相关性计算，从而允许模型在全局范围内捕捉到所有位置的信息
    parser.add_argument('--corr_radius_list', default=[-1], type=int, nargs='+',
                        help='correlation radius for matching, -1 indicates global matching')
    # 指定在自注意力机制中用于计算传播的半径列表。在自注意力机制中，除了计算位置之间的相关性外，
    # 还可以通过传播机制将信息从一个位置传递到另一个位置。传播机制可以帮助模型在序列中传递和交互信息，从而增强模型的表示能力。
    # 当 prop_radius_list 设置为 [-1] 时，模型将使用全连接的传播机制，即每个位置都与所有其他位置进行传播计算，从而允许信息在整个序列中自由传递
    parser.add_argument('--prop_radius_list', default=[-1], type=int, nargs='+',
                        help='self-attention radius for flow propagation, -1 indicates global attention')

    # 损失权重: 指定不同损失函数的权重或重要性，以在多个损失函数之间进行平衡或调整
    # 损失权重可以是一个标量值，也可以是一个权重向量或矩阵，其维度与损失函数的数量相匹配。
    # 较高的权重值表示对应损失函数的重要性更高，而较低的权重值表示对应损失函数的重要性较低
    parser.add_argument('--gamma', default=0.9, type=float,
                        help='loss weight')

    # 评估
    parser.add_argument('--eval', action='store_true')                          # 是否评估
    parser.add_argument('--save_eval_to_file', action='store_true')             # 是否存储评估结果到文件中
    parser.add_argument('--evaluate_matched_unmatched', action='store_true')    # 评估模型在匹配和不匹配样本上的性能

    # inference on a directory  在文件夹中推断
    parser.add_argument('--inference_dir', default=None, type=str)
    parser.add_argument('--inference_size', default=None, type=int, nargs='+',
                        help='can specify the inference size')
    parser.add_argument('--dir_paired_data', action='store_true',
                        help='Paired data in a dir instead of a sequence')
    parser.add_argument('--save_flo_flow', action='store_true')
    parser.add_argument('--pred_bidir_flow', action='store_true',
                        help='predict bidirectional flow')
    parser.add_argument('--fwd_bwd_consistency_check', action='store_true',
                        help='forward backward consistency check with bidirection flow')


    parser.add_argument('--submission', action='store_true',                    # 在sintel and kitti 测试集上为上传的进行预测
                        help='submission to sintel or kitti test sets')
    parser.add_argument('--output_path', default='output', type=str,            # 输出地址
                        help='where to save the prediction results')
    parser.add_argument('--save_vis_flow', action='store_true',                 # 存储光流，可视化存储成图片
                        help='visualize flow prediction as .png image')
    parser.add_argument('--no_save_flo', action='store_true',                   # 不将光流文件存储成.flo文件
                        help='not save flow as .flo')

    # 分布式训练参数
    parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument('--distributed', action='store_true')
    parser.add_argument('--launcher', default='none', type=str, choices=['none', 'pytorch'])    # none->非分布式训练
    parser.add_argument('--gpu_ids', default=0, type=int, nargs='+')

    parser.add_argument('--count_time', action='store_true',
                        help='measure the inference time on sintel')

    return parser


def main(args):
    # 不执行评估操作（args.eval 为假）
    # 不执行提交操作（args.submission 为假）
    # 没有指定推理目录（args.inference_dir 为空）
    if not args.eval and not args.submission and args.inference_dir is None:
        if args.local_rank == 0:
            print('pytorch version:', torch.__version__)
            # print('Args = ' , args)
            Misc.save_args(args)
            Misc.check_path(args.checkpoint_dir)
            Misc.save_command(args.checkpoint_dir)

    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    torch.backends.cudnn.benchmark = True


    if args.launcher == 'none':     # 不进行分布式训练
        args.distributed = False
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:                           # 进行分布式训练
        args.distributed = True
        # 调整每个GPU的BatchSize
        assert args.batch_size % torch.cuda.device_count() == 0
        args.batch_size = args.batch_size // torch.cuda.device_count()

        dist_params = dict(backend='nccl')
        init_dist(args.launcher, **dist_params)
        # 使用分布式训练模式重新设置gpu id
        _, world_size = get_dist_info()
        args.gpu_ids = range(world_size)
        device = torch.device('cuda:{}'.format(args.local_rank))

        setup_for_distributed(args.local_rank == 0)

    # 模型
    model = GMFlow(feature_channels=args.feature_channels,
                   num_scales=args.num_scales,
                   upsample_factor=args.upsample_factor,
                   num_head=args.num_head,
                   attention_type=args.attention_type,
                   ffn_dim_expansion=args.ffn_dim_expansion,
                   num_transformer_layers=args.num_transformer_layers,
                   ).to(device)

    # 不执行评估操作（args.eval 为假）
    # 不执行提交操作（args.submission 为假）
    # 没有指定推理目录（args.inference_dir 为假）
    if not args.eval and not args.submission and not args.inference_dir:
        print('Model definition:')
        print(model)

    if args.distributed:            # 如果是分布式训练
        print('It is distributed train')
        model = torch.nn.parallel.DistributedDataParallel(
            model.to(device),
            device_ids=[args.local_rank],
            output_device=args.local_rank)
        model_without_ddp = model.module
    else:                           # 非分布式训练，选择训练的GPU
        print('It is not distributed train')
        if torch.cuda.device_count() > 1:
            print('Use %d GPUs' % torch.cuda.device_count())
            model = torch.nn.DataParallel(model)

            model_without_ddp = model.module
        else:
            print('Current GPU Num = ', torch.cuda.device_count(), 'GPU Name = ', torch.cuda.get_device_name())
            model_without_ddp = model

    num_params = sum(p.numel() for p in model.parameters())
    print('Number of params:', num_params)

    # 满足特定条件（不执行评估操作、不执行提交操作、未指定推理目录），
    # 它会创建一个名为 save_name 的文件，并将其保存在指定的检查点目录中。同时，它还会打印出模型中的参数数量
    if not args.eval and not args.submission and args.inference_dir is None:
        save_name = '%d_parameters' % num_params    # 它使用了格式化字符串，将参数数量 num_params 格式化为整数，并将其与字符串 "_parameters" 连接起来
        #  这行代码打开了一个文件，并立即关闭它。它使用了 os.path.join() 函数来构建文件的完整路径，
        #  将 args.checkpoint_dir（检查点目录）和 save_name（文件名）连接起来。这个操作相当于创建一个空文件，并将其保存在指定的检查点目录中
        open(os.path.join(args.checkpoint_dir, save_name), 'a').close()

    # model_without_ddp.parameters() = 表示从一个没有使用分布式数据并行（DDP）的模型中获取所有的参数。
    optimizer = torch.optim.AdamW(model_without_ddp.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)

    start_epoch = 0
    start_step = 0
    # (中断后)重新开始CheckPoint
    if args.resume:
        print('Load checkpoint: %s' % args.resume)

        loc = 'cuda:{}'.format(args.local_rank)
        checkpoint = torch.load(args.resume, map_location=loc)

        weights = checkpoint['model'] if 'model' in checkpoint else checkpoint

        model_without_ddp.load_state_dict(weights, strict=args.strict_resume)

        if 'optimizer' in checkpoint and 'step' in checkpoint and 'epoch' in checkpoint and not \
                args.no_resume_optimizer:
            print('Load optimizer')
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch']
            start_step = checkpoint['step']

        print('start_epoch: %d, start_step: %d' % (start_epoch, start_step))

    # 评估
    if args.eval:
        val_results = {}

        if 'chairs' in args.val_dataset:
            results_dict = validate_chairs(model_without_ddp,
                                           with_speed_metric=args.with_speed_metric,
                                           attn_splits_list=args.attn_splits_list,
                                           corr_radius_list=args.corr_radius_list,
                                           prop_radius_list=args.prop_radius_list,
                                           )

            val_results.update(results_dict)

        if 'things' in args.val_dataset:
            results_dict = validate_things(model_without_ddp,
                                           padding_factor=args.padding_factor,
                                           with_speed_metric=args.with_speed_metric,
                                           attn_splits_list=args.attn_splits_list,
                                           corr_radius_list=args.corr_radius_list,
                                           prop_radius_list=args.prop_radius_list,
                                           )
            val_results.update(results_dict)

        if 'sintel' in args.val_dataset:
            results_dict = validate_sintel(model_without_ddp,
                                           count_time=args.count_time,
                                           padding_factor=args.padding_factor,
                                           with_speed_metric=args.with_speed_metric,
                                           evaluate_matched_unmatched=args.evaluate_matched_unmatched,
                                           attn_splits_list=args.attn_splits_list,
                                           corr_radius_list=args.corr_radius_list,
                                           prop_radius_list=args.prop_radius_list,
                                           )
            val_results.update(results_dict)

        if 'kitti' in args.val_dataset:
            results_dict = validate_kitti(model_without_ddp,
                                          padding_factor=args.padding_factor,
                                          with_speed_metric=args.with_speed_metric,
                                          attn_splits_list=args.attn_splits_list,
                                          corr_radius_list=args.corr_radius_list,
                                          prop_radius_list=args.prop_radius_list,
                                          )
            val_results.update(results_dict)

        if args.save_eval_to_file:
            Misc.check_path(args.checkpoint_dir)
            val_file = os.path.join(args.checkpoint_dir, 'val_results.txt')
            with open(val_file, 'a') as f:
                f.write('\neval results after training done\n\n')
                metrics = ['chairs_epe', 'chairs_s0_10', 'chairs_s10_40', 'chairs_s40+',
                           'things_clean_epe', 'things_clean_s0_10', 'things_clean_s10_40', 'things_clean_s40+',
                           'things_final_epe', 'things_final_s0_10', 'things_final_s10_40', 'things_final_s40+',
                           'sintel_clean_epe', 'sintel_clean_s0_10', 'sintel_clean_s10_40', 'sintel_clean_s40+',
                           'sintel_final_epe', 'sintel_final_s0_10', 'sintel_final_s10_40', 'sintel_final_s40+',
                           'kitti_epe', 'kitti_f1', 'kitti_s0_10', 'kitti_s10_40', 'kitti_s40+',
                           ]
                eval_metrics = []
                for metric in metrics:
                    if metric in val_results.keys():
                        eval_metrics.append(metric)

                metrics_values = [val_results[metric] for metric in eval_metrics]

                num_metrics = len(eval_metrics)

                # save as markdown format
                f.write(("| {:>20} " * num_metrics + '\n').format(*eval_metrics))
                f.write(("| {:20.3f} " * num_metrics).format(*metrics_values))

                f.write('\n\n')

        return

    # Sintel and KITTI 数据集提交
    if args.submission:
        # NOTE: args.val_dataset is a list
        if args.val_dataset[0] == 'sintel':
            create_sintel_submission(model_without_ddp,
                                     output_path=args.output_path,
                                     padding_factor=args.padding_factor,
                                     save_vis_flow=args.save_vis_flow,
                                     no_save_flo=args.no_save_flo,
                                     attn_splits_list=args.attn_splits_list,
                                     corr_radius_list=args.corr_radius_list,
                                     prop_radius_list=args.prop_radius_list,
                                     )
        elif args.val_dataset[0] == 'kitti':
            create_kitti_submission(model_without_ddp,
                                    output_path=args.output_path,
                                    padding_factor=args.padding_factor,
                                    save_vis_flow=args.save_vis_flow,
                                    attn_splits_list=args.attn_splits_list,
                                    corr_radius_list=args.corr_radius_list,
                                    prop_radius_list=args.prop_radius_list,
                                    )
        else:
            raise ValueError(f'Not supported dataset for submission')

        return

    # 在文件夹中推理
    if args.inference_dir is not None:
        inference_on_dir(model_without_ddp,
                         inference_dir=args.inference_dir,
                         output_path=args.output_path,
                         padding_factor=args.padding_factor,
                         inference_size=args.inference_size,
                         paired_data=args.dir_paired_data,
                         save_flo_flow=args.save_flo_flow,
                         attn_splits_list=args.attn_splits_list,
                         corr_radius_list=args.corr_radius_list,
                         prop_radius_list=args.prop_radius_list,
                         pred_bidir_flow=args.pred_bidir_flow,
                         fwd_bwd_consistency_check=args.fwd_bwd_consistency_check,
                         )

        return

    # 训练数据集
    train_dataset = build_train_dataset(args)
    print('Number of training images:', len(train_dataset))

    # Multi-processing
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset,
            num_replicas=torch.cuda.device_count(),
            rank=args.local_rank)
    else:
        train_sampler = None

    shuffle = False if args.distributed else True
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                               shuffle=shuffle, num_workers=args.num_workers,
                                               pin_memory=True, drop_last=True,
                                               sampler=train_sampler)

    last_epoch = start_step if args.resume and start_step > 0 else -1
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, args.lr,
        args.num_steps + 10,
        pct_start=0.05,
        cycle_momentum=False,
        anneal_strategy='cos',
        last_epoch=last_epoch,
    )

    if args.local_rank == 0:
        summary_writer = SummaryWriter(args.checkpoint_dir)
        logger = Logger(lr_scheduler, summary_writer, args.summary_freq,
                        start_step=start_step)

    total_steps = start_step
    epoch = start_epoch
    print('Start training')



    while total_steps < args.num_steps:
        model.train()

        # mannual change random seed for shuffling every epoch 手动改变洗牌每个迭代的随机种子
        if args.distributed:
            train_sampler.set_epoch(epoch)

        for i, sample in enumerate(train_loader):
            img1, dpt1, img2, dpt2, flow_gt, valid = [x.to(device) for x in sample]

            results_dict = model(img1, dpt1, img2, dpt2,
                                 attn_splits_list=args.attn_splits_list,
                                 corr_radius_list=args.corr_radius_list,
                                 prop_radius_list=args.prop_radius_list,
                                 )

            flow_preds = results_dict['flow_preds']

            loss, metrics = flow_loss_func(flow_preds, flow_gt, valid,
                                           gamma=args.gamma,
                                           max_flow=args.max_flow,
                                           )

            if isinstance(loss, float):
                continue

            if torch.isnan(loss):
                continue

            metrics.update({'total_loss': loss.item()})

            # more efficient zero_grad
            for param in model_without_ddp.parameters():
                param.grad = None

            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            optimizer.step()

            lr_scheduler.step()

            if args.local_rank == 0:
                logger.push(metrics)

                logger.add_image_summary(img1, img2, flow_preds, flow_gt)

            total_steps += 1

            if total_steps % args.save_ckpt_freq == 0 or total_steps == args.num_steps:
                if args.local_rank == 0:
                    checkpoint_path = os.path.join(args.checkpoint_dir, 'step_%06d.pth' % total_steps)
                    torch.save({
                        'model': model_without_ddp.state_dict()
                    }, checkpoint_path)

            if total_steps % args.save_latest_ckpt_freq == 0:
                checkpoint_path = os.path.join(args.checkpoint_dir, 'checkpoint_latest.pth')

                if args.local_rank == 0:
                    torch.save({
                        'model': model_without_ddp.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'step': total_steps,
                        'epoch': epoch,
                    }, checkpoint_path)

            if total_steps % args.val_freq == 0:
                print('Start validation at step : %d' % total_steps)

                val_results = {}
                # support validation on multiple datasets
                if 'chairs' in args.val_dataset:
                    results_dict = validate_chairs(model_without_ddp,
                                                   with_speed_metric=args.with_speed_metric,
                                                   attn_splits_list=args.attn_splits_list,
                                                   corr_radius_list=args.corr_radius_list,
                                                   prop_radius_list=args.prop_radius_list,
                                                   )
                    if args.local_rank == 0:
                        val_results.update(results_dict)

                if 'things' in args.val_dataset:
                    results_dict = validate_things(model_without_ddp,
                                                   padding_factor=args.padding_factor,
                                                   with_speed_metric=args.with_speed_metric,
                                                   attn_splits_list=args.attn_splits_list,
                                                   corr_radius_list=args.corr_radius_list,
                                                   prop_radius_list=args.prop_radius_list,
                                                   )
                    if args.local_rank == 0:
                        val_results.update(results_dict)

                if 'sintel' in args.val_dataset:
                    results_dict = validate_sintel(model_without_ddp,
                                                   count_time=args.count_time,
                                                   padding_factor=args.padding_factor,
                                                   with_speed_metric=args.with_speed_metric,
                                                   evaluate_matched_unmatched=args.evaluate_matched_unmatched,
                                                   attn_splits_list=args.attn_splits_list,
                                                   corr_radius_list=args.corr_radius_list,
                                                   prop_radius_list=args.prop_radius_list,
                                                   )
                    if args.local_rank == 0:
                        val_results.update(results_dict)

                if 'kitti' in args.val_dataset:
                    results_dict = validate_kitti(model_without_ddp,
                                                  padding_factor=args.padding_factor,
                                                  with_speed_metric=args.with_speed_metric,
                                                  attn_splits_list=args.attn_splits_list,
                                                  corr_radius_list=args.corr_radius_list,
                                                  prop_radius_list=args.prop_radius_list,
                                                  )
                    if args.local_rank == 0:
                        val_results.update(results_dict)

                if args.local_rank == 0:
                    logger.write_dict(val_results)

                    # Save validation results
                    val_file = os.path.join(args.checkpoint_dir, 'val_results.txt')
                    with open(val_file, 'a') as f:
                        f.write('step: %06d\n' % total_steps)
                        if args.evaluate_matched_unmatched:
                            metrics = ['chairs_epe',
                                       'chairs_s0_10', 'chairs_s10_40', 'chairs_s40+',
                                       'things_clean_epe', 'things_clean_s0_10', 'things_clean_s10_40',
                                       'things_clean_s40+',
                                       'sintel_clean_epe', 'sintel_clean_matched', 'sintel_clean_unmatched',
                                       'sintel_clean_s0_10', 'sintel_clean_s10_40',
                                       'sintel_clean_s40+',
                                       'sintel_final_epe', 'sintel_final_matched', 'sintel_final_unmatched',
                                       'sintel_final_s0_10', 'sintel_final_s10_40',
                                       'sintel_final_s40+',
                                       'kitti_epe', 'kitti_f1', 'kitti_s0_10', 'kitti_s10_40', 'kitti_s40+',
                                       ]
                        else:
                            metrics = ['chairs_epe', 'chairs_s0_10', 'chairs_s10_40', 'chairs_s40+',
                                       'things_clean_epe', 'things_clean_s0_10', 'things_clean_s10_40',
                                       'things_clean_s40+',
                                       'sintel_clean_epe', 'sintel_clean_s0_10', 'sintel_clean_s10_40',
                                       'sintel_clean_s40+',
                                       'sintel_final_epe', 'sintel_final_s0_10', 'sintel_final_s10_40',
                                       'sintel_final_s40+',
                                       'kitti_epe', 'kitti_f1', 'kitti_s0_10', 'kitti_s10_40', 'kitti_s40+',
                                       ]

                        eval_metrics = []
                        for metric in metrics:
                            if metric in val_results.keys():
                                eval_metrics.append(metric)

                        metrics_values = [val_results[metric] for metric in eval_metrics]

                        num_metrics = len(eval_metrics)

                        # save as markdown format
                        if args.evaluate_matched_unmatched:
                            f.write(("| {:>25} " * num_metrics + '\n').format(*eval_metrics))
                            f.write(("| {:25.3f} " * num_metrics).format(*metrics_values))
                        else:
                            f.write(("| {:>20} " * num_metrics + '\n').format(*eval_metrics))
                            f.write(("| {:20.3f} " * num_metrics).format(*metrics_values))

                        f.write('\n\n')

                model.train()

            if total_steps >= args.num_steps:
                print('Training done')
                return
            
            # if total_steps % 100 == 0:
            #     print('now at step : %d' % total_steps)
        
        epoch += 1


# 按间距中的绿色按钮以运行脚本。
if __name__ == '__main__':
    CheckEnv()
    Parser = get_args_parser()
    args = Parser.parse_args()

    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    main(args)
