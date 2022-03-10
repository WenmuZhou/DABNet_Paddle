import logging
import os
import time

import timeit
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser

import paddle
import paddle.nn as nn

# user
from builders.model_builder import build_model
from builders.dataset_builder import build_dataset_train
from utils.utils import setup_seed, init_weight, netParams, init_logger
from utils.metric import get_iou
from utils.loss import CrossEntropyLoss2d, ProbOhemCrossEntropy2d
from utils.lr_scheduler import WarmupPolyLR

GLOBAL_SEED = 1234


def val(args, val_loader, model, logger):
    """
    args:
      val_loader: loaded for validation dataset
      model: model
    return: mean IoU and IoU class
    """
    # evaluation mode
    model.eval()
    total_batches = len(val_loader)

    data_list = []
    for i, (input, label, size, name) in enumerate(val_loader):
        start_time = time.time()
        with paddle.no_grad():
            output = model(input)
        time_taken = time.time() - start_time
        if (i + 1) % 100 == 0:
            logger.info("[{}/{}]  time: {:.4f}".format(i + 1, total_batches, time_taken))
        output = output[0].numpy()
        gt = label.numpy()[0].astype(np.uint8)
        output = output.transpose(1, 2, 0)
        output = np.asarray(np.argmax(output, axis=2), dtype=np.uint8)
        data_list.append([gt.flatten(), output.flatten()])

    meanIoU, per_class_iu = get_iou(data_list, args.classes)
    model.train()
    return meanIoU, per_class_iu


def train(args, train_loader, model, criterion, optimizer, scheduler, epoch, logger):
    """
    args:
       train_loader: loaded for training dataset
       model: model
       criterion: loss function
       optimizer: optimization algorithm, such as ADAM or SGD
       epoch: epoch number
    return: average loss, per class IoU, and mean IoU
    """
    epoch_loss = []

    total_batches = len(train_loader)
    logger.info("=====> the number of iterations per epoch: {}".format(total_batches))
    st = time.time()
    for iteration, batch in enumerate(train_loader, 0):
        lr = optimizer.get_lr()

        start_time = time.time()
        images, labels, _, _ = batch
        output = model(images)
        loss = criterion(output, labels)

        optimizer.clear_grad()  # set the grad to zero
        loss.backward()
        optimizer.step()
        scheduler.step()
        epoch_loss.append(loss.item())
        time_taken = time.time() - start_time

        if (iteration + 1) % args.print_batch_step == 0:
            logger.info('=====> epoch[{}/{}] iter: [{}/{}] cur_lr: {:.6f} loss: {:.6f} time:{:.4f}'.format(epoch + 1,
                                                                                                           args.max_epochs,
                                                                                                           iteration + 1,
                                                                                                           total_batches,
                                                                                                           lr,
                                                                                                           loss.item(),
                                                                                                           time_taken))

    time_taken_epoch = time.time() - st
    remain_time = time_taken_epoch * (args.max_epochs - 1 - epoch)
    m, s = divmod(remain_time, 60)
    h, m = divmod(m, 60)
    logger.info("Remaining training time = {} hour {} minutes {} seconds".format(h, m, s))

    average_epoch_loss_train = sum(epoch_loss) / len(epoch_loss)

    return average_epoch_loss_train, lr


def train_model(args, logger):
    """
    args:
       args: global arguments
    """
    h, w = map(int, args.input_size.split(','))
    args.input_size = (h, w)
    logger.info("=====> input size:{}".format(args.input_size))

    logger.info(args)

    if args.cuda:
        logger.info("=====> use gpu id: '{}'".format(args.gpus))
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
        if not paddle.is_compiled_with_cuda():
            raise Exception("No GPU found or Wrong gpu id, please run without --cuda")

    # set the seed
    setup_seed(GLOBAL_SEED)
    logger.info("=====> set Global Seed: {}".format(GLOBAL_SEED))

    # build the model and initialization
    model = build_model(args.model, num_classes=args.classes)
    init_weight(model, nn.initializer.KaimingNormal(), nn.BatchNorm2D, 1e-3, 0.1)

    logger.info("=====> computing network parameters and FLOPs")
    total_paramters = netParams(model)
    logger.info("the number of parameters: {} ==> {} M".format(total_paramters, (total_paramters / 1e6)))

    # load data and data augmentation
    datas, trainLoader, valLoader = build_dataset_train(args)

    logger.info('=====> Dataset statistics')
    logger.info("data['classWeights']: {}".format(datas['classWeights']))
    logger.info('mean and std: {}, {}'.format(datas['mean'], datas['std']))

    # define loss function, respectively
    weight = paddle.to_tensor(datas['classWeights'])

    if args.dataset == 'camvid':
        criteria = CrossEntropyLoss2d(weight=weight, ignore_label=ignore_label)
    elif args.dataset == 'cityscapes':
        min_kept = int(args.batch_size // len(args.gpus) * h * w // 16)
        criteria = ProbOhemCrossEntropy2d(use_weight=True, ignore_label=ignore_label,
                                          thresh=0.7, min_kept=min_kept)
    else:
        raise NotImplementedError(
            "This repository now supports two datasets: cityscapes and camvid, {} is not included".format(args.dataset))

    start_epoch = 1

    # continue training
    if args.checkpoint:
        if os.path.isfile(args.checkpoint):
            checkpoint = paddle.load(args.checkpoint)
            start_epoch = checkpoint['epoch']
            model.set_state_dict(checkpoint['model'])
            logger.info("=====> loaded checkpoint '{}' (epoch {})".format(args.checkpoint, checkpoint['epoch']))
        else:
            logger.info("=====> no checkpoint found at '{}'".format(args.checkpoint))

    model.train()

    logger.info("Parameters: {} Seed: {}".format(str(total_paramters), GLOBAL_SEED))

    # define optimization criteria
    args.per_iter = len(trainLoader)
    scheduler = WarmupPolyLR(learning_rate=args.lr, step_each_epoch=len(trainLoader),
                             epochs=args.max_epochs, warmup_epoch=500 / len(trainLoader), power=0.9)()
    if args.dataset == 'camvid':
        optimizer = paddle.optimizer.Adam(learning_rate=scheduler, parameters=model.parameters(),
                                          weight_decay=2e-4)
    elif args.dataset == 'cityscapes':
        optimizer = paddle.optimizer.Momentum(learning_rate=scheduler, parameters=model.parameters(), momentum=0.9,
                                              weight_decay=1e-4)
    else:
        raise NotImplementedError

    lossTr_list = []
    epoches = []
    mIOU_val_list = []

    best_metric = {'mIOU': 0, 'epoch': 0}
    logger.info('=====> beginning training')
    for epoch in range(start_epoch, args.max_epochs):
        # training
        lossTr, lr = train(args, trainLoader, model, criteria, optimizer, scheduler, epoch, logger)
        lossTr_list.append(lossTr)

        model_file_name = os.path.join(args.savedir, 'latest.params')
        state = {"epoch": epoch + 1, "model": model.state_dict()}
        paddle.save(state, model_file_name)

        # validation
        if epoch % args.eval_epoch == 0 or epoch == (args.max_epochs - 1):
            epoches.append(epoch)
            mIOU_val, per_class_iu = val(args, valLoader, model, logger)
            mIOU_val_list.append(mIOU_val)
            # record train information
            logger.info("Epoch : {} Details".format(str(epoch)))
            logger.info("Epoch No.: {}\tTrain Loss = {:.6f}\t mIOU(val) = {:.6f}\t lr= {:.6f}".format(epoch,
                                                                                                      lossTr,
                                                                                                      mIOU_val, lr))
            if best_metric['mIOU'] < mIOU_val:
                best_metric = {'mIOU': mIOU_val, 'epoch': epoch + 1}
                model_file_name = os.path.join(args.savedir, 'best.params')
                paddle.save(state, model_file_name)
            logger.info('cur mIOU: {:.6f}, best mIOU: {:.6f}'.format(mIOU_val, best_metric['mIOU']))
        else:
            # record train information
            logger.info("Epoch : " + str(epoch) + ' Details')
            logger.info("Epoch No.: {}\tTrain Loss = {:.6f}\t lr= {:.6f}".format(epoch, lossTr, lr))

        # draw plots for visualization
        if epoch % 50 == 0 or epoch == (args.max_epochs - 1):
            # Plot the figures per 50 epochs
            fig1, ax1 = plt.subplots(figsize=(11, 8))

            ax1.plot(range(start_epoch, epoch + 1), lossTr_list)
            ax1.set_title("Average training loss vs epochs")
            ax1.set_xlabel("Epochs")
            ax1.set_ylabel("Current loss")

            plt.savefig(args.savedir + "loss_vs_epochs.png")

            plt.clf()

            fig2, ax2 = plt.subplots(figsize=(11, 8))

            ax2.plot(epoches, mIOU_val_list, label="Val IoU")
            ax2.set_title("Average IoU vs epochs")
            ax2.set_xlabel("Epochs")
            ax2.set_ylabel("Current IoU")
            plt.legend(loc='lower right')

            plt.savefig(args.savedir + "iou_vs_epochs.png")

            plt.close('all')


if __name__ == '__main__':
    start = timeit.default_timer()
    parser = ArgumentParser()
    parser.add_argument('--model', default="DABNet", help="model name: Context Guided Network (CGNet)")
    parser.add_argument('--dataset', default="cityscapes", help="dataset: cityscapes or camvid")
    parser.add_argument('--data_root', default="", help="dataset folder")
    parser.add_argument('--train_file', default="dataset/cityscapes/cityscapes_train_list.txt", help="dataset folder")
    parser.add_argument('--val_file', default="dataset/cityscapes/cityscapes_val_list.txt", help="dataset folder")
    parser.add_argument('--inform_data_file', default="dataset/inform/cityscapes_inform.pkl", help="dataset folder")
    parser.add_argument('--max_epochs', type=int, default=1000,
                        help="the number of epochs: 300 for train set, 350 for train+val set")
    parser.add_argument('--input_size', type=str, default="512,1024", help="input size of model")
    parser.add_argument('--random_mirror', type=bool, default=True, help="input image random mirror")
    parser.add_argument('--random_scale', type=bool, default=True, help="input image resize 0.5 to 2")
    parser.add_argument('--num_workers', type=int, default=4, help=" the number of parallel threads")
    parser.add_argument('--lr', type=float, default=4.5e-2, help="initial learning rate")
    parser.add_argument('--batch_size', type=int, default=8, help="the batch size is set to 16 for 2 GPUs")
    parser.add_argument('--savedir', default="./checkpoint/", help="directory to save the model snapshot")
    parser.add_argument('--checkpoint', type=str, default="",
                        help="use this file to load last checkpoint for continuing training")
    parser.add_argument('--classes', type=int, default=19,
                        help="the number of classes in the dataset. 19 and 11 for cityscapes and camvid, respectively")
    parser.add_argument('--cuda', type=bool, default=True, help="running on CPU or GPU")
    parser.add_argument('--gpus', type=str, default="0", help="default GPU devices (0,1)")
    parser.add_argument('--print_batch_step', type=int, default=10, help="print ")
    parser.add_argument('--eval_epoch', type=int, default=50, help="print ")
    args = parser.parse_args()

    if args.dataset == 'cityscapes':
        args.classes = 19
        args.input_size = '512,1024'
        ignore_label = 255
    elif args.dataset == 'camvid':
        args.classes = 11
        args.input_size = '360,480'
        ignore_label = 11
    else:
        raise NotImplementedError(
            "This repository now supports two datasets: cityscapes and camvid, %s is not included" % args.dataset)

    if not os.path.exists(args.savedir):
        os.makedirs(args.savedir)

    logFileLoc = os.path.join(args.savedir, 'train.log')
    logger = init_logger(logFileLoc)
    train_model(args, logger)
    end = timeit.default_timer()
    hour = 1.0 * (end - start) / 3600
    minute = (hour - int(hour)) * 60
    logger.info("training time: %d hour %d minutes" % (int(hour), int(minute)))
