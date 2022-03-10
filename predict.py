import os
import time
import paddle
import numpy as np
from argparse import ArgumentParser
# user
from builders.model_builder import build_model
from builders.dataset_builder import build_dataset_test
from utils.utils import save_predict, init_logger


def predict(args, test_loader, model, logger):
    """
    args:
      test_loader: loaded for test dataset, for those that do not provide label on the test set
      model: model
    return: class IoU and mean IoU
    """
    # evaluation or test mode
    model.eval()
    total_batches = len(test_loader)
    for i, (input, size, name) in enumerate(test_loader):
        start_time = time.time()
        with paddle.no_grad():
            output = model(input)
        time_taken = time.time() - start_time
        logger.info('[%d/%d]  time: %.2f' % (i + 1, total_batches, time_taken))
        output = output.numpy()[0].transpose(1, 2, 0)
        output = np.asarray(np.argmax(output, axis=2), dtype=np.uint8)

        # Save the predict greyscale output for Cityscapes official evaluation
        # Modify image name to meet official requirement
        name[0] = name[0].rsplit('_', 1)[0] + '*'
        save_predict(output, None, name[0], args.dataset, args.save_seg_dir,
                     output_grey=False, output_color=True, gt_color=False)


def test_model(args, logger):
    """
     main function for testing
     param args: global arguments
     return: None
    """
    logger.info(args)

    if args.cuda:
        logger.info("=====> use gpu id: '{}'".format(args.gpus))
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
        if not paddle.is_compiled_with_cuda():
            raise Exception("no GPU found or wrong gpu id, please run without --cuda")

    # build the model
    model = build_model(args.model, num_classes=args.classes)

    if not os.path.exists(args.save_seg_dir):
        os.makedirs(args.save_seg_dir)

    # load the test set
    datas, testLoader = build_dataset_test(args, none_gt=True)

    if args.checkpoint:
        if os.path.isfile(args.checkpoint):
            logger.info("=====> loading checkpoint '{}'".format(args.checkpoint))
            checkpoint = paddle.load(args.checkpoint)
            model.set_state_dict(checkpoint['model'])
        else:
            logger.info("=====> no checkpoint found at '{}'".format(args.checkpoint))
            raise FileNotFoundError("no checkpoint found at '{}'".format(args.checkpoint))

    logger.info("=====> beginning testing")
    logger.info("test set length: {}".format(len(testLoader)))
    predict(args, testLoader, model, logger)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model', default="DABNet", help="model name: Context Guided Network (CGNet)")
    parser.add_argument('--dataset', default="cityscapes", help="dataset: cityscapes or camvid")
    parser.add_argument('--data_root', default="", help="dataset folder")
    parser.add_argument('--val_file', default="dataset/cityscapes/cityscapes_val_list.txt", help="dataset folder")
    parser.add_argument('--inform_data_file', default="dataset/inform/cityscapes_inform.pkl", help="dataset folder")
    parser.add_argument('--num_workers', type=int, default=1, help="the number of parallel threads")
    parser.add_argument('--batch_size', type=int, default=1,
                        help=" the batch_size is set to 1 when evaluating or testing")
    parser.add_argument('--checkpoint', type=str,
                        default="",
                        help="use the file to load the checkpoint for evaluating or testing ")
    parser.add_argument('--save_seg_dir', type=str, default="./result/",
                        help="saving path of prediction result")
    parser.add_argument('--cuda', default=True, help="run on CPU or GPU")
    parser.add_argument("--gpus", default="0", type=str, help="gpu ids (default: 0)")
    args = parser.parse_args()

    args.save_seg_dir = os.path.join(args.save_seg_dir, args.dataset, 'predict', args.model)

    if args.dataset == 'cityscapes':
        args.classes = 19
    elif args.dataset == 'camvid':
        args.classes = 11
    else:
        raise NotImplementedError(
            "This repository now supports two datasets: cityscapes and camvid, %s is not included" % args.dataset)
    logger = init_logger()
    test_model(args, logger)
