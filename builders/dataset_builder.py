import os
import pickle
from paddle.io import DataLoader
from dataset.cityscapes import CityscapesDataSet, CityscapesTrainInform, CityscapesValDataSet, CityscapesTestDataSet
from dataset.camvid import CamVidDataSet, CamVidValDataSet, CamVidTrainInform, CamVidTestDataSet


def build_dataset_train(args):
    train_data_list = args.train_file
    val_data_list = args.val_file
    inform_data_file = args.inform_data_file

    # inform_data_file collect the information of mean, std and weigth_class
    if not os.path.isfile(inform_data_file):
        print("{} is not found".format(inform_data_file))
        if args.dataset == "cityscapes":
            dataCollect = CityscapesTrainInform(args.data_root, 19, inform_data_file=inform_data_file)
        elif args.dataset == 'camvid':
            dataCollect = CamVidTrainInform(args.data_root, 11, inform_data_file=inform_data_file)
        else:
            raise NotImplementedError(
                "This repository now supports two datasets: cityscapes and camvid, {} is not included".format(args.dataset))

        datas = dataCollect.collectDataAndSave(train_data_list, val_data_list)
        if datas is None:
            print("error while pickling data. Please check.")
            exit(-1)
    else:
        print("find file: ", str(inform_data_file))
        datas = pickle.load(open(inform_data_file, "rb"))

    if args.dataset == "cityscapes":
        trainLoader = DataLoader(
            CityscapesDataSet(args.data_root, train_data_list, crop_size=args.input_size, scale=args.random_scale,
                              mirror=args.random_mirror, mean=datas['mean']),
            batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)

        valLoader = DataLoader(
            CityscapesValDataSet(args.data_root, val_data_list, f_scale=1, mean=datas['mean']),
            batch_size=1, shuffle=True, num_workers=args.num_workers,
            drop_last=True)

        return datas, trainLoader, valLoader

    elif args.dataset == "camvid":
        trainLoader = DataLoader(
            CamVidDataSet(args.data_root, train_data_list, crop_size=args.input_size, scale=args.random_scale,
                          mirror=args.random_mirror, mean=datas['mean']),
            batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,drop_last=True)

        valLoader = DataLoader(
            CamVidValDataSet(args.data_root, val_data_list, f_scale=1, mean=datas['mean']),
            batch_size=1, shuffle=True, num_workers=args.num_workers)

        return datas, trainLoader, valLoader


def build_dataset_test(args, none_gt=False):
    val_data_list = args.val_file
    inform_data_file = args.inform_data_file

    # inform_data_file collect the information of mean, std and weigth_class
    if not os.path.isfile(inform_data_file):
        print("{} is not found".format(inform_data_file))
        if args.dataset == "cityscapes":
            dataCollect = CityscapesTrainInform(args.data_root, 19, inform_data_file=inform_data_file)
        elif args.dataset == 'camvid':
            dataCollect = CamVidTrainInform(args.data_root, 11, inform_data_file=inform_data_file)
        else:
            raise NotImplementedError(
                "This repository now supports two datasets: cityscapes and camvid, {} is not included".format(dataset))
        
        datas = dataCollect.collectDataAndSave(val_data_list)
        if datas is None:
            print("error while pickling data. Please check.")
            exit(-1)
    else:
        print("find file: ", str(inform_data_file))
        datas = pickle.load(open(inform_data_file, "rb"))

    if args.dataset == "cityscapes":
        # for cityscapes, if test on validation set, set none_gt to False
        # if test on the test set, set none_gt to True
        if none_gt:
            testLoader = DataLoader(
                CityscapesTestDataSet(args.data_root, val_data_list, mean=datas['mean']),
                batch_size=1, shuffle=False, num_workers=args.num_workers)
        else:
            testLoader = DataLoader(
                CityscapesValDataSet(args.data_root, val_data_list, mean=datas['mean']),
                batch_size=1, shuffle=False, num_workers=args.num_workers)

        return datas, testLoader

    elif args.dataset == "camvid":
        testLoader = DataLoader(
            CamVidValDataSet(args.data_root, val_data_list, mean=datas['mean']),
            batch_size=1, shuffle=False, num_workers=args.num_workers)

        return datas, testLoader
