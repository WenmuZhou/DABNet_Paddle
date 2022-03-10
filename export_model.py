# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, "..")))

from argparse import ArgumentParser
import paddle
from paddle.jit import to_static

from utils.utils import init_logger
from builders.model_builder import build_model

def main(args):
    logger = init_logger()
    # build post process
    
    model = build_model(args.model, num_classes=args.classes)
    if args.checkpoint:
        if os.path.isfile(args.checkpoint):
            logger.info("=====> loading checkpoint '{}'".format(args.checkpoint))
            checkpoint = paddle.load(args.checkpoint)
            model.set_state_dict(checkpoint['model'])
            # model.load_state_dict(convert_state_dict(checkpoint['model']))
        else:
            print("=====> no checkpoint found at '{}'".format(args.checkpoint))
            raise FileNotFoundError("no checkpoint found at '{}'".format(args.checkpoint))
    model.eval()

    infer_shape = [3, -1, -1]
    model = to_static(
        model,
        input_spec=[
            paddle.static.InputSpec(
                shape=[None] + infer_shape, dtype="float32")
        ])
    os.makedirs(args.save_path,exist_ok=True)
    paddle.jit.save(model, os.path.join(args.save_path, "inference"))
    logger.info("inference model is saved to {}".format(os.path.join(args.save_path, "inference")))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--model', default="DABNet", help="model name: Context Guided Network (CGNet)")
    parser.add_argument('--checkpoint', type=str,
                        default="./checkpoint/camvid/DABNetbs16gpu1_trainval/model_1000.pth",
                        help="use the file to load the checkpoint for evaluating or testing ")
    parser.add_argument('--dataset', default="cityscapes", help="dataset: cityscapes or camvid")
    parser.add_argument('--save_path', default="inference_models", help="inference model save path")
    args = parser.parse_args()
    if args.dataset == 'cityscapes':
        args.classes = 19
    elif args.dataset == 'camvid':
        args.classes = 11
    else:
        raise NotImplementedError(
            "This repository now supports two datasets: cityscapes and camvid, %s is not included" % args.dataset)
    main(args)
