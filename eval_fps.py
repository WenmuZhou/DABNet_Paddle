import time
import paddle

from argparse import ArgumentParser
from builders.model_builder import build_model


def compute_speed(model, input_size, iteration=100):

    model.eval()
    model = model.cuda()

    input = paddle.randn(list(input_size))

    for _ in range(50):
        model(input)

    print('=========Speed Testing=========')
    t_start = time.time()
    for _ in range(iteration):
        model(input)
    elapsed_time = time.time() - t_start

    speed_time = elapsed_time / iteration * 1000
    fps = iteration / elapsed_time

    print('Elapsed Time: [%.2f s / %d iter]' % (elapsed_time, iteration))
    print('Speed Time: %.2f ms / iter   FPS: %.2f' % (speed_time, fps))
    return speed_time, fps


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument("size", type=str, default="512,1024", help="input size of model")
    parser.add_argument('--num-channels', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--classes', type=int, default=19)
    parser.add_argument('--iter', type=int, default=100)
    parser.add_argument('--model', type=str, default='DABNet')
    parser.add_argument("--gpus", type=str, default="0", help="gpu ids (default: 0)")
    args = parser.parse_args()

    h, w = map(int, args.size.split(','))
    model = build_model(args.model, num_classes=args.classes)
    paddle.set_device('gpu:{}'.format(int(args.gpus)))
    compute_speed(model, (args.batch_size, args.num_channels, h, w), iteration=args.iter)
