import paddle
import paddle.nn as nn
import paddle.nn.functional as F


class CrossEntropyLoss2d(nn.Layer):
    '''
    This file defines a cross entropy loss for 2D images
    '''

    def __init__(self, weight=None, ignore_label=255):
        '''
        :param weight: 1D weight vector to deal with the class-imbalance
        Obtaining log-probabilities in a neural network is easily achieved by adding a LogSoftmax layer in the last layer of your network. 
        You may use CrossEntropyLoss instead, if you prefer not to add an extra layer.
        '''
        super().__init__()

        # self.loss = nn.NLLLoss2d(weight, ignore_index=255)
        self.loss = nn.NLLLoss(weight, ignore_index=ignore_label)

    def forward(self, outputs, targets):
        return self.loss(F.log_softmax(outputs, 1).transpose([0, 2, 3, 1]), targets.astype('int'))


class FocalLoss2d(nn.Layer):
    def __init__(self, alpha=0.5, gamma=2, weight=None, ignore_index=255):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index
        self.ce_fn = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index)

    def forward(self, preds, labels):
        logpt = -self.ce_fn(preds.transpose([0, 2, 3, 1]), labels.astype('int'))
        pt = paddle.exp(logpt)
        loss = -((1 - pt) ** self.gamma) * self.alpha * logpt
        return loss


class ProbOhemCrossEntropy2d(nn.Layer):
    def __init__(self, ignore_label, reduction='mean', thresh=0.6, min_kept=256,
                 down_ratio=1, use_weight=False):
        super(ProbOhemCrossEntropy2d, self).__init__()
        self.ignore_label = ignore_label
        self.thresh = float(thresh)
        self.min_kept = int(min_kept)
        self.down_ratio = down_ratio
        if use_weight:
            weight = paddle.to_tensor(
                [0.8373, 0.918, 0.866, 1.0345, 1.0166, 0.9969, 0.9754, 1.0489,
                 0.8786, 1.0023, 0.9539, 0.9843, 1.1116, 0.9037, 1.0865, 1.0955,
                 1.0865, 1.1529, 1.0507], dtype='float32')
            self.criterion = nn.CrossEntropyLoss(reduction=reduction,
                                                 weight=weight,
                                                 ignore_index=ignore_label)
        else:
            self.criterion = nn.CrossEntropyLoss(reduction=reduction,
                                                 ignore_index=ignore_label)

    def forward(self, pred, target):
        b, c, h, w = pred.shape
        target = target.reshape([-1])
        valid_mask = target != self.ignore_label
        target = target * valid_mask
        num_valid = valid_mask.astype('int').sum()

        prob = F.softmax(pred, axis=1)
        prob.stop_gradient = True
        prob = (prob.transpose([1, 0, 2, 3])).reshape([c, -1])

        if num_valid < self.min_kept:
            pass
        elif num_valid > 0:
            prob[:, valid_mask == 0] = 1
            mask_prob = prob[
                target.astype('int'), paddle.arange(len(target), dtype='int')]
            threshold = self.thresh
            if self.min_kept > 0:
                index = mask_prob.argsort()
                threshold_index = index[min(len(index), self.min_kept) - 1]
                if mask_prob[threshold_index] > self.thresh:
                    threshold = mask_prob[threshold_index]
                kept_mask = mask_prob <= threshold
                target = target * kept_mask
                valid_mask = valid_mask * kept_mask
        target[valid_mask == 0] = self.ignore_label
        target = target.reshape([b, h, w])
        return self.criterion(pred.transpose([0, 2, 3, 1]), target.astype('int'))