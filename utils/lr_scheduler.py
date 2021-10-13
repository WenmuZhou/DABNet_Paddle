from paddle.optimizer import lr


class WarmupPolyLR(object):
    """
    Cosine learning rate decay
    lr = 0.05 * (math.cos(epoch * (math.pi / epochs)) + 1)
    Args:
        lr(float): initial learning rate
        step_each_epoch(int): steps each epoch
        epochs(int): total training epochs
        last_epoch (int, optional):  The index of last epoch. Can be set to restart training. Default: -1, means initial learning rate.
    """

    def __init__(self,
                 learning_rate,
                 step_each_epoch,
                 epochs,
                 warmup_epoch=0,
                 power=0.9,
                 end_lr=0,
                 cycle=False,
                 last_epoch=-1,
                 **kwargs):
        super(WarmupPolyLR, self).__init__()
        self.learning_rate = learning_rate
        self.total_step = step_each_epoch * epochs
        self.last_epoch = last_epoch
        self.warmup_epoch = round(warmup_epoch * step_each_epoch)
        self.power = power
        self.end_lr = end_lr
        self.cycle = cycle

    def __call__(self):
        learning_rate = lr.PolynomialDecay(
            learning_rate=self.learning_rate,
            decay_steps=self.total_step,
            end_lr=self.end_lr,
            power=self.power,
            cycle=self.cycle,
            last_epoch=self.last_epoch)
        if self.warmup_epoch > 0:
            learning_rate = lr.LinearWarmup(
                learning_rate=learning_rate,
                warmup_steps=self.warmup_epoch,
                start_lr=0.0,
                end_lr=self.learning_rate,
                last_epoch=self.last_epoch)
        return learning_rate
