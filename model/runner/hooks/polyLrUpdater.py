from configs.cfg_run import optimizer


class PolyLrUpdater:
    def __init__(self,
                 by_epoch=False,
                 power=1.,
                 min_lr=0.,
                 warmup=None,
                 warmup_iters=0,
                 warmup_ratio=0.1,
                 warmup_by_epoch=False):
        # validate the "warmup" argument
        if warmup is not None:
            if warmup not in ['constant', 'linear', 'exp']:
                raise ValueError(f'"{warmup}" is not a supported type for warming up, valid types are "constant" and "linear"')
        if warmup is not None:
            assert warmup_iters > 0, '"warmup_iters" must be a positive integer'
            assert 0 < warmup_ratio <= 1.0, '"warmup_ratio" must be in range (0,1]'
        # ----------------- PolyLrUpdater -----------------
        self.power = power
        self.min_lr = min_lr

        self.by_epoch = by_epoch
        self.warmup = warmup
        self.warmup_iters = warmup_iters
        self.warmup_ratio = warmup_ratio
        self.warmup_by_epoch = warmup_by_epoch

        if self.warmup_by_epoch:
            self.warmup_epochs = self.warmup_iters
            self.warmup_iters = None
        else:
            self.warmup_epochs = None

        self.base_lr = []      # initial lr for all param groups
        self.regular_lr = []   # expected lr if no warming up is performed

        # --------------- iterbasedrunner -----------------
        # self.epoch = 0
        # self.iter = 0
        # self.inner_iter = 0
        # self.max_epochs = 0
        # self.max_iters = 0
        #
        # self.model = None
        # self.data_loader = None
        # # self.optimizer = optimizer
        # self.logger = None
        # self.meta = None
        # self.work_dir = None
        # self.rank, self.world_size = None, None
        # self.timestamp = None
        # self.mode = None
        # self.log_buffer = None

    def _set_lr(self, optimizer, lr_groups):
        if isinstance(optimizer, dict):
            for k, optim in optimizer.items():
                for param_group, lr in zip(optim.param_groups, lr_groups[k]):
                    param_group['lr'] = lr
        else:
            for param_group, lr in zip(optimizer.param_groups, lr_groups):
                param_group['lr'] = lr

    def get_lr(self, epoch, max_epochs, iiter, max_iters, base_lr):
        if self.by_epoch:
            progress = epoch
            max_progress = max_epochs
        else:
            progress = iiter
            max_progress = max_iters
        coeff = (1 - progress / max_progress)**self.power
        return (base_lr - self.min_lr) * coeff + self.min_lr

    def get_regular_lr(self, epoch, max_epochs, iiter, max_iters, optimizer):
        if isinstance(optimizer, dict):
            lr_groups = {}
            for k in optimizer.keys():
                _lr_group = [self.get_lr(epoch, max_epochs, iiter, max_iters, _base_lr) for _base_lr in self.base_lr[k]]
                lr_groups.update({k: _lr_group})
            return lr_groups
        else:
            return [self.get_lr(epoch, max_epochs, iiter, max_iters, _base_lr) for _base_lr in self.base_lr]

    def get_warmup_lr(self, cur_iters):

        def _get_warmup_lr(cur_iters, regular_lr):
            if self.warmup == 'constant':
                warmup_lr = [_lr * self.warmup_ratio for _lr in regular_lr]
            elif self.warmup == 'linear':
                k = (1 - cur_iters / self.warmup_iters) * (1 - self.warmup_ratio)
                warmup_lr = [_lr * (1 - k) for _lr in regular_lr]
            elif self.warmup == 'exp':
                k = self.warmup_ratio**(1 - cur_iters / self.warmup_iters)
                warmup_lr = [_lr * k for _lr in regular_lr]
            return warmup_lr

        if isinstance(self.regular_lr, dict):
            lr_groups = {}
            for key, regular_lr in self.regular_lr.items():
                lr_groups[key] = _get_warmup_lr(cur_iters, regular_lr)
            return lr_groups
        else:
            return _get_warmup_lr(cur_iters, self.regular_lr)

    def before_run(self, optimizer):
        if isinstance(optimizer, dict):
            self.base_lr = {}
            for k, optim in optimizer.items():
                for group in optim.param_groups:
                    group.setdefault('initial_lr', group['lr'])
                _base_lr = [group['initial_lr'] for group in optim.param_groups]
                self.base_lr.update({k: _base_lr})
        else:
            for group in optimizer.param_groups:
                group.setdefault('initial_lr', group['lr'])
            self.base_lr = [group['initial_lr'] for group in optimizer.param_groups]

    def before_train_epoch(self, epoch, max_epochs, iiter, max_iters, data_loader, optimizer):
        if self.warmup_iters is None:
            epoch_len = len(data_loader)
            self.warmup_iters = self.warmup_epochs * epoch_len

        if not self.by_epoch:
            return

        self.regular_lr = self.get_regular_lr(epoch, max_epochs, iiter, max_iters, optimizer)
        self._set_lr(optimizer, self.regular_lr)

    def before_train_iter(self, epoch, max_epochs, iiter, max_iters, optimizer):
        cur_iter = iiter
        if not self.by_epoch:
            self.regular_lr = self.get_regular_lr(epoch, max_epochs, iiter, max_iters, optimizer)
            if self.warmup is None or cur_iter >= self.warmup_iters:
                self._set_lr(optimizer, self.regular_lr)
            else:
                warmup_lr = self.get_warmup_lr(cur_iter)
                self._set_lr(optimizer, warmup_lr)
        elif self.by_epoch:
            if self.warmup is None or cur_iter > self.warmup_iters:
                return
            elif cur_iter == self.warmup_iters:
                self._set_lr(optimizer, self.regular_lr)
            else:
                warmup_lr = self.get_warmup_lr(cur_iter)
                self._set_lr(optimizer, warmup_lr)



