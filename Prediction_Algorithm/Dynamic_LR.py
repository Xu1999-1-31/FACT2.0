class DynamicLearningRateScheduler:
    def __init__(self, optimizer, initial_lr=0.0005, min_lr=0.0001, factor=0.5):
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.min_lr = min_lr
        self.factor = factor
        self.initial_loss = None
        self.previous_loss = None
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.initial_lr

    def step(self, loss):
        if self.initial_loss is None:
            self.initial_loss = loss
            self.previous_loss = loss
            return
        
        if loss <= self.initial_loss * 0.1 and loss < self.previous_loss:
            new_lr = max(self.optimizer.param_groups[0]['lr'] * self.factor, self.min_lr)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = new_lr
            self.initial_loss *= 0.1
        
        self.previous_loss = loss