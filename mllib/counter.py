class EpsCounter:
    """エピソードカウンター
    """
    def __init__(self, ls):
        self.ls = ls

        self.dpts_curr = 0
        self.dpts_eval_next = ls.dpts_eval if ls.skip_init_eval else 0
        self.dpts_snapshot_next = ls.dpts_snapshot
        self.dpts_train_log_next = 0
        self.dpts_update_lr_next = 0

        self.loss_stat_train_log = []
        self.loss_stat_update_lr = []
        self.loss_terms_stat_train_log = {}
        
        self.dpts_update_lr_prev = None
        self.loss_update_lr_prev = None

    def do_eval(self):
        return self.dpts_curr >= self.dpts_eval_next

    def incr_eval(self):
        dpts_eval_curr = self.dpts_eval_next
        self.dpts_eval_next = (self.dpts_curr // self.ls.dpts_eval + 1) * self.ls.dpts_eval
        return dpts_eval_curr

    def do_snapshot(self):
        return self.dpts_curr >= self.dpts_snapshot_next

    def incr_snapshot(self):
        dpts_snapshot_curr = self.dpts_snapshot_next
        self.dpts_snapshot_next = (self.dpts_curr // self.ls.dpts_snapshot + 1) * self.ls.dpts_snapshot
        return dpts_snapshot_curr

    def do_train_log(self):
        return self.dpts_curr >= self.dpts_train_log_next

    def incr_train_log(self):
        dpts_train_log_curr = self.dpts_train_log_next
        self.dpts_train_log_next = (self.dpts_curr // self.ls.dpts_train_log + 1) * self.ls.dpts_train_log

        loss = np.mean(self.loss_stat_train_log)
        loss_terms = {k: np.mean(v) for k, v in self.loss_terms_stat_train_log.items()}
        self.loss_stat_train_log = []
        self.loss_terms_stat_train_log = {}

        return dpts_train_log_curr, loss, loss_terms

    def incr_train_stat(self, result):
        self.dpts_curr += result['dpts']
        self.loss_stat_train_log.append(result['loss'])
        self.loss_stat_update_lr.append(result['loss'])
        for k, v in result['loss_terms'].items():
            self.loss_terms_stat_train_log.setdefault(k, [])
            self.loss_terms_stat_train_log[k].append(v)

    def do_update_lr(self):
        return self.dpts_curr >= self.dpts_update_lr_next

    def incr_update_lr(self):
        dpts_update_lr_curr = self.dpts_update_lr_next
        self.dpts_update_lr_next = (self.dpts_curr // self.ls.dpts_update_lr + 1) * self.ls.dpts_update_lr

        loss = np.mean(self.loss_stat_update_lr)
        self.loss_stat_update_lr = []

        gain = None
        if self.dpts_update_lr_prev is not None:
            gain = -np.log(loss/self.loss_update_lr_prev)/(self.dpts_curr-self.dpts_update_lr_prev)

        self.dpts_update_lr_prev = self.dpts_curr
        self.loss_update_lr_prev = loss

        return dpts_update_lr_curr, gain

    def finish_training(self):
        return self.dpts_curr >= self.ls.dpts_train