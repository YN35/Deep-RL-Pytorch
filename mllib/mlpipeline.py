import time
from pathlib import Path
from datetime import datetime

import shutil
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from .torch_utils import temporal_freeze
from mllib.optimizer import get_opt_func


class MLPipeline:
    def __init__(self, cfg):
        self.cfg = cfg
        
        # Make an output directory.
        now = datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')
        self.outdir = str(Path(self.cfg.mainmodel.outdir).joinpath(f'{self.cfg.mainmodel.task}', f'{now}'))
        Path(self.outdir).mkdir(parents=True, exist_ok=False)
        print('[Directory] ' + self.outdir)
        # Copy the config yaml to the output directory.
        shutil.copy2(self.cfg.info.config_path, self.outdir)
        
        self._epsd = 0
        self.__step = 0
        self._step_sum = 0
        self._train_sum = 0
        self.done_epsd = False
        
        self.t0 = time.perf_counter()
        
        self.writer = SummaryWriter(self.outdir)
        
        self._set_random_seed(self.cfg.mainmodel.seed)
        

        for comp in self.cfg.model.comp.values():
            if len(comp.params) > 0:
                path = comp.params.pop()
                comp._params_curr = Path(self.cfg.model.root).joinpath(path)
            else:
                comp._params_curr = None

                
    def add_step(self):
        self.__step = self.__step + 1
        self._step_sum = self._step_sum + 1
        if self.cfg.mainmodel.max_train_step < self._step_sum:
            self.done_epsd = True
        
        
    def reset_step(self):
        self.__step = 0
        
    def add_epsd(self):
        self._epsd = self._epsd + 1
    
    def train_log(self):
        if self._train_sum % self.cfg.mainmodel.log_interval == 0:
            print(f'[Train] {self._epsd}epsd {self._step_sum}step ({time.perf_counter() - t0:.0f}s)')
        
    def do_train(self):
        if self._step_sum % self.cfg.mainmodel.train_frq == self.cfg.mainmodel.exp_num_use-1 and self.__step > self.cfg.mainmodel.train_frq:
            self._train_sum = self._train_sum + 1
            return True
        else:
            return False
        
    
    def load_model(self, model, name):
        pass
        
    
    def save_model(self, model, name):
        snapshot_name_prefix = self.outdir + name + self.epsd + self.step
        model.save_state_dict(snapshot_name_prefix)
        print(f'[Snapshot] {snapshot_name_prefix}')

    def run_opt(self, loss):
        for opt in self.cfg.opt.values():
            if not opt._module==None:
                opt._module.zero_grad(set_to_none=True)
                self.scaler.scale(loss).backward()#TODO: retain_graph=Trueは必要か？
                self.scaler.step(opt._module)
                self.scaler.update()

    def _setup_model_params(self):
        for comp in self.cfg.model.comp.values():
            if comp._module:
                comp._module.to(self.cfg.model._device)
                if comp._params_curr:
                    print(f'loading {comp._params_curr}')
                    comp._module.load_state_dict(torch.load(comp._params_curr, map_location=self.cfg.model.device))

    def _setup_optimizer(self, custom_ratio=None):
        for opt in self.cfg.opt.values():
            #今読み込まれているoptを利用するモデルをすべてリストに入れる
            params = []
            for comp in self.cfg.model.comp.values():
                if comp.opt_model == opt.profile_name:
                    for n, p in comp._module.named_parameters():
                        params.append({'params': p, 'opt_params': opt.opt_params})
                        
            opt._module = get_opt_func(opt.name, params) if not len(params)==0 else None

        self.scaler = torch.cuda.amp.GradScaler()

    def _draw_next_train_batch(self):
        try:
            sample = next(self.iter_train)
        except StopIteration:
            self.iter_train = iter(self.loader['train'])
            sample = next(self.iter_train)
        return sample

    def _backprop(self, loss):
        self.cfg.model._optimizer.zero_grad(set_to_none=True)
        self.scaler.scale(loss).backward()

        # Gradient clipping for stabler training.
        if self.cfg.misc.grad_clip_value > 0 or self.cfg.misc.grad_clip_norm > 0:
            self.scaler.unscale_(self.cfg.model._optimizer)
            for comp in self.cfg.model.comp.values():
                if comp._module:
                    if self.cfg.misc.grad_clip_value > 0:
                        torch.nn.utils.clip_grad_value_(comp._module.parameters(), self.cfg.misc.grad_clip_value)
                    if self.cfg.misc.grad_clip_norm > 0:
                        torch.nn.utils.clip_grad_norm_(comp._module.parameters(), self.cfg.misc.grad_clip_norm)

        self.scaler.step(self.cfg.model._optimizer)
        self.scaler.update()

    def _train_mode(func):
        def wrap(self):
            for comp in self.cfg.model.comp.values():
                if comp._module:
                    comp._module.train()
            return func(self)
        return wrap

    def _eval_mode(func):
        # TODO: handle the non-positional argumants
        def wrap(self):
            modules = []
            for comp in self.cfg.model.comp.values():
                if comp._module:
                    modules.append(comp._module)
            with temporal_freeze(modules):
                ret = func(self)
            return ret
        return wrap

    def epsd_finish_training(self):###############
        return self._epsd == self.cfg.mainmodel.max_train_epsd or self.done_epsd

    # def train_epsd(self):
    #     raise Exception()

    # @_eval_mode
    # def eval_epsd(self):
    #     raise Exception()

    def get_lr(self):
        return self.cfg.ls._lr

    def save_state_dict(self, name_prefix):
        for name, comp in self.cfg.model.comp.items():
            if comp._module and not comp.no_snapshot:
                torch.save(comp._module.state_dict(), Path(self.cfg.misc.output_dir).joinpath(f'{name_prefix}_{name}.pth'))

    def get_dataset_info(self):
        result = {}

        for div_name, div in self.cfg.dataset.divisions.items():
            result[div_name] = {}
            for env in div.envs:
                result[div_name][env.name] = env._dataset.get_info()

        return result

    def get_params_path(self):
        ret = None
        for comp in self.cfg.model.comp.values():
            if comp._params_curr:
                ret = comp._params_curr
                break
        return ret

    @staticmethod
    def _set_random_seed(seed):
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
