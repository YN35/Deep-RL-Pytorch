from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, ConcatDataset

from .model import get_module_class
from .torch_utils import temporal_freeze


class MLPipeline:
    def __init__(self, cfg):
        self.cfg = cfg
        
        self._set_random_seed(self.cfg.misc.seed)
        self.cfg.model._device = torch.device(self.cfg.model.device) if self.cfg.model.device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('MLPipeline init - set device as' + self.cfg.model._device)

        for comp in self.cfg.model.components.values():
            if len(comp.params) > 0:
                path = comp.params.pop()
                comp._params_curr = Path(self.cfg.model.root).joinpath(path)
            else:
                comp._params_curr = None

        for division in self.cfg.dataset.divisions.values():
            for env in division.envs:
                env._path = Path(self.cfg.dataset.root).joinpath(env.path)

    def _setup_model_params(self):
        for comp in self.cfg.model.components.values():
            if comp._module:
                comp._module.to(self.cfg.model._device)
                if comp._params_curr:
                    print(f'loading {comp._params_curr}')
                    comp._module.load_state_dict(torch.load(comp._params_curr, map_location=self.cfg.model.device))

    def _setup_optimizer(self, custom_ratio=None):
        div_train = self.cfg.dataset.divisions.get('train')
        if div_train and div_train.envs:
            self.cfg.ls._initial_lr = self.cfg.ls.base_initial_lr * (custom_ratio or (div_train.minibatch_size / 256))
            self.cfg.ls._lr = self.cfg.ls._initial_lr
            self.cfg.ls._gain_stat = []
            self.cfg.ls._is_warmup_phase = True
            self.cfg.ls._suspend_cooldown = True

            params = []
            for comp in self.cfg.model.components.values():
                if comp._module:
                    no_wd_module_classes = (
                        get_module_class('MultiresolutionHashEncoding'),
                        get_module_class('LipschitzNorm'),
                    )
                    no_wd = {n for n, m in comp._module.named_modules() if isinstance(m, no_wd_module_classes)}
                    for n, p in comp._module.named_parameters():
                        wd = 0.0 if any(n.startswith(bn) for bn in no_wd) else self.cfg.misc.wd
                        params.append({'params': p, 'lr': self.cfg.ls._lr, 'weight_decay': wd, 'betas': self.cfg.misc.adam_betas, 'eps': self.cfg.misc.adam_eps})

            self.cfg.model._optimizer = torch.optim.AdamW(params)
            self.scaler = torch.cuda.amp.GradScaler()

            # For lipschitz normalization (https://nv-tlabs.github.io/lip-mlp/)
            self.lipschitz_norms = []  # TODO: Can use the same objective if there are more than one mlp??
            for comp in self.cfg.model.components.values():
                if comp._module:
                    self.lipschitz_norms.extend([m for m in comp._module.modules() if isinstance(m, get_module_class('LipschitzNorm'))])

    def _setup_dataset_and_loader(self):
        self.loader = {}

        div_train = self.cfg.dataset.divisions.get('train')
        if div_train and div_train.envs:
            dataset = ConcatDataset([env._dataset for env in div_train.envs])
            self.loader['train'] = DataLoader(dataset, batch_size=div_train.minibatch_size, shuffle=div_train.shuffle, drop_last=div_train.drop_last, num_workers=div_train.num_workers, persistent_workers=True, pin_memory=True)
            self.iter_train = iter(self.loader['train'])

        for div_name, div in self.cfg.dataset.divisions.items():
            if div_name == 'train':
                continue
        
            self.loader[div_name] = {
                env.name: DataLoader(env._dataset, batch_size=div.minibatch_size, shuffle=False, drop_last=False, num_workers=div.num_workers, persistent_workers=False, pin_memory=True)
                for env in div.envs
            }

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
            for comp in self.cfg.model.components.values():
                if comp._module:
                    if self.cfg.misc.grad_clip_value > 0:
                        torch.nn.utils.clip_grad_value_(comp._module.parameters(), self.cfg.misc.grad_clip_value)
                    if self.cfg.misc.grad_clip_norm > 0:
                        torch.nn.utils.clip_grad_norm_(comp._module.parameters(), self.cfg.misc.grad_clip_norm)

        self.scaler.step(self.cfg.model._optimizer)
        self.scaler.update()

    def _eval_mode(func):
        # TODO: handle the non-positional argumants
        def wrap(self):
            modules = []
            for comp in self.cfg.model.components.values():
                if comp._module:
                    modules.append(comp._module)
            with temporal_freeze(modules):
                ret = func(self)
            return ret
        return wrap

    def finish_training(self):
        return self.cfg.ls._lr < self.cfg.ls._initial_lr

    # def train_epsd(self):
    #     raise Exception()

    # @_eval_mode
    # def eval_epsd(self):
    #     raise Exception()

    def update_lr(self, gain):
        self.cfg.ls._gain_stat.append(gain)
        window_size = self.cfg.ls.gain_window_size_warmup_phase if self.cfg.ls._is_warmup_phase else self.cfg.ls.gain_window_size_cooldown_phase
        self.cfg.ls._gain_stat = self.cfg.ls._gain_stat[-window_size:]
        gain = np.mean(self.cfg.ls._gain_stat)

        if self.cfg.ls._is_warmup_phase:
            if gain < self.cfg.ls.gain_threshold_turn_phase:
                self.cfg.ls._is_warmup_phase = not self.cfg.ls._is_warmup_phase
                self.cfg.ls._gain_stat = []
            elif gain < self.cfg.ls.gain_threshold_warmup_phase:
                self.cfg.ls._lr *= self.cfg.ls.lr_warmup_ratio
                for group in self.cfg.model._optimizer.param_groups:
                    group['lr'] *= self.cfg.ls.lr_warmup_ratio
        else:
            if self.cfg.ls._suspend_cooldown:
                if gain < self.cfg.ls.suspend_cooldown_threshold_ratio * self.cfg.ls.gain_threshold_cooldown_phase:
                    self.cfg.ls._suspend_cooldown = not self.cfg.ls._suspend_cooldown
            else:
                if gain < self.cfg.ls.gain_threshold_cooldown_phase:
                    self.cfg.ls._lr *= self.cfg.ls.lr_cooldown_ratio
                    for group in self.cfg.model._optimizer.param_groups:
                        group['lr'] *= self.cfg.ls.lr_cooldown_ratio
                else:
                    self.cfg.ls._suspend_cooldown = not self.cfg.ls._suspend_cooldown

        return gain

    def get_lr(self):
        return self.cfg.ls._lr

    def save_state_dict(self, name_prefix):
        for name, comp in self.cfg.model.components.items():
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
        for comp in self.cfg.model.components.values():
            if comp._params_curr:
                ret = comp._params_curr
                break
        return ret

    @staticmethod
    def _set_random_seed(seed):
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
