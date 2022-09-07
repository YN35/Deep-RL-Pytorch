import time
import random

import torch

from ...mlpipeline import MLPipeline
from ...experience import Epx


class DQN(MLPipeline):
    def __init__(self, config):
        super().__init__(config)

        self._setup_model_params()
        self._setup_optimizer()
        self.exp = Epx(device=self.cfg.model._device, profile_name='DQN', exp_buffer_size=self.cfg.mainmodel.train_frq+self.cfg.mainmodel.exp_num_use)
        

    def finish_training(self):
        return False

    @MLPipeline._train_mode
    def train_epsd(self):
        """Train the model for one episode."""
        
        obs = self.cfg.env._module.reset()
        network = self.cfg.model.comp.get('q_net')._module

        while True:
            obs = torch.from_numpy(obs).to(self.cfg.model._device)
            with torch.cuda.amp.autocast(enabled=self.cfg.mainmodel.enable_fp16):
                
                Q_val = network(obs)
                if random.random() < self.cfg.mainmodel.random_act_prob:
                    act = random.randint(0, int(self.cfg.mainmodel.act_space - 1))
                    act = torch.tensor(act).to(self.cfg.model._device)
                else:      
                    act = Q_val.argmax()
                
                obs, reward, done, info = self.cfg.env._module.step(act)
                
                #update experience buffer
                self.exp.roll()
                self.exp.rewards[0] = reward
                self.exp.done[0] = done
                self.exp.esti_qmax[0] = Q_val[act]

            if self.do_train():
                with torch.cuda.amp.autocast(enabled=self.cfg.mainmodel.enable_fp16):
                    target_q_val = self.exp.rewards[1:] + self.cfg.mainmodel.discount_rate * self.exp.esti_qmax[:-1].detach() * (1 - self.exp.done[1:].type(torch.uint8))
                    loss = ((self.exp.esti_qmax[1:] - target_q_val) ** 2).mean()
                    
                    self.exp.esti_qmax = torch.empty(self.cfg.mainmodel.train_frq+self.cfg.mainmodel.exp_num_use).to(self.cfg.model._device)
                
                self.run_opt(loss)

                
                
                if self._train_sum % self.cfg.log.bord.train == 0:
                    self.writer.add_scalar('train_reco/loss', loss, self._train_sum)
                    self.writer.add_scalar('train_reco/buffa_reward', self.exp.rewards.sum(), self._train_sum)
                if self._train_sum % self.cfg.log.print.train == 0:
                    print(f'[Train] ({time.perf_counter() - self.t0:.0f}s) {self._train_sum}train {self._epsd}epsd {self._step_sum}step {loss}loss')
            
            if self._step_sum % self.cfg.log.bord.step == 0:
                self.writer.add_scalar('step_reco/time', time.perf_counter() - self.t0, self._step_sum)
            if self._step_sum % self.cfg.log.print.step == 0:
                print(f'[Step {self._step_sum/self.cfg.mainmodel.max_train_step}] --{self._step_sum}/{self.cfg.mainmodel.max_train_step}-- ({time.perf_counter() - self.t0:.0f}s)')
                
            
            self.add_step()
            if done or self.cfg.mainmodel.max_train_step < self._step_sum:
                break
        
        if self._epsd % self.cfg.log.bord.epsd == 0:
            self.writer.add_scalar('step_reco/epsd_reward', self.cfg.env._module.epsd_reward, self._step_sum)
        if self._epsd % self.cfg.log.print.epsd == 0:
            print(f'[Epsd {self._step_sum/self.cfg.mainmodel.max_train_epsd}] --{self._epsd}/{self.cfg.mainmodel.max_train_epsd}-- {self.cfg.env._module.epsd_reward}reward {self.cfg.env._module.epsd_step}step(fin) {self._step_sum}step')

    @MLPipeline._eval_mode
    def eval_epsd(self):
        result = {}

        if self.loader.get('eval'):
            for env_name, loader in self.loader['eval'].items():
                for n, sample in enumerate(loader):
                    image, ray = sample['image'][0], sample['ray'][0]
                    image = image.permute(1, 2, 0)
                    h, w, c = image.shape

                    image = image.reshape(-1, 3)
                    ray = ray.reshape(-1, 6)

                    image_, ray_ = image.to(self.cfg.model._device), ray.to(self.cfg.model._device)
                    rgb_ = torch.empty_like(image_)

                    positional_encoding = self.cfg.model.comp['positional_encoding']._module
                    angular_encoding = self.cfg.model.comp['angular_encoding']._module
                    sdf = self.cfg.model.comp['sdf']._module
                    albedo = self.cfg.model.comp['albedo']._module
                    renderer = self.cfg.model.comp['renderer']._module

                    with torch.cuda.amp.autocast():
                        for i in range(0, ray_.shape[0], self.cfg.inverse_rendering.test_ray_batch_size):
                            rgb_[i:i+self.cfg.inverse_rendering.test_ray_batch_size, :] = renderer(ray_[i:i+self.cfg.inverse_rendering.test_ray_batch_size, :], positional_encoding, sdf, albedo)

                    rgb = (rgb_.clip(0, 1) * 255).to(torch.uint8).reshape(h, w, c).cpu().numpy()
                    result[env_name] = {
                        'stat': {},
                        'recon_sample': rgb,
                    }
                    break  # Currently exporting just a single test result.
                
                #TODO: save result data

        return result

    def export_augmented_samples(self, minibatch_size=16):
        # TODO: export masked images
        pass
    

    @MLPipeline._eval_mode
    def measure_inference_time(self, minibatch_size=1, trials=100):
        result = {}

        return result