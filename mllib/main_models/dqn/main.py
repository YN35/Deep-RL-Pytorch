import torch

from ...mlpipeline import MLPipeline


class DQN(MLPipeline):
    def __init__(self, config):
        super().__init__(config)

        self._setup_model_params()
        self._setup_optimizer()
        

    def finish_training(self):
        return False

    @MLPipeline._train_mode
    def train_epsd(self):
        
        obs = self.cfg.env._module.reset()
        network = self.cfg.model.comp.get('q_net')._module
        result = {}

        while True:
            with torch.cuda.amp.autocast(enabled=self.cfg.mainmodel.enable_fp16):
                
                Q_val = network(obs)
                act = Q_val.argmax()
                
                obs, reward, done, info = self.cfg.env._module.step(act)
                if done:
                    break
                

                loss_recon_ = self.cfg.model._loss['recon'](rgb_, image_).mean()
                loss_ = loss_recon_

        self.cfg.model._optimizer.zero_grad(set_to_none=True)
        self.scaler.scale(loss_).backward()
        self.scaler.step(self.cfg.model._optimizer)
        self.scaler.update()

        result['dpts'] = 1
        result['loss'] = loss_.item()
        result['loss_terms'] = {
            'mse': loss_recon_.item(),
            'psnr': -10*torch.log10(loss_recon_).item(),
        }
        self.add_step_log()
            
        return epsd_log

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