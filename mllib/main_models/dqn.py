import time
import torch

from ..mlpipeline import MLPipeline
from .data_utils import save_torch_image_batch, rescale_image
from .data_transform import transform_presets
from .dataset_scene_multiviews import SceneMultiViewsCOLMAP, SceneMultiViewsBlender, SceneMultiViewsPhySG
from .model import get_module_class


class DQN(MLPipeline):
    def __init__(self, config, models):
        super().__init__(config)

        # Build the inverse rendering model
        network = self.cfg.model.components.get('DQN_network')
        
        network._module = get_module_class(network.name)(dim_emb_position=positional_encoding._module.dim_feat)

        self._setup_model_params()
        self._setup_optimizer()
        self.scaler = torch.cuda.amp.GradScaler()

        # Build the datasets
        for div_name, div in self.cfg.dataset.divisions.items():
            for env in div.envs:
                if env.dataset_class == 'SceneMultiViewsBlender':
                    if div_name == 'eval':
                        div_name = 'train'
                    env._dataset = SceneMultiViewsBlender(
                        root=env._path, transform=transform_presets[env.transform], division=div_name,##########################
                    )
                elif env.dataset_class == 'SceneMultiViewsPhySG':
                    if div_name == 'eval':
                        div_name = 'train'
                    env._dataset = SceneMultiViewsPhySG(
                        root=env._path, division=div_name, gamma=self.cfg.inverse_rendering.gamma,train_cameras=self.cfg.inverse_rendering.train_cameras,
                    )
                else:   
                    env._dataset = SceneMultiViewsCOLMAP(
                        root=env._path, transform=transform_presets[env.transform],
                    )

        self._setup_dataset_and_loader()

        # Preload all samples
        # TODO: move the sampling code into the dataset class??
        images = []
        rays = []
        for sample in self.loader['train']:
            image, ray, mask = sample['image'], sample['ray'], sample['mask']
            image = image.permute(0, 2, 3, 1)  # TODO: use to torch tensor without channel perm
            image[~mask[:, :, :, 0]] = 0.0  # TODO: Match to the env map color in the scene.

            images.append(image)
            rays.append(ray)

        self.images = torch.cat(images, dim=0)
        self.rays = torch.cat(rays, dim=0)

    def finish_training(self):
        return False

    @MLPipeline._train_mode
    def train_epsd(self):
        result = {}

        # Random sample rays (due to limited amount of GPU memory).
        b, h, w, c = self.images.shape
        image_batch_size = min(b, self.cfg.inverse_rendering.train_image_batch_size)
        if image_batch_size == 0:
            image = self.images.view(-1, 3)
            ray = self.rays.view(-1, 6)
            rand_idx = torch.randint(0, b * w * h, (self.cfg.inverse_rendering.train_ray_batch_size,))
            image = image[rand_idx]
            ray = ray[rand_idx]
        else:
            # Use this when the number of ray samples per image is important.
            rand_image_idx = torch.randint(0, b, (image_batch_size,))
            image = self.images[rand_image_idx].reshape(-1, 3)
            ray = self.rays[rand_image_idx].reshape(-1, 6)
            rand_idx = torch.randint(0, image_batch_size * w * h, (self.cfg.inverse_rendering.train_ray_batch_size,))
            image = image[rand_idx]
            ray = ray[rand_idx]

        image_, ray_ = image.to(self.cfg.model._device), ray.to(self.cfg.model._device)

        # TODO: importance sampling here before rendering

        positional_encoding = self.cfg.model.components['positional_encoding']._module
        angular_encoding = self.cfg.model.components['angular_encoding']._module
        sdf = self.cfg.model.components['sdf']._module
        albedo = self.cfg.model.components['albedo']._module
        renderer = self.cfg.model.components['renderer']._module

        with torch.cuda.amp.autocast():
            rgb_ = renderer(ray_, positional_encoding, angular_encoding, sdf, albedo)

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
        return result

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

                    positional_encoding = self.cfg.model.components['positional_encoding']._module
                    angular_encoding = self.cfg.model.components['angular_encoding']._module
                    sdf = self.cfg.model.components['sdf']._module
                    albedo = self.cfg.model.components['albedo']._module
                    renderer = self.cfg.model.components['renderer']._module

                    with torch.cuda.amp.autocast():
                        for i in range(0, ray_.shape[0], self.cfg.inverse_rendering.test_ray_batch_size):
                            rgb_[i:i+self.cfg.inverse_rendering.test_ray_batch_size, :] = renderer(ray_[i:i+self.cfg.inverse_rendering.test_ray_batch_size, :], positional_encoding, sdf, albedo)

                    rgb = (rgb_.clip(0, 1) * 255).to(torch.uint8).reshape(h, w, c).cpu().numpy()
                    result[env_name] = {
                        'stat': {},
                        'recon_sample': rgb,
                    }
                    break  # Currently exporting just a single test result.

        return result

    def export_augmented_samples(self, minibatch_size=16):
        # TODO: export masked images
        pass
    
    def eval_result_writer(self, env_name, env_result, dpts_curr, writer):
        #TODO: そのステップの動画を切り取って保存する
        pass

    @MLPipeline._eval_mode
    def measure_inference_time(self, minibatch_size=1, trials=100):
        result = {}

        return result