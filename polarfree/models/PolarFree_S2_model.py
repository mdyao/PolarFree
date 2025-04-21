import torch
import numpy as np
from collections import OrderedDict
from os import path as osp
from tqdm import tqdm
from functools import partial
import time
import cv2

from torch.nn import functional as F
from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.metrics import calculate_metric
from basicsr.utils import get_root_logger, imwrite, tensor2img
from basicsr.utils.registry import MODEL_REGISTRY

from polarfree.utils.losses import TVLoss, VGGLoss, PhaseLoss
from polarfree.utils.base_model import BaseModel
from polarfree.utils.beta_schedule import make_beta_schedule, default
from ldm.ddpm import DDPM

@MODEL_REGISTRY.register()
class PolarFree_S2(BaseModel):
    """PolarFree Stage 2 model for polarization image enhancement."""

    def __init__(self, opt):
        super(PolarFree_S2, self).__init__(opt)
        self._init_networks(opt)
        self._load_pretrained_models()
        
        # Setup diffusion model
        self.apply_ldm = self.opt['diffusion_schedule'].get('apply_ldm', None)
        self._setup_diffusion()
        
        if self.is_train:
            self.init_training_settings()

    def _init_networks(self, opt):
        """Initialize networks and move to specified device."""
        # Latent encoder
        self.net_le = build_network(opt['network_le'])
        self.net_le = self.model_to_device(self.net_le)
        self.print_network(self.net_le)

        # Latent encoder for diffusion model
        self.net_le_dm = build_network(opt['network_le_dm'])
        self.net_le_dm = self.model_to_device(self.net_le_dm)
        self.print_network(self.net_le_dm)

        # Denoiser network
        self.net_d = build_network(opt['network_d'])
        self.net_d = self.model_to_device(self.net_d)
        self.print_network(self.net_d)

        # Generator network
        self.net_g = build_network(opt['network_g'])
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)
    
    def _load_pretrained_models(self):
        """Load pretrained model weights."""
        # Load latent encoder
        load_path = self.opt['path'].get('pretrain_network_le', None)
        if load_path is not None:
            self._load_network_with_path(self.net_le, load_path, 'le')
        
        # Load latent encoder for diffusion model
        load_path = self.opt['path'].get('pretrain_network_le_dm', None)
        if load_path is not None:
            self._load_network_with_path(self.net_le_dm, load_path, 'le_dm')
        
        # Load denoiser network
        load_path = self.opt['path'].get('pretrain_network_d', None)
        if load_path is not None:
            self._load_network_with_path(self.net_d, load_path, 'd')
        
        # Load generator network
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            self._load_network_with_path(self.net_g, load_path, 'g')
    
    def _load_network_with_path(self, network, load_path, network_label):
        """Helper to load a specific network."""
        param_key = self.opt['path'].get(f'param_key_{network_label}', 'params')
        strict_load = self.opt['path'].get(f'strict_load_{network_label}', True)
        self.load_network(network, load_path, strict_load, param_key)

    def _setup_diffusion(self):
        """Setup diffusion model."""
        if self.apply_ldm:
            # Use LDM implementation
            self.diffusion = DDPM(
                denoise=self.net_d,
                condition=self.net_le_dm,
                n_feats=self.opt['network_g']['embed_dim'],
                group=self.opt['network_g']['group'],
                linear_start=self.opt['diffusion_schedule']['linear_start'],
                linear_end=self.opt['diffusion_schedule']['linear_end'],
                timesteps=self.opt['diffusion_schedule']['timesteps']
            )
            self.diffusion = self.model_to_device(self.diffusion)
        else:
            # Use local implementation
            self.set_new_noise_schedule(self.opt['diffusion_schedule'], self.device)

    def init_training_settings(self):
        """Initialize training settings including losses and optimizers."""
        # Set networks to train mode
        self.net_g.train()
        self.net_d.train()
        self.net_le.train()
        self.net_le_dm.train()
        if self.apply_ldm:
            self.diffusion.train()
        
        train_opt = self.opt['train']
        self.ema_decay = train_opt.get('ema_decay', 0)
        
        # Initialize loss functions
        self._setup_loss_functions(train_opt)
        
        # Set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def _setup_loss_functions(self, train_opt):
        """Setup various loss functions based on configuration."""
        # Pixel loss
        if train_opt.get('pixel_opt'):
            self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device)
            self.cri_pix_diff = build_loss(train_opt['pixel_diff_opt']).to(self.device)
        else:
            self.cri_pix = None
            self.cri_pix_diff = None
        
        # Perceptual loss
        self.cri_perceptual = build_loss(train_opt['perceptual_opt']).to(self.device) if train_opt.get('perceptual_opt') else None
        
        # TV loss
        self.cri_tv = TVLoss(**train_opt['tv_opt']).to(self.device) if train_opt.get('tv_opt') else None
        
        # VGG loss
        self.cri_vgg = VGGLoss(**train_opt['vgg_opt']).to(self.device) if train_opt.get('vgg_opt') else None
        
        # Phase loss
        self.cri_phase = PhaseLoss(**train_opt['phase_opt']).to(self.device) if train_opt.get('phase_opt') else None
        
        # Ensure at least one primary loss is defined
        if self.cri_pix is None and self.cri_perceptual is None:
            raise ValueError('Both pixel and perceptual losses are None.')

    def setup_optimizers(self):
        """Set up optimizers for networks."""
        train_opt = self.opt['train']
        optim_params = []
        
        # Collect generator parameters
        self._collect_trainable_params(self.net_g, optim_params, 'G')
        
        # Collect diffusion model parameters
        if self.apply_ldm:
            self._collect_trainable_params(self.diffusion, optim_params, 'Diffusion')
        else:
            self._collect_trainable_params(self.net_le_dm, optim_params, 'LE-DM')
            self._collect_trainable_params(self.net_d, optim_params, 'D')
        
        # Create optimizer
        optim_type = train_opt['optim_total'].pop('type')
        if optim_type == 'Adam':
            self.optimizer_total = torch.optim.Adam(optim_params, **train_opt['optim_total'])
        elif optim_type == 'AdamW':
            self.optimizer_total = torch.optim.AdamW(optim_params, **train_opt['optim_total'])
        else:
            raise NotImplementedError(f'Optimizer {optim_type} is not supported yet.')
        
        self.optimizers.append(self.optimizer_total)

    def _collect_trainable_params(self, network, param_list, network_name):
        """Collect trainable parameters from a network."""
        logger = get_root_logger()
        for k, v in network.named_parameters():
            if v.requires_grad:
                param_list.append(v)
            else:
                logger.warning(f'Network {network_name}: Params {k} will not be optimized.')

    def set_new_noise_schedule(self, schedule_opt, device):
        """Set up noise schedule for diffusion model."""
        to_torch = partial(torch.tensor, dtype=torch.float32, device=device)
        
        # Create beta schedule
        betas = make_beta_schedule(
            schedule=schedule_opt['schedule'],
            n_timestep=schedule_opt['timesteps'],
            linear_start=schedule_opt['linear_start'],
            linear_end=schedule_opt['linear_end'])
        
        betas = betas.detach().cpu().numpy() if isinstance(betas, torch.Tensor) else betas
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])
        self.sqrt_alphas_cumprod_prev = np.sqrt(np.append(1., alphas_cumprod))
        
        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        
        # Register buffers for diffusion process
        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod - 1)))
        
        # Posterior calculations
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('posterior_variance', to_torch(posterior_variance))
        self.register_buffer(
            'posterior_log_variance_clipped', 
            to_torch(np.log(np.maximum(posterior_variance, 1e-20)))
        )
        self.register_buffer(
            'posterior_mean_coef1', 
            to_torch(betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        )
        self.register_buffer(
            'posterior_mean_coef2', 
            to_torch((1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod))
        )

    # Diffusion model functions
    def predict_start_from_noise(self, x_t, t, noise):
        """Predict x0 from noise."""
        return self.sqrt_recip_alphas_cumprod[t] * x_t - self.sqrt_recipm1_alphas_cumprod[t] * noise

    def q_posterior(self, x_start, x_t, t):
        """Compute posterior q(x_{t-1} | x_t, x_0)."""
        posterior_mean = self.posterior_mean_coef1[t] * x_start + self.posterior_mean_coef2[t] * x_t
        posterior_log_variance_clipped = self.posterior_log_variance_clipped[t]
        return posterior_mean, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, clip_denoised=True, condition_x=None, ema_model=False):
        """Compute mean and variance of p(x_{t-1} | x_t)."""
        if condition_x is None:
            raise RuntimeError('Must have LQ/LR condition')
        
        t_tensor = torch.full(x.shape, t+1, device=self.betas.device, dtype=torch.long)
        noise = self.net_d(x, condition_x, t_tensor)
        x_recon = self.predict_start_from_noise(x, t=t, noise=noise)
        
        if clip_denoised:
            x_recon.clamp_(-1., 1.)
        
        model_mean, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_log_variance
    
    def p_sample_wo_variance(self, x, t, clip_denoised=True, condition_x=None, ema_model=False):
        """Sample from p(x_{t-1} | x_t) without noise."""
        model_mean, _ = self.p_mean_variance(
            x=x, t=t, clip_denoised=clip_denoised, condition_x=condition_x, ema_model=ema_model)
        return model_mean
    
    def p_sample_loop_wo_variance(self, x_in, x_noisy, ema_model=False):
        """Run full reverse process without adding noise."""
        img = x_noisy
        for i in reversed(range(0, self.num_timesteps)):
            img = self.p_sample_wo_variance(img, i, condition_x=x_in, ema_model=ema_model)
        return img

    def p_sample(self, x, t, clip_denoised=True, condition_x=None, ema_model=False):
        """Sample from p(x_{t-1} | x_t)."""
        model_mean, _ = self.p_mean_variance(
            x=x, t=t, clip_denoised=clip_denoised, condition_x=condition_x, ema_model=ema_model)
        return model_mean
    
    def p_sample_loop(self, x_in, x_noisy, ema_model=False):
        """Run full reverse process."""
        img = x_noisy
        for i in reversed(range(0, self.num_timesteps)):
            img = self.p_sample(img, i, condition_x=x_in, ema_model=ema_model)
        return img

    def q_sample(self, x_start, sqrt_alpha_cumprod, noise=None):
        """Forward diffusion sample."""
        noise = default(noise, lambda: torch.randn_like(x_start))
        return (
            sqrt_alpha_cumprod * x_start +
            (1 - sqrt_alpha_cumprod**2).sqrt() * noise
        )

    def feed_data(self, data):
        """Load input and target data."""
        # Load polarization images
        self.lq_img0 = data['lq_img0'].to(self.device)
        self.lq_img45 = data['lq_img45'].to(self.device)
        self.lq_img90 = data['lq_img90'].to(self.device)
        self.lq_img135 = data['lq_img135'].to(self.device)
        self.lq_rgb = data['lq_rgb'].to(self.device)
        self.lq_aolp = data['lq_aolp'].to(self.device)
        self.lq_dolp = data['lq_dolp'].to(self.device)
        self.lq_Ip = data['lq_Ip'].to(self.device)
        self.lq_Inp = data['lq_Inp'].to(self.device)
        
        # Load ground truth
        
        if 'gt_rgb' in data:
            self.gt_rgb = data['gt_rgb'].to(self.device)

    def optimize_parameters(self, current_iter, noise=None):
        """Optimize model parameters for one iteration."""
        # Freeze latent encoder
        for p in self.net_le.parameters():
            p.requires_grad = False
        
        self.optimizer_total.zero_grad()
        
        # Input features for latent encoders
        input_features = [self.lq_rgb, self.lq_img0, self.lq_img45, self.lq_img90, 
                          self.lq_img135, self.lq_aolp, self.lq_dolp]
        
        # Get prior from stage 1 latent encoder
        prior_z = self.net_le(input_features, self.gt_rgb)
        
        # Process through diffusion model
        if self.apply_ldm:
            # Use LDM implementation
            prior, _ = self.diffusion(input_features, prior_z)
        else:
            # Use local implementation
            prior_d = self.net_le_dm(input_features)
            
            # Diffusion forward process
            t = self.opt['diffusion_schedule']['timesteps']
            noise = default(noise, lambda: torch.randn_like(prior_z))
            prior_noisy = self.q_sample(
                x_start=prior_z,
                sqrt_alpha_cumprod=self.alphas_cumprod[t-1],
                noise=noise
            )
            
            # Diffusion reverse process
            prior = self.p_sample_loop_wo_variance(prior_d, prior_noisy)
        
        # Generate output image
        self.output = self.net_g(self.lq_rgb, prior)
        
        # Calculate losses
        l_total = 0
        loss_dict = OrderedDict()
        
        # Pixel loss
        if self.cri_pix:
            l_pix = self.cri_pix(self.output, self.gt_rgb)
            l_total += l_pix
            loss_dict['l_pix'] = l_pix
        
        # Pixel diffusion loss
        if self.cri_pix_diff:
            l_pix_diff = self.cri_pix_diff(prior_z, prior)
            l_total += l_pix_diff
            loss_dict['l_pix_diff'] = l_pix_diff
        
        # Perceptual loss
        if self.cri_perceptual:
            l_percep, l_style = self.cri_perceptual(self.output, self.gt_rgb)
            if l_percep is not None:
                l_total += l_percep
                loss_dict['l_percep'] = l_percep
            if l_style is not None:
                l_total += l_style
                loss_dict['l_style'] = l_style
        
        # TV loss
        if self.cri_tv:
            l_tv = self.cri_tv(self.output)
            if l_tv is not None:
                l_total += l_tv
                loss_dict['l_tv'] = l_tv
        
        # VGG loss
        if self.cri_vgg:
            l_vgg = self.cri_vgg(self.output, self.gt_rgb)
            if l_vgg is not None:
                l_total += l_vgg
                loss_dict['l_vgg'] = l_vgg
        
        # Phase loss
        if self.cri_phase:
            l_phase = self.cri_phase(self.output, self.gt_rgb)
            if l_phase is not None:
                l_total += l_phase
                loss_dict['l_phase'] = l_phase
        
        # Backward pass
        l_total.backward()
        
        # Gradient clipping
        if self.opt['train']['use_grad_clip']:
            if self.apply_ldm:
                torch.nn.utils.clip_grad_norm_(
                    list(self.net_g.parameters()) + list(self.diffusion.parameters()), 0.01)
            else:
                torch.nn.utils.clip_grad_norm_(
                    list(self.net_g.parameters()) + list(self.net_le_dm.parameters()) + list(self.net_d.parameters()), 0.01)
        
        self.optimizer_total.step()
        self.log_dict = self.reduce_loss_dict(loss_dict)

    def test(self):
        """Run inference on the model."""
        self.lq = self.lq_rgb
        
        if hasattr(self, 'gt_rgb'):
            self.gt = self.gt_rgb
        
        # Handle padding for window-based models
        scale = self.opt.get('scale', 1)
        window_size = 8
        
        # Calculate padding to make dimensions divisible by window_size
        _, _, h, w = self.lq.size()
        mod_pad_h = 0 if h % window_size == 0 else window_size - h % window_size
        mod_pad_w = 0 if w % window_size == 0 else window_size - w % window_size
        
        # Pad all input images
        img = F.pad(self.lq, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        img0 = F.pad(self.lq_img0, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        img45 = F.pad(self.lq_img45, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        img90 = F.pad(self.lq_img90, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        img135 = F.pad(self.lq_img135, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        img_aolp = F.pad(self.lq_aolp, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        img_dolp = F.pad(self.lq_dolp, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        
        if hasattr(self, 'gt'):
            gt = F.pad(self.gt, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        
        input_features = [img, img0, img45, img90, img135, img_aolp, img_dolp]
        
        # Run inference
        if hasattr(self, 'net_g_ema'):
            get_root_logger().warning("EMA network exists but is not fully implemented")
        else:
            if self.apply_ldm:
                # LDM implementation
                self._test_with_ldm(input_features, img)
            else:
                # Local implementation
                self._test_without_ldm(input_features, img)
        
        # Remove padding
        _, _, h, w = self.output.size()
        self.output = self.output[:, :, 0:h - mod_pad_h * scale, 0:w - mod_pad_w * scale]

    def _test_with_ldm(self, input_features, img):
        """Test with LDM implementation."""
        self.net_g.eval()
        self.diffusion.eval()
        
        with torch.no_grad():
            prior = self.diffusion(input_features)
            self.output = self.net_g(img, prior)
        
        self.net_g.train()
        self.diffusion.train()

    def _test_without_ldm(self, input_features, img):
        """Test with local implementation."""
        self.net_le.eval()
        self.net_le_dm.eval()
        self.net_d.eval()
        self.net_g.eval()
        
        with torch.no_grad():
            prior_c = self.net_le_dm(input_features)
            prior_noisy = torch.randn_like(prior_c)
            prior = self.p_sample_loop(prior_c, prior_noisy)
            self.output = self.net_g(img, prior)
        
        self.net_le.train()
        self.net_le_dm.train()
        self.net_d.train()
        self.net_g.train()

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        """Distributed validation."""
        if self.opt['rank'] == 0:
            self.nondist_validation(dataloader, current_iter, tb_logger, save_img)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        """Run validation and calculate metrics."""
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        use_pbar = self.opt['val'].get('pbar', False)
        
        # Initialize metrics
        if with_metrics:
            if not hasattr(self, 'metric_results'):
                self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
            self._initialize_best_metric_results(dataset_name)
            self.metric_results = {metric: 0 for metric in self.metric_results}
        
        metric_data = dict()
        if use_pbar:
            pbar = tqdm(total=len(dataloader), unit='image')
        
        # Process each validation image
        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            
            # Run inference
            self.feed_data(val_data)
            self.test()
            
            # Collect results
            visuals = self.get_current_visuals()
            sr_img = tensor2img([visuals['result']])
            lq_img = tensor2img([visuals['lq']])
            metric_data['img'] = sr_img
            
            if 'gt' in visuals:
                gt_img = tensor2img([visuals['gt']])
                metric_data['img2'] = gt_img
                del self.gt
            
            # Release memory
            del self.lq
            del self.output
            torch.cuda.empty_cache()
            
            # Save output image
            if save_img:
                self._save_validation_images(img_name, sr_img, current_iter, dataset_name)
            
            # Calculate metrics
            if with_metrics:
                for name, opt_ in self.opt['val']['metrics'].items():
                    self.metric_results[name] += calculate_metric(metric_data, opt_)
            
            # Update progress bar
            if use_pbar:
                pbar.update(1)
                pbar.set_description(f'Test {img_name}')
        
        if use_pbar:
            pbar.close()
        
        # Process metrics
        if with_metrics:
            self._process_metrics(idx + 1, dataset_name, current_iter, tb_logger)

    def _save_validation_images(self, img_name, sr_img, current_iter, dataset_name):
        """Save validation images."""
        if self.opt['is_train']:
            save_img_path = osp.join(self.opt['path']['visualization'], f'{img_name}.png')
        else:
            if self.opt['val']['suffix']:
                save_img_path = osp.join(
                    self.opt['path']['visualization'],
                    dataset_name,
                    f"{img_name}_{self.opt['val']['suffix']}.png"
                )
            else:
                save_img_path = osp.join(
                    self.opt['path']['visualization'],
                    dataset_name,
                    f'{img_name}.png'
                )
        
        imwrite(sr_img, save_img_path)

    def _process_metrics(self, num_images, dataset_name, current_iter, tb_logger):
        """Process and log metrics."""
        for metric in self.metric_results.keys():
            self.metric_results[metric] /= num_images
            self._update_best_metric_result(dataset_name, metric, self.metric_results[metric], current_iter)
        
        self._log_validation_metric_values(current_iter, dataset_name, tb_logger)

    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        """Log validation metric values."""
        log_str = f'Validation {dataset_name}\n'
        
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}'
            if hasattr(self, 'best_metric_results'):
                log_str += (f'\tBest: {self.best_metric_results[dataset_name][metric]["val"]:.4f} @ '
                            f'{self.best_metric_results[dataset_name][metric]["iter"]} iter')
            log_str += '\n'
        
        logger = get_root_logger()
        logger.info(log_str)
        
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{dataset_name}/{metric}', value, current_iter)

    def get_current_visuals(self):
        """Get current visual results."""
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        return out_dict

    def save(self, epoch, current_iter):
        """Save networks and training state."""
        if hasattr(self, 'net_g_ema'):
            get_root_logger().warning("EMA network saving not implemented")
        else:
            # Extract networks from diffusion model if using LDM
            if self.apply_ldm:
                if self.opt['dist']:
                    self.net_le_dm = self.diffusion.module.condition
                    self.net_d = self.diffusion.module.model
                else:
                    self.net_le_dm = self.diffusion.condition
                    self.net_d = self.diffusion.model
            
            # Save networks
            self.save_network(self.net_g, 'net_g', current_iter)
            self.save_network(self.net_le_dm, 'net_le_dm', current_iter)
            self.save_network(self.net_d, 'net_d', current_iter)
        
        self.save_training_state(epoch, current_iter)
