import torch
import numpy as np
from collections import OrderedDict
from os import path as osp
from tqdm import tqdm
from functools import partial
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
class PolarFree_S1(BaseModel):
    """PolarFree Stage 1 model for polarization image enhancement.
    
    This model combines a latent encoder and a generator network to process 
    polarization images and enhance their quality.
    """
    
    def __init__(self, opt):
        super(PolarFree_S1, self).__init__(opt)
        self._init_networks(opt)
        
        # Load pretrained models if specified
        self._load_pretrained_models()
        
        if self.is_train:
            self.init_training_settings()

    def _init_networks(self, opt):
        """Initialize the latent encoder and generator networks"""
        # Define latent encoder network
        self.net_le = build_network(opt['network_le'])
        self.net_le = self.model_to_device(self.net_le)
        self.print_network(self.net_le)
        
        # Define generator network
        self.net_g = build_network(opt['network_g'])
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)

    def _load_pretrained_models(self):
        """Load pretrained weights for the networks if specified"""
        # Load latent encoder pretrained weights
        load_path = self.opt['path'].get('pretrain_network_le', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_le', 'params')
            self.load_network(
                self.net_le, 
                load_path, 
                self.opt['path'].get('strict_load_le', True), 
                param_key
            )
            
        # Load generator pretrained weights
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_g', 'params')
            self.load_network(
                self.net_g, 
                load_path, 
                self.opt['path'].get('strict_load_g', True), 
                param_key
            )

    def init_training_settings(self):
        """Initialize training settings including losses and optimizers"""
        self.net_le.train()
        self.net_g.train()
        
        train_opt = self.opt['train']
        
        # EMA decay (not implemented yet)
        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.warning("EMA decay is set but not implemented yet")
        
        # Define losses
        self._setup_loss_functions(train_opt)
        
        # Set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def _setup_loss_functions(self, train_opt):
        """Setup various loss functions based on configuration"""
        # Pixel loss
        self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device) if train_opt.get('pixel_opt') else None
        
        # Perceptual loss
        self.cri_perceptual = build_loss(train_opt['perceptual_opt']).to(self.device) if train_opt.get('perceptual_opt') else None
        
        # TV loss
        self.cri_tv = TVLoss(**train_opt['tv_opt']).to(self.device) if train_opt.get('tv_opt') else None
        
        # VGG loss
        self.cri_vgg = VGGLoss(**train_opt['vgg_opt']).to(self.device) if train_opt.get('vgg_opt') else None
        
        # Phase loss
        self.cri_phase = PhaseLoss(**train_opt['phase_opt']).to(self.device) if train_opt.get('phase_opt') else None
        
        # Ensure at least one loss is defined
        if self.cri_pix is None and self.cri_perceptual is None:
            raise ValueError('Both pixel and perceptual losses are None.')

    def setup_optimizers(self):
        """Set up optimizers for both networks"""
        train_opt = self.opt['train']
        
        # Collect parameters to optimize
        optim_params = []
        
        # Generator parameters
        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Network G: Params {k} will not be optimized.')
        
        # Latent encoder parameters
        for k, v in self.net_le.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Network LE: Params {k} will not be optimized.')
        
        # Create optimizer
        optim_type = train_opt['optim_total'].pop('type')
        if optim_type == 'Adam':
            self.optimizer_total = torch.optim.Adam(optim_params, **train_opt['optim_total'])
        elif optim_type == 'AdamW':
            self.optimizer_total = torch.optim.AdamW(optim_params, **train_opt['optim_total'])
        else:
            raise NotImplementedError(f'Optimizer {optim_type} is not supported yet.')
        
        self.optimizers.append(self.optimizer_total)

    def feed_data(self, data):
        """Feed data to the model"""
        # Load all polarization images and data
        self.lq_img0 = data['lq_img0'].to(self.device)
        self.lq_img45 = data['lq_img45'].to(self.device)
        self.lq_img90 = data['lq_img90'].to(self.device)
        self.lq_img135 = data['lq_img135'].to(self.device)
        self.lq_rgb = data['lq_rgb'].to(self.device)
        self.lq_aolp = data['lq_aolp'].to(self.device)
        self.lq_dolp = data['lq_dolp'].to(self.device)
        self.lq_Ip = data['lq_Ip'].to(self.device)
        self.lq_Inp = data['lq_Inp'].to(self.device)
        self.gt_rgb = data['gt_rgb'].to(self.device)
        
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)

    def optimize_parameters(self, current_iter):
        """Optimize model parameters for one iteration"""
        # Clear gradients
        self.optimizer_total.zero_grad()
        
        # Forward pass
        input_features = [
            self.lq_rgb, self.lq_img0, self.lq_img45, 
            self.lq_img90, self.lq_img135, self.lq_aolp, self.lq_dolp
        ]
        prior = self.net_le(input_features, self.gt_rgb)  # latent encoding
        self.output = self.net_g(self.lq_rgb, prior)  # generate enhanced output
        
        # Calculate losses
        l_total = 0
        loss_dict = OrderedDict()
        
        # Apply different loss functions
        if self.cri_pix:
            l_pix = self.cri_pix(self.output, self.gt_rgb)
            l_total += l_pix
            loss_dict['l_pix'] = l_pix
            
        if self.cri_perceptual:
            l_percep, l_style = self.cri_perceptual(self.output, self.gt_rgb)
            if l_percep is not None:
                l_total += l_percep
                loss_dict['l_percep'] = l_percep
            if l_style is not None:
                l_total += l_style
                loss_dict['l_style'] = l_style
                
        if self.cri_tv:
            l_tv = self.cri_tv(self.output)
            if l_tv is not None:
                l_total += l_tv
                loss_dict['l_tv'] = l_tv
                
        if self.cri_vgg:
            l_vgg = self.cri_vgg(self.output, self.gt_rgb)
            if l_vgg is not None:
                l_total += l_vgg
                loss_dict['l_vgg'] = l_vgg
                
        if self.cri_phase:
            l_phase = self.cri_phase(self.output, self.gt_rgb)
            if l_phase is not None:
                l_total += l_phase
                loss_dict['l_phase'] = l_phase
        
        # Backpropagation
        l_total.backward()
        
        # Gradient clipping if enabled
        if self.opt['train'].get('use_grad_clip', False):
            torch.nn.utils.clip_grad_norm_(list(self.net_g.parameters()), 0.01)
            
        # Update parameters
        self.optimizer_total.step()
        
        # Log losses
        self.log_dict = self.reduce_loss_dict(loss_dict)

    def test(self):
        """Run inference on the model"""
        self.lq = self.lq_rgb
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
        gt = F.pad(self.gt, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        
        # Run inference
        if hasattr(self, 'net_g_ema'):
            logger = get_root_logger()
            logger.warning("EMA network exists but is not implemented properly")
        else:
            # Switch to eval mode
            self.net_le.eval()
            self.net_g.eval()
            
            with torch.no_grad():
                # Forward pass
                input_features = [img, img0, img45, img90, img135, img_aolp, img_dolp]
                prior = self.net_le(input_features, gt)
                self.output = self.net_g(img, prior)
            
            # Switch back to training mode
            self.net_le.train()
            self.net_g.train()
        
        # Remove padding
        _, _, h, w = self.output.size()
        self.output = self.output[:, :, 0:h - mod_pad_h * scale, 0:w - mod_pad_w * scale]

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        """Distributed validation function"""
        if self.opt['rank'] == 0:
            self.nondist_validation(dataloader, current_iter, tb_logger, save_img)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        """Validation function"""
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        use_pbar = self.opt['val'].get('pbar', False)
        
        # Initialize metrics if needed
        if with_metrics:
            if not hasattr(self, 'metric_results'):
                self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
            self._initialize_best_metric_results(dataset_name)
            # Reset metric results for this validation run
            self.metric_results = {metric: 0 for metric in self.metric_results}
        
        metric_data = dict()
        if use_pbar:
            pbar = tqdm(total=len(dataloader), unit='image')
        
        # Process each validation image
        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            
            # Feed data and run inference
            self.feed_data(val_data)
            self.test()
            
            # Get output visuals
            visuals = self.get_current_visuals()
            sr_img = tensor2img([visuals['result']])
            lq_img = tensor2img([visuals['lq']])
            
            # Prepare metric data
            metric_data['img'] = sr_img
            if 'gt' in visuals:
                gt_img = tensor2img([visuals['gt']])
                metric_data['img2'] = gt_img
                del self.gt
            
            # Free memory
            del self.lq
            del self.output
            torch.cuda.empty_cache()
            
            # Save output image if requested
            if save_img:
                self._save_validation_images(img_name, sr_img, current_iter, dataset_name)
            
            # Calculate metrics if needed
            if with_metrics:
                for name, opt_ in self.opt['val']['metrics'].items():
                    self.metric_results[name] += calculate_metric(metric_data, opt_)
            
            # Update progress bar
            if use_pbar:
                pbar.update(1)
                pbar.set_description(f'Test {img_name}')
        
        if use_pbar:
            pbar.close()
        
        # Process and log metrics
        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= (idx + 1)
                # Update the best metric result
                self._update_best_metric_result(dataset_name, metric, self.metric_results[metric], current_iter)
            
            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)

    def _save_validation_images(self, img_name, sr_img, current_iter, dataset_name):
        """Save validation output images"""
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

    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        """Log validation metric values"""
        log_str = f'Validation {dataset_name}\n'
        
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}'
            if hasattr(self, 'best_metric_results'):
                log_str += (f'\tBest: {self.best_metric_results[dataset_name][metric]["val"]:.4f} @ '
                            f'{self.best_metric_results[dataset_name][metric]["iter"]} iter')
            log_str += '\n'
        
        # Log to file
        logger = get_root_logger()
        logger.info(log_str)
        
        # Log to tensorboard
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{dataset_name}/{metric}', value, current_iter)

    def get_current_visuals(self):
        """Return current visuals for display or evaluation"""
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
            
        return out_dict

    def save(self, epoch, current_iter):
        """Save network parameters and training state"""
        if hasattr(self, 'net_g_ema'):
            logger = get_root_logger()
            logger.warning("EMA network saving not implemented")
        else:
            self.save_network(self.net_g, 'net_g', current_iter)
            self.save_network(self.net_le, 'net_le', current_iter)
        
        self.save_training_state(epoch, current_iter)
