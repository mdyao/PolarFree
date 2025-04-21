from torch.utils import data
import torch
import cv2
import numpy as np
import os
import glob

from torchvision.transforms.functional import normalize
from basicsr.utils import FileClient
from basicsr.utils.registry import DATASET_REGISTRY
from polarfree.utils.transforms import augment


@DATASET_REGISTRY.register()
class PairedImagePolarDataset(data.Dataset):
    """Paired image dataset for polarization image restoration.
    
    Args:
        opt (dict): Configuration options including:
            dataroot_gt (str): Ground truth data path
            dataroot_lq (str): Low quality data path
            io_backend (dict): IO backend settings
            filename_tmpl (str): Filename template
            gt_size (int): Cropped patch size for training
            phase (str): 'train' or 'val'/'test'
            test_scenes (list): Scenes used for testing
            easy_data_ratio/hard_data_ratio (float): Sampling ratios
    """
    def __init__(self, opt):
        super(PairedImagePolarDataset, self).__init__()
        self.opt = opt
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt.get('mean')
        self.std = opt.get('std')
        self.test_scenes = opt['test_scenes']
        self.gt_folder, self.lq_folder = opt['dataroot_gt'], opt['dataroot_lq']
        self.filename_tmpl = opt.get('filename_tmpl', '{}')
        
        # Build paths list based on training or testing phase
        self._build_paths()
            
    def _build_paths(self):
        """Build paths list for training or testing"""
        self.paths = []
        
        if self.opt['phase'] == 'train':
            for scene_complexity in ['easy', 'hard']:
                ratio = self.opt['easy_data_ratio'] if scene_complexity == 'easy' else self.opt['hard_data_ratio']
                scenes = glob.glob(os.path.join(self.lq_folder, scene_complexity, 'input', '*'))
                
                for scene_ in scenes:
                    scene = scene_.split('/')[-1]
                    if scene not in self.test_scenes:
                        tmp_paths = sorted(glob.glob(os.path.join(
                            self.lq_folder, scene_complexity, 'input', scene, '*_000.png'))) * ratio
                        self.paths += tmp_paths
                
                label = "easy training data" if scene_complexity == 'easy' else "total training data"
                print(f"{label}: {len(self.paths)}")
        else:
            for scene in self.test_scenes:
                tmp_paths = sorted(glob.glob(os.path.join(
                    self.lq_folder, 'input', scene, '*_000.png')))
                self.paths += tmp_paths
                
        self.paths = sorted(self.paths)

    def calculate_ADoLP(self, img0, img45, img90, img135):
        """Calculate Angle and Degree of Linear Polarization from the four polarized images"""
        # Calculate Stokes parameters
        I = 0.5 * (img0 + img45 + img90 + img135) + 1e-4
        Q = img0 - img90
        U = img45 - img135
        
        # Avoid division by zero
        Q[Q == 0] = 0.0001
        I[I == 0] = 0.0001
        
        # Calculate DoLP and AoLP
        DoLP = np.sqrt(Q**2 + U**2) / I
        DoLP[DoLP > 1] = 1
        AoLP = 0.5 * np.arctan2(U, Q)
        
        return AoLP, DoLP

    def gen_stokes(self, img0, img45, img90, img135, img_rgb):
        """Generate Stokes parameters and derived quantities"""
        aolp, dolp = self.calculate_ADoLP(img0, img45, img90, img135)
        
        # Add channel dimension
        aolp = aolp[None, :, :]
        dolp = dolp[None, :, :]
        img0 = img0[None, :, :]
        img45 = img45[None, :, :]
        img90 = img90[None, :, :]
        img135 = img135[None, :, :]
        
        # Calculate polarized and non-polarized components
        Ip = aolp * img_rgb
        Inp = img_rgb - Ip
        
        return img0, img45, img90, img135, aolp, dolp, Ip, Inp
    
    def bgr2gray(self, img):
        """Convert BGR image to grayscale and normalize to [0,1]"""
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        return img_gray / 255.0
    
    def _load_images(self, lq_path):
        """Load all required images given a base path"""
        img_id = lq_path.split('/')[-1].split('_')[0]
        scene_path = lq_path.split('/')[-2]
        tmp_root_path = '/'.join(lq_path.split('/')[:-3])
        
        # Low quality image paths
        lq_paths = {
            'img0': os.path.join(tmp_root_path, 'input', scene_path, f"{img_id}_000.png"),
            'img45': os.path.join(tmp_root_path, 'input', scene_path, f"{img_id}_045.png"),
            'img90': os.path.join(tmp_root_path, 'input', scene_path, f"{img_id}_090.png"),
            'img135': os.path.join(tmp_root_path, 'input', scene_path, f"{img_id}_135.png"),
            'rgb': os.path.join(tmp_root_path, 'input', scene_path, f"{img_id}_rgb.png")
        }
        
        # Ground truth paths
        gt_root_path = '/'.join(lq_paths['img0'].split('/')[:-3])
        gt_rgb_path = glob.glob(os.path.join(gt_root_path, 'gt', scene_path, '*_rgb.png'))[0]
        
        # Load and preprocess images
        lq_images = {}
        for key, path in lq_paths.items():
            if 'rgb' in key:
                img = cv2.imread(path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
                img = np.transpose(img, (2, 0, 1))  # HWC to CHW
            else:
                img = self.bgr2gray(cv2.imread(path))
            lq_images[key] = img
            
        gt_rgb = cv2.imread(gt_rgb_path)
        gt_rgb = cv2.cvtColor(gt_rgb, cv2.COLOR_BGR2RGB) / 255.0
        gt_rgb = np.transpose(gt_rgb, (2, 0, 1))  # HWC to CHW
        
        return lq_images, gt_rgb, lq_paths['rgb']

    def _random_crop(self, images, gt_rgb, GT_size):
        """Apply random cropping to all images"""
        _, h, w = gt_rgb.shape
        random_h = np.random.randint(0, h - GT_size)
        random_w = np.random.randint(0, w - GT_size)
        
        # Apply crop to all images
        cropped = {}
        for k, v in images.items():
            if isinstance(v, np.ndarray) and v.ndim > 1:
                cropped[k] = v[:, random_h:random_h+GT_size, random_w:random_w+GT_size]
            else:
                cropped[k] = v
                
        # Crop ground truth
        gt_rgb_cropped = gt_rgb[:, random_h:random_h+GT_size, random_w:random_w+GT_size]
        
        return cropped, gt_rgb_cropped

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        index = index % len(self.paths)
        lq_path = self.paths[index]
        
        # Load all required images
        lq_images, gt_rgb, lq_rgb_path = self._load_images(lq_path)
        
        # Generate Stokes parameters
        img0, img45, img90, img135, aolp, dolp, Ip, Inp = self.gen_stokes(
            lq_images['img0'], lq_images['img45'], lq_images['img90'], lq_images['img135'], lq_images['rgb']
        )
        
        # Process images for training (crop and augment)
        if self.opt['phase'] == 'train':
            GT_size = self.opt['gt_size']
            
            # Create dictionary for all images that need processing
            all_images = {
                'img0': img0, 'img45': img45, 'img90': img90, 'img135': img135,
                'rgb': lq_images['rgb'], 'aolp': aolp, 'dolp': dolp, 'Ip': Ip, 'Inp': Inp
            }
            
            # Apply random crop
            all_images, gt_rgb = self._random_crop(all_images, gt_rgb, GT_size)
            
            # Apply augmentation
            augment_list = [all_images['img0'], all_images['img45'], 
                           all_images['img90'], all_images['img135'], 
                           all_images['rgb'], gt_rgb, 
                           all_images['aolp'], all_images['dolp'], 
                           all_images['Ip'], all_images['Inp']]
            
            augmented = augment(augment_list)
            
            # Unpack augmented results
            img0, img45, img90, img135, lq_rgb, gt_rgb, aolp, dolp, Ip, Inp = augmented
        else:
            lq_rgb = lq_images['rgb']
        
        # Convert all numpy arrays to PyTorch tensors
        def to_tensor(x):
            return torch.from_numpy(np.ascontiguousarray(x.copy())).float()
        
        result = {
            'lq_img0': to_tensor(img0),
            'lq_img45': to_tensor(img45),
            'lq_img90': to_tensor(img90),
            'lq_img135': to_tensor(img135),
            'lq_rgb': to_tensor(lq_rgb),
            'lq_aolp': to_tensor(aolp),
            'lq_dolp': to_tensor(dolp),
            'lq_Ip': to_tensor(Ip),
            'lq_Inp': to_tensor(Inp),
            'gt_img0': to_tensor(gt_rgb),
            'gt_img45': to_tensor(gt_rgb),
            'gt_img90': to_tensor(gt_rgb),
            'gt_img135': to_tensor(gt_rgb),
            'gt_rgb': to_tensor(gt_rgb),
            'lq_path': lq_rgb_path
        }
        
        return result

    def __len__(self):
        return len(self.paths)
