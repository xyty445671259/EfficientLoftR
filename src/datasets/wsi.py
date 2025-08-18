import os.path as osp
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from loguru import logger
from src.utils.dataset import read_scannet_gray 

class WSIDataset(Dataset):
    def __init__(self,
                 root_dir,
                 npz_path,
                 mode='train',
                 min_overlap_score=0.4,
                 augment_fn=None,
                 img_resize=None,
                 fp16=False,
                 **kwargs):
        """
        WSI 染色图像数据集 (HE, MASSON, PASM)
        
        Args:
            root_dir (str): 图像根目录
            npz_path (str): 包含点匹配信息的 NPZ 文件路径
            mode (str): ['train', 'val', 'test']
            min_overlap_score (float): 最小重叠分数阈值
            img_resize (tuple): 图像调整尺寸 (宽, 高)
            augment_fn (callable): 数据增强函数
        """
        super().__init__()
        self.root_dir = root_dir
        self.mode = mode
        self.img_resize = img_resize
        self.augment_fn = augment_fn
        self.fp16 = fp16
        
        # 加载点匹配数据 (兼容 ScanNet/MegaDepth 格式)
        npz_data = np.load(npz_path, allow_pickle=True)
        
        self.pair_names = npz_data['name']  # 图像对名称
        self.matches = npz_data['matches']  # 匹配点坐标
        self.overlap_scores = npz_data['overlap_scores'] if 'overlap_scores' in npz_data else np.ones(len(self.pair_names))
        
        # 过滤低重叠度的图像对
        if min_overlap_score > 0 and mode != 'test':
            valid_mask = self.overlap_scores > min_overlap_score
            if 'pair_infos' in locals():
                self.pair_infos = [info for i, info in enumerate(self.pair_infos) if valid_mask[i]]
            else:
                self.pair_names = self.pair_names[valid_mask]
                self.matches = self.matches[valid_mask]
                self.overlap_scores = self.overlap_scores[valid_mask]
        
        logger.info(f'Loaded {self.__len__()} image pairs for {mode}')
    
    def __len__(self):
        if hasattr(self, 'pair_infos'):
            return len(self.pair_infos)
        return len(self.pair_names)
    
    def _load_wsi_image(self, img_path):
        """加载并预处理 WSI 染色图像"""
        img = Image.open(img_path)
        
        # 确保为 RGB 格式
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # 调整大小
        if self.img_resize:
            img = img.resize(self.img_resize, Image.LANCZOS)
        
        # 转换为灰度图 (兼容原始模型)
        img_gray = img.convert('L')
        img_array = np.array(img_gray)
        
        # 添加通道维度
        return img_array[None]  # (1, H, W)
    
    def __getitem__(self, idx):
        # 获取图像对信息
        if hasattr(self, 'pair_infos'):  # MegaDepth 格式
            (idx0, idx1), overlap_score, central_matches = self.pair_infos[idx]
            img_name0 = osp.join(self.root_dir, self.scene_info['image_paths'][idx0])
            img_name1 = osp.join(self.root_dir, self.scene_info['image_paths'][idx1])
            matches = central_matches[:, :4]  # (x0, y0, x1, y1)
        else:  # 自定义格式
            scene_name, scene_sub_name, stem_name_0, stem_name_1 = self.pair_names[idx]
            img_name0 = osp.join(self.root_dir, f'{stem_name_0}.png')
            img_name1 = osp.join(self.root_dir, f'{stem_name_1}.png')
            matches = self.matches[idx]
            overlap_score = self.overlap_scores[idx]
        
        # 加载图像
        image0 = self._load_wsi_image(img_name0)
        image1 = self._load_wsi_image(img_name1)
        
        # 提取匹配点
        kpts0 = matches[:, :2]  # (x0, y0)
        kpts1 = matches[:, 2:]  # (x1, y1)
        
        # 数据增强 (兼容 ScanNet/MegaDepth)
        if self.augment_fn and self.mode == 'train':
            # 随机应用增强
            if np.random.rand() > 0.5:
                image0, image1, kpts0, kpts1 = self.augment_fn(image0, image1, kpts0, kpts1)
        
        # 转换为 Tensor
        image0 = torch.tensor(image0).float() / 255.0  # (1, H, W)
        image1 = torch.tensor(image1).float() / 255.0
        
        # 生成伪深度图 (空) 和位姿 (单位矩阵)
        depth0 = depth1 = torch.tensor([])
        T_0to1 = T_1to0 = torch.eye(4)
        
        # 生成伪相机内参 (基于图像尺寸)
        H, W = image0.shape[1:]
        K = torch.tensor([
            [max(H, W), 0, W/2],
            [0, max(H, W), H/2],
            [0, 0, 1]
        ], dtype=torch.float32)
        
        # 缩放因子 (设为1)
        scale0 = scale1 = torch.tensor([1.0, 1.0])
        
        # 半精度支持
        if self.fp16:
            image0 = image0.half()
            image1 = image1.half()
        
        # 返回数据字典 (兼容 ScanNet/MegaDepth 格式)
        data = {
            'image0': image0,        # (1, H, W)
            'image1': image1,        # (1, H, W)
            'depth0': depth0,        # 空张量
            'depth1': depth1,        # 空张量
            'T_0to1': T_0to1,        # (4, 4) 单位矩阵
            'T_1to0': T_1to0,        # (4, 4) 单位矩阵
            'K0': K,                 # (3, 3) 伪内参
            'K1': K,                 # (3, 3) 伪内参
            'scale0': scale0,        # [1.0, 1.0]
            'scale1': scale1,        # [1.0, 1.0]
            'kpts0': kpts0,          # (M, 2) 关键点
            'kpts1': kpts1,          # (M, 2) 关键点
            'overlap_score': overlap_score,
            'dataset_name': 'WSI',
            'scene_id': 'wsi_slide',
            'pair_id': idx,
            'pair_names': (img_name0, img_name1)
        }
        
        # 添加粗粒度掩码 (兼容 MegaDepth)
        if hasattr(self, 'coarse_scale'):
            h, w = image0.shape[1:]
            mask0 = torch.ones(1, h, w, dtype=torch.bool)
            mask1 = torch.ones(1, h, w, dtype=torch.bool)
            
            if self.coarse_scale:
                ts_mask0 = F.interpolate(
                    mask0.float()[None], 
                    scale_factor=self.coarse_scale,
                    mode='nearest',
                    recompute_scale_factor=False
                )[0].bool()
                
                ts_mask1 = F.interpolate(
                    mask1.float()[None], 
                    scale_factor=self.coarse_scale,
                    mode='nearest',
                    recompute_scale_factor=False
                )[0].bool()
                
                data.update({'mask0': ts_mask0, 'mask1': ts_mask1})
        
        return data