import os.path as osp
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from loguru import logger

from src.utils.dataset import read_megadepth_gray, read_megadepth_depth


class StainDataset(Dataset):
    def __init__(self,
                 root_dir,
                 npz_path,
                 mode='train',
                 min_overlap_score=0.4,
                 img_resize=None,
                 df=None,
                 img_padding=False,
                 depth_padding=False,
                 augment_fn=None,
                 fp16=False,
                 **kwargs):
        """
        Manage one scene(npz_path) of MegaDepth dataset.
        
        Args:
            root_dir (str): megadepth root directory that has `phoenix`.
            npz_path (str): {scene_id}.npz path. This contains image pair information of a scene.
            mode (str): options are ['train', 'val', 'test']
            min_overlap_score (float): how much a pair should have in common. In range of [0, 1]. Set to 0 when testing.
            img_resize (int, optional): the longer edge of resized images. None for no resize. 640 is recommended.
                                        This is useful during training with batches and testing with memory intensive algorithms.
            df (int, optional): image size division factor. NOTE: this will change the final image size after img_resize.
            img_padding (bool): If set to 'True', zero-pad the image to squared size. This is useful during training.
            depth_padding (bool): If set to 'True', zero-pad depthmap to (2000, 2000). This is useful during training.
            augment_fn (callable, optional): augments images with pre-defined visual effects.
        """
        super().__init__()
        self.root_dir = root_dir
        self.mode = mode
        self.scene_id = npz_path.split('.')[0]

        # prepare scene_info and pair_info
        if mode == 'test' and min_overlap_score != 0:
            logger.warning("You are using `min_overlap_score`!=0 in test mode. Set to 0.")
            min_overlap_score = 0
        self.scene_info = dict(np.load(npz_path, allow_pickle=True))
        self.pair_infos = self.scene_info['pair_infos'].copy()

        del self.scene_info['pair_infos']
        self.pair_infos = [pair_info for pair_info in self.pair_infos if pair_info[1] > min_overlap_score]

        # parameters for image resizing, padding and depthmap padding
        if mode == 'train':
            assert img_resize is not None and img_padding
        self.img_resize = img_resize
        self.df = df

        
        self.img_padding = img_padding

        # for training LoFTR
        self.augment_fn = augment_fn if mode == 'train' else None
        self.coarse_scale = getattr(kwargs, 'coarse_scale', 0.0625)
        
        self.fp16 = fp16
        self.image_to_annotation = {}
    
        # 遍历所有配对信息
        for pair_idx, ((img_idx0, img_idx1), _) in enumerate(self.pair_infos):
            # 获取这个配对的标注点
            ann0 = self.scene_info['annotated0'][pair_idx]
            ann1 = self.scene_info['annotated1'][pair_idx]
            
            # 映射图像索引到对应的标注点
            self.image_to_annotation[img_idx0] = ann0
            self.image_to_annotation[img_idx1] = ann1
        
    def __len__(self):
        return len(self.pair_infos)

    def __getitem__(self, idx):
        (idx0, idx1), overlap_score = self.pair_infos[idx]

        # read grayscale image and mask. (1, h, w) and (h, w)
        img_name0 = osp.join(self.root_dir, self.scene_info['image_paths'][idx0])
        img_name1 = osp.join(self.root_dir, self.scene_info['image_paths'][idx1])
        
        #标注点
        annotated0 = self.image_to_annotation[idx0]
        annotated1 = self.image_to_annotation[idx1]

        max_points= 20
        if len(annotated0) > max_points:
            indices = torch.randperm(len(annotated0))[:max_points]
            annotated0 = annotated0[indices]
            annotated1 = annotated1[indices]

        # TODO: Support augmentation & handle seeds for each worker correctly.
        image0, mask0, scale0 = read_megadepth_gray(
            img_name0, self.img_resize, self.df, self.img_padding, None)
            # np.random.choice([self.augment_fn, None], p=[0.5, 0.5]))
        image1, mask1, scale1 = read_megadepth_gray(
            img_name1, self.img_resize, self.df, self.img_padding, None)
            # np.random.choice([self.augment_fn, None], p=[0.5, 0.5]))

        if self.fp16:
            image0, image1, scale0, scale1 = map(lambda x: x.half(),
                                                                 [image0, image1, scale0, scale1])
        data = {
            'image0': image0,  # (1, h, w)
            'image1': image1,
            'scale0': scale0,  # [scale_w, scale_h]
            'scale1': scale1,
            'dataset_name': 'Stain',
            'scene_id': self.scene_id,
            'pair_id': idx,
            'pair_names': (self.scene_info['image_paths'][idx0], self.scene_info['image_paths'][idx1]),
            'annotated0': annotated0,
            'annotated1': annotated1
        }
        # for LoFTR training, (H,W)->(2,H,W)->(1,2,H,W)->float->使用最邻近插值->(2,H',W')->bool->(H',W')
        #这里的interpolate跟补全是不一样的。它是查找特征图对应的原图，只要原图所在的感受野有一个像素为True，那么这个特征图也为True。为的是让有效区域与特征图一一对应
        if mask0 is not None:  # img_padding is True
            if self.coarse_scale:
                [ts_mask_0, ts_mask_1] = F.interpolate(torch.stack([mask0, mask1], dim=0)[None].float(),
                                                        scale_factor=self.coarse_scale,
                                                        mode='nearest',
                                                        recompute_scale_factor=False)[0].bool()
            data.update({'mask0': ts_mask_0, 'mask1': ts_mask_1})

        return data
