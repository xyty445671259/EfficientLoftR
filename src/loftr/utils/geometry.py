import torch
import cv2
import numpy as np 

@torch.no_grad()
def warp_kpts(kpts0, depth0, depth1, T_0to1, K0, K1):
    """ 
    把一张图像上的关键点通过深度和相机位姿，投影/变换到另一张图像上，并且检查投影是否合理（比如深度一致以及是否在图像内）
    Warp kpts0 from I0 to I1 with depth, K and Rt
    Also check covisibility and depth consistency.
    Depth is consistent if relative error < 0.2 (hard-coded).
    
    Args:
        kpts0 (torch.Tensor): [N, L, 2] - <x, y>,
        depth0 (torch.Tensor): [N, H, W],
        depth1 (torch.Tensor): [N, H, W],
        T_0to1 (torch.Tensor): [N, 3, 4],
        K0 (torch.Tensor): [N, 3, 3],
        K1 (torch.Tensor): [N, 3, 3],
    Returns:
        calculable_mask (torch.Tensor): [N, L]
        warped_keypoints0 (torch.Tensor): [N, L, 2] <x0_hat, y1_hat>
    """
    kpts0_long = kpts0.round().long()

    # Sample depth, get calculable_mask on depth != 0
    kpts0_depth = torch.stack(
        [depth0[i, kpts0_long[i, :, 1], kpts0_long[i, :, 0]] for i in range(kpts0.shape[0])], dim=0
    )  # (N, L), 是深度值标量，不是坐标
    nonzero_mask = kpts0_depth != 0

    # Unproject
    #将2D坐标转换为齐次坐标，乘以深度值, 然后通过相机的内参矩阵的逆变换到相机坐标系
    kpts0_h = torch.cat([kpts0, torch.ones_like(kpts0[:, :, [0]])], dim=-1) * kpts0_depth[..., None]  # (N, L, 3)
    kpts0_cam = K0.inverse() @ kpts0_h.transpose(2, 1)  # (N, 3, L)

    # Rigid Transform
    #刚性变换，使用T_0to1将点从相机0坐标系变换到相机1坐标系
    w_kpts0_cam = T_0to1[:, :3, :3] @ kpts0_cam + T_0to1[:, :3, [3]]    # (N, 3, L)
    w_kpts0_depth_computed = w_kpts0_cam[:, 2, :]

    # Project
    #重新投影到2D平面，使用相机1的内参矩阵将3D点投影到图像平面。
    w_kpts0_h = (K1 @ w_kpts0_cam).transpose(2, 1)  # (N, L, 3)
    w_kpts0 = w_kpts0_h[:, :, :2] / (w_kpts0_h[:, :, [2]] + 1e-4)  # (N, L, 2), +1e-4 to avoid zero depth

    # Covisible Check
    #检查变换后的点是否在图像边界内，并验证深度一致性
    h, w = depth1.shape[1:3]
    covisible_mask = (w_kpts0[:, :, 0] > 0) * (w_kpts0[:, :, 0] < w-1) * \
        (w_kpts0[:, :, 1] > 0) * (w_kpts0[:, :, 1] < h-1)
    w_kpts0_long = w_kpts0.long()
    w_kpts0_long[~covisible_mask, :] = 0

    w_kpts0_depth = torch.stack(
        [depth1[i, w_kpts0_long[i, :, 1], w_kpts0_long[i, :, 0]] for i in range(w_kpts0_long.shape[0])], dim=0
    )  # (N, L)
    consistent_mask = ((w_kpts0_depth - w_kpts0_depth_computed) / w_kpts0_depth).abs() < 0.2
    valid_mask = nonzero_mask * covisible_mask * consistent_mask

    return valid_mask, w_kpts0


# 使用标注点估计变换矩阵（单应性矩阵或仿射变换）
def estimate_transform(pts_src, pts_dst):
    # 使用RANSAC等方法估计变换矩阵
    M, mask = cv2.findHomography(pts_src, pts_dst, cv2.RANSAC, 5.0)
    return M

# 应用变换到所有网格点
def apply_transform(points, transform_matrix, h=None, w=None, str=None):
    # 确保变换矩阵有正确的形状
    if transform_matrix.dim() == 2:
        transform_matrix = transform_matrix.unsqueeze(0)  # [1, 3, 3]
    if points.dim() == 2:
        points = points.unsqueeze(0)  # [1, L, 2]
    
    # 将点转换为齐次坐标
    ones = torch.ones_like(points[:, :, :1])
    points_homo = torch.cat([points, ones], dim=-1)  # [N, L, 3]
    
    # 应用变换
    points_homo = points_homo.transpose(1, 2)  # [N, 3, L]
    transformed_homo = torch.bmm(transform_matrix, points_homo)  # [N, 3, L]
    transformed_homo = transformed_homo.transpose(1, 2)  # [N, L, 3]
    
    # 转换回笛卡尔坐标
    transformed_points = transformed_homo[:, :, :2] / (transformed_homo[:, :, 2:] + 1e-6)
    if str =='fine':
        visible_mask = (points[:, :, 0]>=0) & (points[:, :, 0]<w) & (points[:, :, 1]>=0) & (points[:,:,1] < h)
        return visible_mask, transformed_points
    return transformed_points

