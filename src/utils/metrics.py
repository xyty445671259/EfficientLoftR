import torch
import cv2
import numpy as np
from collections import OrderedDict
from loguru import logger
from kornia.geometry.epipolar import numeric
from kornia.geometry.conversions import convert_points_to_homogeneous
import pprint


# --- METRICS ---

def relative_pose_error(T_0to1, R, t, ignore_gt_t_thr=0.0):
    # angle error between 2 vectors
    t_gt = T_0to1[:3, 3]
    n = np.linalg.norm(t) * np.linalg.norm(t_gt)
    t_err = np.rad2deg(np.arccos(np.clip(np.dot(t, t_gt) / n, -1.0, 1.0)))
    t_err = np.minimum(t_err, 180 - t_err)  # handle E ambiguity
    if np.linalg.norm(t_gt) < ignore_gt_t_thr:  # pure rotation is challenging
        t_err = 0

    # angle error between 2 rotation matrices
    R_gt = T_0to1[:3, :3]
    cos = (np.trace(np.dot(R.T, R_gt)) - 1) / 2
    cos = np.clip(cos, -1., 1.)  # handle numercial errors
    R_err = np.rad2deg(np.abs(np.arccos(cos)))

    return t_err, R_err


def symmetric_epipolar_distance(pts0, pts1, E, K0, K1):
    """Squared symmetric epipolar distance.
    This can be seen as a biased estimation of the reprojection error.
    Args:
        pts0 (torch.Tensor): [N, 2]
        E (torch.Tensor): [3, 3]
    """
    pts0 = (pts0 - K0[[0, 1], [2, 2]][None]) / K0[[0, 1], [0, 1]][None]
    pts1 = (pts1 - K1[[0, 1], [2, 2]][None]) / K1[[0, 1], [0, 1]][None]
    pts0 = convert_points_to_homogeneous(pts0)
    pts1 = convert_points_to_homogeneous(pts1)

    Ep0 = pts0 @ E.T  # [N, 3]
    p1Ep0 = torch.sum(pts1 * Ep0, -1)  # [N,]
    Etp1 = pts1 @ E  # [N, 3]

    d = p1Ep0**2 * (1.0 / (Ep0[:, 0]**2 + Ep0[:, 1]**2) + 1.0 / (Etp1[:, 0]**2 + Etp1[:, 1]**2))  # N
    return d


def compute_symmetrical_epipolar_errors(data):
    """ 
    Update:
        data (dict):{"epi_errs": [M]}
    """
    Tx = numeric.cross_product_matrix(data['T_0to1'][:, :3, 3])
    E_mat = Tx @ data['T_0to1'][:, :3, :3]

    m_bids = data['m_bids']
    pts0 = data['mkpts0_f']
    pts1 = data['mkpts1_f']

    epi_errs = []
    for bs in range(Tx.size(0)):
        mask = m_bids == bs
        epi_errs.append(
            symmetric_epipolar_distance(pts0[mask], pts1[mask], E_mat[bs], data['K0'][bs], data['K1'][bs]))
    epi_errs = torch.cat(epi_errs, dim=0)

    data.update({'epi_errs': epi_errs})


def compute_reprojection_errors(data):
    """ 
    基于单应性矩阵计算重投影误差
    
    Update:
        data (dict):{"reproj_errs": [M]} 这里为了后续很多代码的修改，故意写成epi_errs，实际是投影误差
    """
    # 假设您有单应性矩阵 H_0to1
    H_0to1 = data['M_0to1']  # [3, 3] 或 [bs, 3, 3]
    
    pts0 = data['mkpts0_f']  # [M, 2]
    pts1 = data['mkpts1_f']  # [M, 2]
    
    # 将 pts0 转换为齐次坐标
    ones = torch.ones(pts0.shape[0], 1, device=pts0.device)
    pts0_homo = torch.cat([pts0, ones], dim=1)  # [M, 3]
    
    # 应用单应性变换
    pts0_transformed_homo = torch.matmul(pts0_homo, H_0to1.t())  # [M, 3]
    
    # 转换回笛卡尔坐标
    pts0_transformed = pts0_transformed_homo[:, :2] / pts0_transformed_homo[:, 2:3]
    
    # 计算重投影误差
    reproj_errs = torch.norm(pts0_transformed - pts1, dim=1)  # [M]
    
    data.update({'epi_errs': reproj_errs})


def estimate_pose(kpts0, kpts1, K0, K1, thresh, conf=0.99999):
    if len(kpts0) < 5:
        return None
    # normalize keypoints
    kpts0 = (kpts0 - K0[[0, 1], [2, 2]][None]) / K0[[0, 1], [0, 1]][None]
    kpts1 = (kpts1 - K1[[0, 1], [2, 2]][None]) / K1[[0, 1], [0, 1]][None]

    # normalize ransac threshold
    ransac_thr = thresh / np.mean([K0[0, 0], K1[1, 1], K0[0, 0], K1[1, 1]])

    # compute pose with cv2
    E, mask = cv2.findEssentialMat(
        kpts0, kpts1, np.eye(3), threshold=ransac_thr, prob=conf, method=cv2.RANSAC)
    if E is None:
        print("\nE is None while trying to recover pose.\n")
        return None

    # recover pose from E
    best_num_inliers = 0
    ret = None
    for _E in np.split(E, len(E) / 3):
        n, R, t, _ = cv2.recoverPose(_E, kpts0, kpts1, np.eye(3), 1e9, mask=mask)
        if n > best_num_inliers:
            ret = (R, t[:, 0], mask.ravel() > 0)
            best_num_inliers = n

    return ret


def estimate_lo_pose(kpts0, kpts1, K0, K1, thresh, conf=0.99999):
    from .warppers import Camera, Pose
    import poselib
    camera0, camera1 = Camera.from_calibration_matrix(K0).float(), Camera.from_calibration_matrix(K1).float()
    pts0, pts1 = kpts0, kpts1

    M, info = poselib.estimate_relative_pose(
        pts0,
        pts1,
        camera0.to_cameradict(),
        camera1.to_cameradict(),
        {
            "max_epipolar_error": thresh,
        },
    )
    success = M is not None and ( ((M.t != [0., 0., 0.]).all()) or ((M.q != [1., 0., 0., 0.]).all()) )
    if success:
        M = Pose.from_Rt(torch.tensor(M.R), torch.tensor(M.t)) # .to(pts0)
        # print(M)
    else:
        M = Pose.from_4x4mat(torch.eye(4).numpy()) # .to(pts0)
        # print(M)

    estimation = {
        "success": success,
        "M_0to1": M,
        "inliers": torch.tensor(info.pop("inliers")), # .to(pts0),
        **info,
    }
    return estimation


def compute_pose_errors(data, config):
    """ 
    Update:
        data (dict):{
            "R_errs" List[float]: [N]
            "t_errs" List[float]: [N]
            "inliers" List[np.ndarray]: [N]
        }
    """
    pixel_thr = config.TRAINER.RANSAC_PIXEL_THR  # 0.5
    conf = config.TRAINER.RANSAC_CONF  # 0.99999
    RANSAC = config.TRAINER.POSE_ESTIMATION_METHOD
    data.update({'R_errs': [], 't_errs': [], 'inliers': []})

    m_bids = data['m_bids'].cpu().numpy()
    pts0 = data['mkpts0_f'].cpu().numpy()
    pts1 = data['mkpts1_f'].cpu().numpy()
    K0 = data['K0'].cpu().numpy()
    K1 = data['K1'].cpu().numpy()
    T_0to1 = data['T_0to1'].cpu().numpy()

    for bs in range(K0.shape[0]):
        mask = m_bids == bs
        if config.LOFTR.EVAL_TIMES >= 1:
            bpts0, bpts1 = pts0[mask], pts1[mask]
            R_list, T_list, inliers_list = [], [], []
            # for _ in range(config.LOFTR.EVAL_TIMES):
            for _ in range(5):
                shuffling = np.random.permutation(np.arange(len(bpts0)))
                if _ >= config.LOFTR.EVAL_TIMES:
                    continue
                bpts0 = bpts0[shuffling]
                bpts1 = bpts1[shuffling]
                
                if RANSAC == 'RANSAC':
                    ret = estimate_pose(bpts0, bpts1, K0[bs], K1[bs], pixel_thr, conf=conf)
                    if ret is None:
                        R_list.append(np.inf)
                        T_list.append(np.inf)
                        inliers_list.append(np.array([]).astype(bool))
                    else:
                        R, t, inliers = ret
                        t_err, R_err = relative_pose_error(T_0to1[bs], R, t, ignore_gt_t_thr=0.0)
                        R_list.append(R_err)
                        T_list.append(t_err)
                        inliers_list.append(inliers)

                elif RANSAC == 'LO-RANSAC':
                    est = estimate_lo_pose(bpts0, bpts1, K0[bs], K1[bs], pixel_thr, conf=conf)
                    if not est["success"]:
                        R_list.append(90)
                        T_list.append(90)
                        inliers_list.append(np.array([]).astype(bool))
                    else:
                        M = est["M_0to1"]
                        inl = est["inliers"].numpy()
                        t_error, r_error = relative_pose_error(T_0to1[bs], M.R, M.t, ignore_gt_t_thr=0.0)
                        R_list.append(r_error)
                        T_list.append(t_error)
                        inliers_list.append(inl)
                else:
                    raise ValueError(f"Unknown RANSAC method: {RANSAC}")

            data['R_errs'].append(R_list)
            data['t_errs'].append(T_list)
            data['inliers'].append(inliers_list[0])


# --- METRIC AGGREGATION ---

def error_auc(errors, thresholds):
    """
    Args:
        errors (list): [N,]
        thresholds (list)
    """
    errors = [0] + sorted(list(errors))
    recall = list(np.linspace(0, 1, len(errors)))

    aucs = []
    thresholds = [5, 10, 20]
    for thr in thresholds:
        last_index = np.searchsorted(errors, thr)
        y = recall[:last_index] + [recall[last_index-1]]
        x = errors[:last_index] + [thr]
        aucs.append(np.trapz(y, x) / thr)

    return {f'auc@{t}': auc for t, auc in zip(thresholds, aucs)}


def epidist_prec(errors, thresholds, ret_dict=False):
    precs = []
    for thr in thresholds:
        prec_ = []
        for errs in errors:
            correct_mask = errs < thr
            prec_.append(np.mean(correct_mask) if len(correct_mask) > 0 else 0)
        precs.append(np.mean(prec_) if len(prec_) > 0 else 0)
    if ret_dict:
        return {f'prec@{t:.0e}': prec for t, prec in zip(thresholds, precs)}
    else:
        return precs


def aggregate_metrics(metrics, epi_err_thr=5e-4, config=None):
    """ Aggregate metrics for the whole dataset:
    (This method should be called once per dataset)
    1. AUC of the pose error (angular) at the threshold [5, 10, 20]
    2. Mean matching precision at the threshold 5e-4(ScanNet), 1e-4(MegaDepth)
    """
    # filter duplicates
    unq_ids = OrderedDict((iden, id) for id, iden in enumerate(metrics['identifiers']))
    unq_ids = list(unq_ids.values())
    logger.info(f'Aggregating metrics over {len(unq_ids)} unique items...')

    # pose auc
    angular_thresholds = [5, 10, 20]

    if config.LOFTR.EVAL_TIMES >= 1:
        pose_errors = np.max(np.stack([metrics['R_errs'], metrics['t_errs']]), axis=0).reshape(-1, config.LOFTR.EVAL_TIMES)[unq_ids].reshape(-1)
    else:
        pose_errors = np.max(np.stack([metrics['R_errs'], metrics['t_errs']]), axis=0)[unq_ids]
    aucs = error_auc(pose_errors, angular_thresholds)  # (auc@5, auc@10, auc@20)

    # matching precision
    dist_thresholds = [epi_err_thr]
    precs = epidist_prec(np.array(metrics['epi_errs'], dtype=object)[unq_ids], dist_thresholds, True)  # (prec@err_thr)
    
    u_num_mathces = np.array(metrics['num_matches'], dtype=object)[unq_ids]
    num_matches = {f'num_matches': u_num_mathces.mean() }
    return {**aucs, **precs, **num_matches}

def aggregate_metrics_(metrics, epi_err_thr=5e-4, config=None):
    """ Aggregate metrics for the whole dataset:
    (This method should be called once per dataset)
    1. 重投影误差统计
    2. 匹配精度
    """
    # filter duplicates
    unq_ids = OrderedDict((iden, id) for id, iden in enumerate(metrics['identifiers']))
    unq_ids = list(unq_ids.values())
    logger.info(f'Aggregating metrics over {len(unq_ids)} unique items...')

    # 重投影误差统计
    if 'epi_errs' in metrics and len(metrics['epi_errs']) > 0:
        # 获取所有重投影误差
        epi_errors = np.concatenate([np.array(err) for err in metrics['epi_errs']])
        
        # 计算各种统计量
        reproj_stats = {
            'reproj_mean': np.mean(epi_errors),
            'reproj_median': np.median(epi_errors),
            'reproj_std': np.std(epi_errors),
            'reproj_max': np.max(epi_errors),
            'reproj_min': np.min(epi_errors),
        }
        
        # 计算在不同阈值下的内点比率
        thresholds = [1, 2, 5, 10]  # 像素阈值
        inlier_ratios = {}
        for thr in thresholds:
            inlier_ratio = np.mean(epi_errors <= thr)
            inlier_ratios[f'inlier_ratio_{thr}px'] = inlier_ratio
    else:
        reproj_stats = {
            'reproj_mean': np.nan,
            'reproj_median': np.nan,
            'reproj_std': np.nan,
            'reproj_max': np.nan,
            'reproj_min': np.nan,
        }
        inlier_ratios = {}
    
    # 匹配数量统计
    if 'num_matches' in metrics:
        num_matches = np.array(metrics['num_matches'])[unq_ids]
        match_stats = {
            'num_matches_mean': np.mean(num_matches),
            'num_matches_median': np.median(num_matches),
            'num_matches_std': np.std(num_matches),
            'num_matches_max': np.max(num_matches),
            'num_matches_min': np.min(num_matches),
        }
    else:
        match_stats = {}
    
    # 如果有其他自定义指标，也可以在这里添加
    # 例如，如果您计算了基于单应性的内点比率
    if 'homography_inlier_ratio' in metrics:
        homography_stats = {
            'homography_inlier_mean': np.mean(metrics['homography_inlier_ratio']),
            'homography_inlier_median': np.median(metrics['homography_inlier_ratio']),
        }
    else:
        homography_stats = {}
    
    # 合并所有统计结果
    return {**reproj_stats, **inlier_ratios, **match_stats, **homography_stats}