import numpy as np


# --- PL-DATAMODULE ---

def get_local_split(items: list, world_size: int, rank: int, seed: int):
    """ The local rank only loads a split of the dataset. 
        用于分布式训练分割数据集, 在多个进程(GPU?)之间均匀地分配数据集, 确保每个进程获得不同的数据子集, 同时处理数据集不能被进程整除的情况
        items -- 完整的数据集列表
        world_size -- GPU数量?
        rank -- 当前进程的排名,0-7?
        seed --种子
    """
    n_items = len(items)
    items_permute = np.random.RandomState(seed).permutation(items) #创建随机排列
    if n_items % world_size == 0:
        padded_items = items_permute
    else:
        padding = np.random.RandomState(seed).choice(
            items,
            world_size - (n_items % world_size),
            replace=True) #有放回地随机选择项目作为填充. world_size - (n_items % world_size)为需要填充的项目数
        padded_items = np.concatenate([items_permute, padding])
        assert len(padded_items) % world_size == 0, \
            f'len(padded_items): {len(padded_items)}; world_size: {world_size}; len(padding): {len(padding)}'
    n_per_rank = len(padded_items) // world_size #每个进程应该获得的数据量
    local_items = padded_items[n_per_rank * rank: n_per_rank * (rank+1)] #根据当前进程排名，提取对应的切片

    return local_items
