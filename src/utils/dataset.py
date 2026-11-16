import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torch.utils.data as data
import pandas as pd
import utils.tools as tools

# The UCF features are ~3.4 GB in total and every epoch touches all of them, so
# re-reading them per epoch is pure disk traffic.  Keep them in RAM: the dict is
# filled in the parent process before DataLoader forks, which makes the sample
# buffers shareable read-only pages instead of one private copy per worker.
_CACHE = {}
_CACHE_MAX = int(os.environ.get('VAD_CACHE_MAX', '60000'))


def _load_feature(path):
    arr = _CACHE.get(path)
    if arr is None:
        arr = np.load(path)
        if len(_CACHE) < _CACHE_MAX:
            _CACHE[path] = arr
    return arr


def _prewarm(paths):
    paths = [p for p in dict.fromkeys(paths) if p not in _CACHE]
    if not paths:
        return 0
    with ThreadPoolExecutor(max_workers=32) as ex:
        for arr, p in zip(ex.map(_load_feature, paths), paths):
            _CACHE[p] = arr
    return len(paths)

class UCFDataset(data.Dataset):
    def __init__(self, clip_dim: int, file_path: str, test_mode: bool, label_map: dict, normal: bool = False):
        self.df = pd.read_csv(file_path)
        self.clip_dim = clip_dim
        self.test_mode = test_mode
        self.label_map = label_map
        self.normal = normal
        if normal == True and test_mode == False:
            self.df = self.df.loc[self.df['label'] == 'Normal']
            self.df = self.df.reset_index()
        elif test_mode == False:
            self.df = self.df.loc[self.df['label'] != 'Normal']
            self.df = self.df.reset_index()
        if os.environ.get('VAD_PREWARM', '1') == '1':
            _prewarm(self.df['path'].tolist())
        
    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        row = self.df.iloc[index]
        clip_feature = _load_feature(row['path'])
        if self.test_mode == False:
            clip_feature, clip_length = tools.process_feat(clip_feature, self.clip_dim)
        else:
            clip_feature, clip_length = tools.process_split(clip_feature, self.clip_dim)

        clip_feature = torch.tensor(clip_feature)
        clip_label = row['label']
        return clip_feature, clip_label, clip_length

class XDDataset(data.Dataset):
    def __init__(self, clip_dim: int, file_path: str, test_mode: bool, label_map: dict, is_random: bool = False):
        self.df = pd.read_csv(file_path)
        self.clip_dim = clip_dim
        self.test_mode = test_mode
        self.label_map = label_map
        self.is_random = is_random
        
    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        clip_feature = _load_feature(self.df.loc[index]['path'])
        if self.test_mode == False:
            clip_feature, clip_length = tools.process_feat(clip_feature, self.clip_dim, is_random=self.is_random)
        else:
            clip_feature, clip_length = tools.process_split(clip_feature, self.clip_dim)

        clip_feature = torch.tensor(clip_feature)
        clip_label = self.df.loc[index]['label']
        return clip_feature, clip_label, clip_length