"""Data-path smoke check: do the CSV feature paths resolve on this machine?"""
import os

import numpy as np
import pandas as pd

for name, csv in (('ucf-train', '../list/ucf_CLIP_rgb.csv'),
                  ('ucf-test', '../list/ucf_CLIP_rgbtest.csv')):
    d = pd.read_csv(csv)
    paths = d['path'].tolist()
    missing = [p for p in paths[:50] if not os.path.exists(p)]
    a = np.load(paths[0])
    print(f'[{name}] rows={len(d)}  first={paths[0]}')
    print(f'[{name}] shape={a.shape} dtype={a.dtype}  missing(first 50)={len(missing)}')
    print(f'[{name}] labels={d["label"].value_counts().to_dict()}')
