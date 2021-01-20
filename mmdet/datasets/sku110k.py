import numpy as np
import pandas as pd
from mmdet.datasets import CustomDataset
from torchvision.datasets.utils import download_and_extract_archive
from pathlib import Path
import os.path

from .builder import DATASETS

DATA_URL = ('http://trax-geometry.s3.amazonaws.com'
            '/cvpr_challenge'
            '/SKU110K_fixed.tar.gz')

COLUMN_NAMES = ['image_name', 'x1', 'y1', 'x2', 'y2', 'class', 'image_width',
                'image_height']

CORRUPT_IMAGES = [
    'train_882.jpg',
    'train_924.jpg',
    'train_4222.jpg',
    'train_5822.jpg',
    'test_274.jpg'
]


@DATASETS.register_module()
class SKU110KDataset(CustomDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.download()

    def _check_exists(self):
        """Return True if the SKU110K dataset exists already.
        """
        return (
            Path(self.ann_file).exists()
            and Path(self.img_prefix).is_dir()
        )

    def download(self):
        if self._check_exists():
            return

        download_and_extract_archive(DATA_URL, Path(self.data_root).parent,
                                     filename=Path(self.data_root).name)
        os.path.pardir

    def load_annotations(self, ann_file):
        df = pd.read_csv(ann_file, names=COLUMN_NAMES)
        df = df[df['image_name'].apply(lambda img:
                                       img not in CORRUPT_IMAGES)]
        return [{'filename': image_name,
                 'width': width,
                 'height': height,
                 'ann': {'bboxes': (group[['x1', 'y1',
                                           'x2', 'y2']]
                                    .astype('float32').values),
                         'labels': np.array([0]*len(group))}}
                for (image_name,
                     width,
                     height), group
                in df.groupby(['image_name', 'image_width', 'image_height'])]
