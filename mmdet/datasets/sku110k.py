import json
from pathlib import Path

from mmdet.datasets import CocoDataset
import pandas as pd
from torchvision.datasets.utils import download_and_extract_archive

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
class SKU110KDataset(CocoDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, classes=('product',), **kwargs)
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

    def xyxy_to_coco_xywh(self, xyxy_bbox):
        x1, y1, x2, y2 = xyxy_bbox
        w = x2 - x1
        h = y2 - y1
        return [int(x1), int(y1), int(w), int(h)]

    def xyxy_to_coco_area(self, xyxy_bbox):
        x1, y1, x2, y2 = xyxy_bbox
        w = x2 - x1
        h = y2 - y1
        return int(w * h)

    def df_to_coco(self, df: pd.DataFrame):
        info = {
            "year": 2018,
            "version": "",
            "description": "",
            "contributor": "",
            "url": DATA_URL,
            "data_created": "2018/01/01"
        }

        groupby = list(df.groupby(['image_name',
                                   'image_width',
                                   'image_height']))
        images = [
            {
                'id': idx,
                'file_name': image_name,
                'width': int(width),
                'height': int(height),
            }
            for idx, ((image_name, width, height), group)
            in enumerate(groupby)
        ]

        annots = [
            {
                'image_id': img_idx,
                'category_id': 1,
                'bbox': self.xyxy_to_coco_xywh(bbox),
                'area': self.xyxy_to_coco_area(bbox),
                'segmentation': [],
                'iscrowd': 0
            }
            for img_idx, ((image_name, width, height), group)
            in enumerate(groupby)
            for bbox in group[['x1', 'y1', 'x2', 'y2']].values
        ]

        annots = [
            {
                'id': idx,
                **annot
            }
            for idx, annot in enumerate(annots)
        ]

        categories = [
            {
                'id': 1,
                'name': 'product'
            }
        ]

        return {
            'info': info,
            'images': images,
            'annotations': annots,
            'licenses': [],
            'categories': categories,
        }

    def load_annotations(self, ann_file):
        ann_path = Path(self.ann_file)
        coco_json_path = ann_path.parent / 'coco' / f'{ann_path.stem}.json'

        if not coco_json_path.parent.exists():
            coco_json_path.parent.mkdir(parents=True)

        if not coco_json_path.exists():
            df = pd.read_csv(ann_file, names=COLUMN_NAMES)
            df = df[df['image_name'].apply(lambda img:
                                           img not in CORRUPT_IMAGES)]
            json.dump(self.df_to_coco(df), coco_json_path.open('w'))

        return super().load_annotations(str(coco_json_path))
