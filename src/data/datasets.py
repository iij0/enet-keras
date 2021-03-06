# coding=utf-8
from __future__ import absolute_import, division, print_function
from pycocotools import mask
from pycocotools.coco import COCO
import numpy as np
import numbers
import abc
import os
from . import utils
from keras.utils import get_file


def load(dataset_name, data_dir=None, data_type=None):
    if dataset_name == 'mscoco':
        if data_dir is None or data_type is None:
            return MSCOCO
        else:
            return MSCOCO(data_dir=data_dir, data_type=data_type)
    elif dataset_name == 'mscoco_reduced':
        if data_dir is None or data_type is None:
            return MSCOCOReduced
        else:
            return MSCOCOReduced(data_dir=data_dir, data_type=data_type)
    else:
        raise NotImplementedError('Unknown dataset {}'.format('dataset_name'))


class Dataset(object):
    __metaclass__ = abc.ABCMeta

    NAME = 'dataset'
    CATEGORIES = []
    IDS = []
    PALETTE = []

    # def __init(self, data_dir, data_type):
    #     # this function should be overriden
    #     pass

    @abc.abstractmethod
    def load(self, data_dir, data_type):
        """Method documentation"""
    #
    # def id_to_category(self, primary_id):
    #     return self.CATEGORIES[primary_id]


class MSCOCO(Dataset):
    NAME = 'mscoco'
    CATEGORIES = [
        'background',  # class zero
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
        'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
        'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
        'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
        'hair drier', 'toothbrush']
    IDS = [
        0,
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17,
        18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36,
        37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53,
        54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73,
        74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]
    PALETTE = [(cid, cid, cid) for cid in range(max(IDS) + 1)]

    def __init__(self, data_dir=None, data_type='train2014'):
        """

        :param data_dir: base ms-coco path (parent of annotation json directory)
        :param data_type: 'train2014', 'val2014' or 'test2015'
        """
        Dataset.__init__(self)

        if data_dir is None:
            data_dir = os.path.join(os.path.expanduser('~'), '.keras', 'datasets', 'coco')
        self.download(data_dir=data_dir)
        valid_data_types = ['train2014', 'val2014', 'test2015']
        if data_type not in valid_data_types:
            raise ValueError('Unknown data type {}. Valid values are {}'.format(data_type, valid_data_types))

        annotation_file = '{}/annotations/instances_{}.json'.format(data_dir, data_type)
        print('Initializing MS-COCO: Loading annotations from {}'.format(annotation_file))
        self._coco = COCO(annotation_file)

        # self._data_dir = os.path.join(data_dir, data_type)
        self._data_dir = {'annotations': annotation_file,
                          'images': os.path.join(data_dir, data_type, 'images')}

        self._area_threshold = 2500

        # build indices
        self._cid_to_id = {cid: idx for idx, cid in enumerate(self.IDS)}
        self._category_to_id = {category: idx for idx, category in enumerate(self.CATEGORIES)}
        self._palette_to_id = {category: idx for idx, category in enumerate(self.CATEGORIES)}

        # pass through the dataset and count all valid samples
        sample_counter = 0
        for idx, img_id in enumerate(self._coco.getImgIds()):
            annotation_ids = self._coco.getAnnIds(imgIds=img_id)
            for annotation in self._coco.loadAnns(annotation_ids):
                if annotation['area'] > self._area_threshold:
                    sample_counter += 1
        self._num_samples = sample_counter

    @property
    def categories(self):
        return MSCOCO.CATEGORIES

    @property
    def palette(self):
        return MSCOCO.PALETTE

    @property
    def num_instances(self):
        return self._num_samples

    @property
    def num_images(self):
        return len(self._coco.imgs)

    @staticmethod
    def num_classes():
        return len(MSCOCO.IDS)

    @staticmethod
    def download(data_dir=None):
        """Download MSCOCO into data_dir, verify hashes, then extract files.
        If the files are already present, only the hashes are checked.
        """
        if data_dir is None:
            data_dir = os.path.join(os.path.expanduser('~'), '.keras', 'datasets', 'coco')

        # http://stackoverflow.com/questions/600268/mkdir-p-functionality-in-python
        try:
            os.makedirs(data_dir)
        except OSError as exc:  # Python >2.5
            if exc.errno == errno.EEXIST and os.path.isdir(data_dir):
                pass
            else:
                raise

        urls = [
            'http://msvocds.blob.core.windows.net/coco2014/train2014.zip',
            'http://msvocds.blob.core.windows.net/coco2014/val2014.zip',
            'http://msvocds.blob.core.windows.net/coco2014/test2014.zip',
            'http://msvocds.blob.core.windows.net/coco2015/test2015.zip',
            'http://msvocds.blob.core.windows.net/annotations-1-0-3/instances_train-val2014.zip',
            'http://msvocds.blob.core.windows.net/annotations-1-0-3/person_keypoints_trainval2014.zip',
            'http://msvocds.blob.core.windows.net/annotations-1-0-4/image_info_test2014.zip',
            'http://msvocds.blob.core.windows.net/annotations-1-0-4/image_info_test2015.zip',
            'http://msvocds.blob.core.windows.net/annotations-1-0-3/captions_train-val2014.zip'
        ]
        data_prefixes = [
            'train2014',
            'val2014',
            'test2014',
            'test2015',
        ]
        image_filenames = [prefix + '.zip' for prefix in data_prefixes]
        annotation_filenames = [
            'instances_train-val2014.zip',  # training AND validation info
            'image_info_test2014.zip',  # basic info like download links + category
            'image_info_test2015.zip',  # basic info like download links + category
            'person_keypoints_trainval2014.zip',  # elbows, head, wrist etc
            'captions_train-val2014.zip',  # descriptions of images
        ]
        md5s = [
            '0da8c0bd3d6becc4dcb32757491aca88',  # train2014.zip
            'a3d79f5ed8d289b7a7554ce06a5782b3',  # val2014.zip
            '04127eef689ceac55e3a572c2c92f264',  # test2014.zip
            '65562e58af7d695cc47356951578c041',  # test2015.zip
            '59582776b8dd745d649cd249ada5acf7',  # instances_train-val2014.zip
            '926b9df843c698817ee62e0e049e3753',  # person_keypoints_trainval2014.zip
            'f3366b66dc90d8ae0764806c95e43c86',  # image_info_test2014.zip
            '8a5ad1a903b7896df7f8b34833b61757',  # image_info_test2015.zip
            '5750999c8c964077e3c81581170be65b'   # captions_train-val2014.zip
        ]
        filenames = image_filenames + annotation_filenames

        for url, filename, md5 in zip(urls, filenames, md5s):
            path = get_file(filename, url, md5_hash=md5, extract=True, cache_subdir=data_dir)

    def load(self, data_dir, data_type):
        annotation_file = '{}/annotations/instances_{}.json'.format(data_dir, data_type)
        print('Initializing MS-COCO: Loading annotations from {}'.format(annotation_file))
        self._coco = COCO(annotation_file)

    def _annotation_generator(self, sample_size=None):
        """
        Generates sample_size annotations. No pre/post-processing
        :type sample_size: int
        :param sample_size: How many samples to retrieve before stopping
        :return: coco annotation (dictionary)
        """

        img_ids = self._coco.getImgIds()
        if isinstance(sample_size, numbers.Number):
            # TODO: random.sample might cause problems when target size is larger than image dimensions \
            # TODO: Examine shuffle + break as an alternative
            img_ids = np.random.choice(img_ids, size=int(sample_size))

        while True:
            for img_id in img_ids:
                annotation_ids = self._coco.getAnnIds(imgIds=img_id)
                for annotation in self._coco.loadAnns(annotation_ids):
                    if annotation['area'] > self._area_threshold:
                        yield annotation

    def _retrieve_sample(self, annotation):
        epsilon = 0.05
        high_val = 1 - epsilon
        low_val = 0 + epsilon
        coco_image = self._coco.loadImgs(annotation['image_id'])[0]
        image_path = os.path.join(self._data_dir['images'], coco_image['file_name'])
        image = utils.load_image(image_path)

        ann_mask = self._coco.annToMask(annotation)

        mask_categorical = np.full((ann_mask.shape[0], ann_mask.shape[1], self.num_classes()), low_val, dtype=np.float32)
        mask_categorical[:, :, 0] = high_val  # every pixel begins as background

        class_index = self._cid_to_id[annotation['category_id']]
        mask_categorical[ann_mask > 0, class_index] = high_val
        mask_categorical[ann_mask > 0, 0] = low_val  # remove background label from pixels of this (non-bg) category
        return image, mask_categorical

    def _retrieve_instance(self, annotation, keep_context=0.):
        """
        crops a pair of image label arrays according to an annotation id
        :type keep_context: float < 1
        :type annotation: dict
        """
        image, mask_categorical = self._retrieve_sample(annotation=annotation)

        # x, y, width, height = [int(elem + 0.5) for elem in annotation['bbox']]
        x, y, width, height = annotation['bbox']
        x = int(max(0., x - keep_context * width + 0.5))
        y = int(max(0., y - keep_context * height + 0.5))
        height = int(min(height + 2 * keep_context * height + 0.5, mask_categorical.shape[0]))
        width = int(min(width + 2 * keep_context * width + 0.5, mask_categorical.shape[1]))

        cropped_image = image[y: y + height, x: x + width, :]
        cropped_label = mask_categorical[y: y + height, x: x + width, :]
        return cropped_image, cropped_label

    def instance_generator(self, sample_size=None, keep_context=0.):
        """
        :type keep_context: float < 1
        :param sample_size:
        :param keep_context: How much context to keep around bbox (percentage).
        If possible, adds as many pixels as keep_context*height on both the top and the bottom and keep_context*width
        on the left and on the right
        """
        for ann in self._annotation_generator(sample_size=sample_size):
            cropped_image, cropped_label = self._retrieve_instance(ann, keep_context=keep_context)
            yield cropped_image, cropped_label

    def raw_sample_generator(self, sample_size=None):
        """
        Generates image/mask pairs from dataset.
        MS-COCO categories have some gaps among the ids. This function compresses them to eliminate these gaps.
        :type sample_size: int
        :param sample_size: How many samples to retrieve before stopping
        :return: generator of image-mask pairs.
        Each pair is a tuple. Image shape is (height, width, 3) and the shape of the mask is (height, width, 91)
        """
        for annotation in self._annotation_generator(sample_size=sample_size):
            image, mask_categorical = self._retrieve_sample(annotation=annotation)
            yield image, mask_categorical

    # def combined_sample_generator(self, cover_gaps=True, target_h=None, target_w=None, sample_size=None):
    def combined_sample_generator(self, cover_gaps=True, sample_size=None):
        """
        Generates image/mask pairs from dataset.
        :type sample_size: int
        :param cover_gaps: if True, the category ids are compressed so that gaps are eliminated (makes training easier)
        :param target_h:
        :param target_w:
        :param sample_size:
        :return: generator of image-mask pairs.
        Each pair is a tuple. Image shape is (width, height, 3) and the shape of the mask is (width, height, 91)
        """
        # default_target_h = target_h
        # default_target_w = target_w

        img_ids = self._coco.getImgIds()
        if isinstance(sample_size, numbers.Number):
            # TODO: random.sample might cause problems when target size is larger than image dimensions \
            # TODO: Examine shuffle + break as an alternative
            img_ids = [int(i) for i in np.random.choice(img_ids, size=sample_size)]

        epsilon = 0.05
        high_val = 1 - epsilon
        low_val = 0 + epsilon
        while True:
            for img_id in img_ids:
                coco_image = self._coco.loadImgs(img_id)[0]

                image_path = os.path.join(self._data_dir['images'], coco_image['file_name'])
                image = utils.load_image(image_path)

                dimensions = len(self.IDS) if cover_gaps else max(self.IDS) + 1
                # target_h = coco_image['height'] if not default_target_h else default_target_h
                # target_w = coco_image['width'] if not default_target_w else default_target_w
                target_h = coco_image['height']
                target_w = coco_image['width']

                # if target_h > coco_image['height'] or target_w > coco_image['width']:
                #     continue

                # image = utils.resize(image, target_h=target_h, target_w=target_w)

                mask_one_hot = np.full((target_h, target_w, self.num_classes()), low_val, dtype=np.float32)
                mask_one_hot[:, :, 0] = high_val  # every pixel begins as background

                annotation_ids = self._coco.getAnnIds(imgIds=coco_image['id'])

                for annotation in self._coco.loadAnns(annotation_ids):
                    mask_partial = self._coco.annToMask(annotation)
                    # mask_partial = utils.resize(mask_partial, target_h=target_h, target_w=target_w)
                    assert mask_one_hot.shape[:2] == mask_partial.shape[:2]  # width and height match
                    if cover_gaps:
                        class_index = self._cid_to_id[annotation['category_id']]
                    else:
                        class_index = annotation['category_id']
                    assert class_index > 0
                    mask_one_hot[mask_partial > 0, class_index] = high_val
                    mask_one_hot[mask_partial > 0, 0] = low_val  # remove bg label from pixels of this (non-bg) category
                yield image, mask_one_hot

    def sample_generator(self, sample_size=None, instance_mode=True, keep_context=0.):
        if instance_mode:
            return self.instance_generator(sample_size=sample_size, keep_context=keep_context)
        else:
            return self.combined_sample_generator(sample_size=sample_size)

    @staticmethod
    def mask_to_mscoco(alpha, annotations, img_id, mode='rle'):
        if mode == 'rle':
            in_ = np.reshape(np.asfortranarray(alpha), (alpha.shape[0], alpha.shape[1], 1))
            in_ = np.asfortranarray(in_)
            rle = mask.encode(in_)
            segmentation = rle[0]
        else:
            raise ValueError('Unknown mask mode "{}"'.format(mode))
        for idx, c in enumerate(np.unique(alpha)):
            area = mask.area(rle).tolist()
            if isinstance(area, list):
                area = area[0]
            bbox = mask.toBbox(rle).tolist()
            if isinstance(bbox[0], list):
                bbox = bbox[0]
            annotation = {
                'area': area,
                'bbox': bbox,
                'category_id': c,
                'id': len(annotations)+idx,
                'image_id': img_id,
                'iscrowd': 0,
                'segmentation': segmentation}
            annotations.append(annotation)
        return annotations


class MSCOCOReduced(MSCOCO):
    NAME = 'mscoco_reduced'

    CATEGORIES = [
        'background',  # class zero
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
        'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'baseball bat', 'bottle', 'wine glass',
        'cup', 'fork', 'knife', 'spoon', 'cake', 'chair', 'couch', 'bed', 'dining table', 'toilet',
        'tv', 'laptop', 'cell phone', 'refrigerator', 'book', 'clock', 'vase']
    IDS = [
         0,
         1,  2,  3,  4,  5,  6,  7,  8,  9, 10,
        11, 13, 14, 15, 16, 17, 18, 19, 20, 21,
        22, 23, 27, 28, 31, 32, 33, 39, 44, 46,
        47, 48, 49, 50, 61, 62, 63, 65, 67, 70,
        72, 73, 77, 82, 84, 85, 86]

    def __init__(self, data_dir, data_type):
        MSCOCO.__init__(self, data_dir=data_dir, data_type=data_type)

        # pass through the dataset and count all valid samples
        sample_counter = 0
        imgs_new = {}
        anns_new = {}
        # cats_new = {0: {'id': 0, 'name': 'background', 'supercategory': 'background'}}
        cats_new = {}
        for cat_id in self.IDS[1:]:
            cats_new[cat_id] = self._coco.cats[cat_id]
        for img_id in self._coco.getImgIds():
            annotation_ids = self._coco.getAnnIds(imgIds=img_id)
            for annotation in self._coco.loadAnns(annotation_ids):
                if annotation['category_id'] not in MSCOCOReduced.IDS:
                    continue
                if annotation['area'] > self._area_threshold:
                    imgs_new[img_id] = self._coco.imgs[img_id]
                    sample_counter += 1
                    anns_new[annotation['id']] = self._coco.anns[annotation['id']]
        self._coco.imgs = imgs_new
        self._coco.anns = anns_new
        self._coco.cats = cats_new
        self._num_samples = sample_counter

    # def _annotation_generator(self, sample_size=None):
    #     ann_generator = super(MSCOCOReduced, self)._annotation_generator(sample_size)
    #     for ann in ann_generator:
    #         if ann['category_id'] in MSCOCOReduced.IDS:
    #             yield ann
