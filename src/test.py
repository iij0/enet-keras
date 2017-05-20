# coding=utf-8
from __future__ import absolute_import
from __future__ import print_function

import json
import numpy as np
import os
import sys

from src.data import utils, datasets
from predict import predict
from keras import backend as K
from keras.preprocessing.image import array_to_img
from pycocotools import mask as cocomask
from PIL import Image as PILImage

import models


def masks_as_fortran_order(masks):
    # masks = masks.transpose((1, 2, 0))
    masks = masks.transpose((2, 0, 1))
    masks = np.asfortranarray(masks)
    masks = masks.astype(np.uint8)
    return masks


def ann_dict_generator(alpha, scores, img_id, filename=None):
    """
    split alpha into binary masks (one per class)
    then for each mask get score and save dictionary according to the format: http://mscoco.org/dataset/#format

    [{
        "image_id"    : int,
        "category_id" : int,
        "segmentation": RLE,
        "score"       : float,
    }]
    :param alpha: mask to be converted to annotation
    :param scores: score per class
    :param img_id: 
    :param filename: 
    """
    filename = '' if not filename else filename
    # copyfile(filename, '/tmp/{}'.format(os.path.basename(filename)))
    for idx, c in enumerate(np.unique(alpha)):
        mask = np.zeros(alpha.shape)
        mask[alpha == c] = 1
        mask = masks_as_fortran_order(np.asarray(mask))

        # outfile = '/tmp/{}_{}.jpg'.format(basename_without_ext(filename), str(c))
        # print('Saving mask to {}'.format(outfile))
        # cv2.imwrite(outfile, mask * 255)

        rle = cocomask.encode(mask)
        segmentation = rle[0]
        ann = {'image_id': img_id,
               'category_id': int(c),
               'filename': filename,
               'segmentation': segmentation,
               #'score': float(scores[c])}
               'score': 1.0}
        yield ann


def load_data(data_type, mscoco_dir):
    filetxt = '{}/{}/images.txt'.format(mscoco_dir, data_type)
    with open(filetxt) as fin:
        basedir = os.path.dirname(filetxt)
        files = [os.path.join(basedir, line.rstrip('\n')) for line in fin]
    return files


def load_model(h5file, model_name):
    model = models.select_model(model_name)
    # h5file = os.path.join(result_dir, 'mscoco', model_name, '{}.h5'.format(model_name, h5filename))
    segmenter, model_name = model.build(nc=nc, w=dw, h=dh)
    segmenter.load_weights(h5file)
    return segmenter


def build_detections(segmenter, files, target_h, target_w):
    results = []
    for idx, imfile in enumerate(files):
        if idx >= test_sample_size:
            break
        print('Processing {} out of {}'.format(idx + 1, test_sample_size), end='\r')
        sys.stdout.flush()
        try:
            img = utils.load_image(imfile).astype(np.uint8)
            pred, scores = predict(segmenter, img, h=target_h, w=target_w)
            bname = os.path.basename(imfile)

            img_id = int(imfile[-10:-4])
            results += [ann for ann in ann_dict_generator(pred, scores, img_id, imfile)]
            pred = pred[:, :, 0]
            pred[pred > 0.5] = 255
            PILImage.fromarray(pred, mode='L').save('/tmp/inference_results/{}.png'.format(bname))
        except:
            if img is None:
                print('Skipping corrupted image')
                continue
            else:
                raise  # Exception('corrupted image')
    return results


def save_to_json(dt, evaluation_dir, data_type, ann_type='segm'):
    print('Saving to file...')
    prefix = 'person_keypoints' if ann_type=='keypoints' else 'instances'
    if not os.path.exists(evaluation_dir):
        os.makedirs(evaluation_dir)
    with open(os.path.join(evaluation_dir, '{}_{}_{}_results.json'.format(prefix, data_type, ann_type)), "wb") as f:
        json.dump(dt, f, indent=4)
    print('Done!')

if __name__ == '__main__':
    if K.backend() == 'tensorflow':
        print('Tensorflow backend detected; Applying memory usage constraints')
        ss = K.tf.Session(config=K.tf.ConfigProto(gpu_options=K.tf.GPUOptions(allow_growth=True)))
        K.set_session(ss)
        ss.run(K.tf.global_variables_initializer())

    np.random.seed(1337)  # for reproducibility

    # eval_config_json = sys.argv[1]
    eval_config_json = 'config/evaluation.json'
    metadata = json.load(open(eval_config_json))

    dh = metadata['dh']
    dw = metadata['dw']
    nc = metadata['nc']
    model_name = metadata['model_name']
    test_sample_size = metadata['test_sample_size']
    # result_dir = metadata['result_dir']
    result_dir = 'models'

    h5file = os.path.join(result_dir, 'mscoco', model_name, 'weights', metadata['h5file'] + '.h5')
    print('{} has been selected'.format(h5file))

    model = load_model(h5file, model_name)
    
    imfiles = load_data(metadata['data_type'], mscoco_dir=os.path.join('data', 'mscoco'))
    # datasets.load(dataset_name='mscoco')
    dt = build_detections(model, imfiles, target_h=dh, target_w=dw)

    # prepare output and save to json
    evaluation_dir = os.path.join(result_dir, 'mscoco', model_name, 'results')
    save_to_json(dt, evaluation_dir, metadata['data_type'], metadata['ann_type'])
