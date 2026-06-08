#! /usr/bin/env python3
import argparse
import os

import numpy as np
import torch
from tqdm import tqdm

from . import overlaps
from .utils.base_model import dynamic_load
from .utils.utils import (read_image, read_overlap_image, resize_pad_images,
                          visualize_overlap, visualize_overlap_crop)
"""
A set of standard configurations that can be directly selected from the command
line using their name. Each is a dictionary with the following entries:
    - output: the name of the feature file that will be generated.
    - model: the model configuration, as passed to a feature extractor.
    - preprocessing: how to preprocess the images read from disk.
"""
confs = {
    'oetr': {
        'output': 'oetr',
        'model': {
            'name': 'oetr',
            'num_layers': 50,
            'weights': 'oetr/oetr.pth',
        },
        'preprocessing': {
            'resize_mode': 'square',
        },
    },
    'oetr_fullimg': {
        'output': 'oetr_fullimg',
        'model': {
            'name': 'oetr',
            'num_layers': 50,
            'weights': 'oetr/oetr.pth',
        },
        'preprocessing': {
            'resize_mode': 'pad',
        },
    },
}


def process(
    config,
    pair,
    matching,
    with_desc,
    scales0,
    inp0,
    overlap_inp0,
    overlap_scales0,
    scales1,
    inp1,
    overlap_inp1,
    overlap_scales1,
    overlap_mask0=None,
    overlap_mask1=None,
    dataset_name='googleurban',
    overlap=True,
    warp_origin=True,
):
    """main process of match pipeline with overlap estimation.

    Args:
        config (Dict): configuration of extractor, matcher
        pair (list): data info
        matching (model): matching model
        with_desc (bool, optional): output descriptor. Defaults to False.
        scales0 (_type_): _description_
        inp0 (_type_): _description_
        overlap_inp0 (_type_): _description_
        overlap_scales0 (_type_): _description_
        scales1 (_type_): _description_
        inp1 (_type_): _description_
        overlap_inp1 (_type_): _description_
        overlap_scales1 (_type_): _description_
        overlap (bool, optional): _description_. Defaults to True.

    Returns:
        _type_: _description_
    """
    # Perform the matching.
    if config['landmark']:
        landmark = np.array(pair[2:], dtype=float).reshape(-1, 2)
        landmark_len = int(landmark.shape[0] / 2)
        template_kpts = landmark[:landmark_len] / scales0
        pred = matching(
            {
                'image0': inp0,
                'image1': inp1,
                'landmark': template_kpts,
                'overlap_image0': overlap_inp0,
                'overlap_image1': overlap_inp1,
                'overlap_scales0': overlap_scales0,
                'overlap_scales1': overlap_scales1,
                'overlap_mask0': overlap_mask0,
                'overlap_mask1': overlap_mask1,
                'dataset_name': dataset_name,
            },
            overlap,
        )
    else:
        pred = matching(
            {
                'image0': inp0,
                'image1': inp1,
                'overlap_image0': overlap_inp0,
                'overlap_image1': overlap_inp1,
                'overlap_scales0': overlap_scales0,
                'overlap_scales1': overlap_scales1,
                'overlap_mask0': overlap_mask0,
                'overlap_mask1': overlap_mask1,
                'dataset_name': dataset_name,
            },
            overlap,
        )

    if 'ratio0' in pred:
        ratio0 = pred['ratio0']
        ratio1 = pred['ratio1']
    pred = dict((k, v[0].cpu().numpy()) for k, v in pred.items())
    # pred = dict((k, v[0].cpu().numpy()) for k, v in pred.items() if "ratio" not in k)
    if 'ratio0' in pred and warp_origin:
        kpts0 = (pred['keypoints0'] / ratio0.cpu().numpy() +
                 pred['bbox0'][:2]) * scales0
        kpts1 = (pred['keypoints1'] / ratio1.cpu().numpy() +
                 pred['bbox1'][:2]) * scales1
    else:
        kpts0 = pred['keypoints0']  # * scales0
        kpts1 = pred['keypoints1']  # * scales1

    matches, conf = pred['matches0'], pred['matching_scores0']
    if with_desc:
        desc0, desc1 = pred['descriptors0'], pred['descriptors1']
    else:
        desc0, desc1 = None, None

    valid = matches > -1
    index0 = np.nonzero(valid)[0]
    index1 = matches[valid]

    return (
        kpts0,
        desc0,
        index0,
        kpts1,
        desc1,
        index1,
        conf,
        valid,
        pred['bbox0'][:2],
        pred['bbox1'][:2],
        pred['ratio0'],
        pred['ratio1'],
    )


def _load_overlap_inputs(
    input_dir,
    name,
    device,
    resize,
    resize_float,
    gray,
    align,
    resize_mode,
):
    path = os.path.join(input_dir, name)
    if resize_mode == 'pad':
        image, overlap_inp, inp, overlap_scales, overlap_mask = resize_pad_images(
            path,
            device,
            resize,
            0,
            resize_float,
            gray,
            size_divisor=32,
            overlap=True,
        )
        scales = (1.0, 1.0)
        return image, overlap_inp, inp, scales, overlap_scales, overlap_mask

    image, overlap_inp, inp, scales, overlap_scales = read_overlap_image(
        path,
        device,
        resize,
        0,
        resize_float,
        gray,
        align,
        True,
    )
    return image, overlap_inp, inp, scales, overlap_scales, None


def preprocess_overlap_pipeline(
    input,
    name0,
    name1,
    device,
    resize,
    resize_float,
    gray,
    align,
    config,
    pair,
    matching,
    with_desc=False,
    warp_origin=True,
):
    resize_mode = config.get('overlaper', {}).get(
        'preprocessing', {},
    ).get('resize_mode', 'square')
    (
        image0,
        overlap_inp0,
        inp0,
        scales0,
        overlap_scales0,
        overlap_mask0,
    ) = _load_overlap_inputs(
        input,
        name0,
        device,
        resize,
        resize_float,
        gray,
        align,
        resize_mode,
    )
    (
        image1,
        overlap_inp1,
        inp1,
        scales1,
        overlap_scales1,
        overlap_mask1,
    ) = _load_overlap_inputs(
        input,
        name1,
        device,
        resize,
        resize_float,
        gray,
        align,
        resize_mode,
    )
    if image0 is None or image1 is None:
        raise ValueError('Problem reading image pair: {}/{} {}/{}'.format(
            input, name0, input, name1))
    # Get dataset name
    dataset_name = name1.split('/')[0]

    (
        kpts0,
        desc0,
        index0,
        kpts1,
        desc1,
        index1,
        conf,
        valid,
        oxy0,
        oxy1,
        ratio0,
        ratio1,
    ) = process(
        config,
        pair,
        matching,
        with_desc,
        scales0,
        inp0,
        overlap_inp0,
        overlap_scales0,
        scales1,
        inp1,
        overlap_inp1,
        overlap_scales1,
        overlap_mask0,
        overlap_mask1,
        dataset_name,
        warp_origin=warp_origin,
    )

    if index0.shape[0] < 30:
        (
            kpts0,
            desc0,
            index0,
            kpts1,
            desc1,
            index1,
            conf,
            valid,
            oxy0,
            oxy1,
            ratio0,
            ratio1,
        ) = process(
            config,
            pair,
            matching,
            with_desc,
            scales0,
            inp0,
            overlap_inp0,
            overlap_scales0,
            scales1,
            inp1,
            overlap_inp1,
            overlap_scales1,
            overlap_mask0,
            overlap_mask1,
            dataset_name,
            False,
        )
    return {
        'image0': image0,
        'image1': image1,
        'kpts0': kpts0,
        'kpts1': kpts1,
        'desc0': desc0,
        'desc1': desc1,
        'index0': index0,
        'index1': index1,
        'mconf': conf[valid],
        'scales0': scales0,
        'scales1': scales1,
        'oxy0': oxy0,
        'oxy1': oxy1,
        'ratio0': ratio0,
        'ratio1': ratio1,
    }


@torch.no_grad()
def main(conf, opt):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    Model = dynamic_load(overlaps, conf['model']['name'])
    model = Model(conf['model']).eval().to(device)

    with open(opt.input_pairs, 'r') as f:
        pairs = [line.split() for line in f.readlines()]

    for i, pair in tqdm(enumerate(pairs), total=len(pairs)):
        if i % 10 != 0:
            continue
        name1, name2 = pair[:2]
        # Load the image pair.
        resize_mode = conf.get('preprocessing', {}).get('resize_mode', 'square')
        if resize_mode == 'pad':
            image1, inp1, _, _, mask1 = resize_pad_images(
                os.path.join(opt.input_dir, name1),
                device,
                opt.resize,
                0,
                opt.resize_float,
                size_divisor=32,
                overlap=True,
            )
            image2, inp2, _, _, mask2 = resize_pad_images(
                os.path.join(opt.input_dir, name2),
                device,
                opt.resize,
                0,
                opt.resize_float,
                size_divisor=32,
                overlap=True,
            )
            box1, box2 = model({
                'image0': inp1,
                'image1': inp2,
                'mask0': mask1,
                'mask1': mask2,
            })
        else:
            image1, inp1, _ = read_image(
                os.path.join(opt.input_dir, name1),
                device,
                opt.resize,
                0,
                opt.resize_float,
                overlap=True,
            )
            image2, inp2, _ = read_image(
                os.path.join(opt.input_dir, name2),
                device,
                opt.resize,
                0,
                opt.resize_float,
                overlap=True,
            )
            box1, box2 = model({'image0': inp1, 'image1': inp2})
        name1 = name1.split('/')[-1]
        name2 = name2.split('/')[-1]
        output = os.path.join(opt.output_dir, name1 + '-' + name2)
        np_box1 = box1[0].cpu().numpy().astype(int)
        np_box2 = box2[0].cpu().numpy().astype(int)
        crop_output = os.path.join(opt.output_dir,
                                   'crop_' + name1 + '-' + name2)
        visualize_overlap_crop(image1, np_box1, image2, np_box2, crop_output)
        # if len(pair) > 2:
        #     gt_box1 = np.array(pair[2:6]).astype(int)
        #     gt_box2 = np.array(pair[6:10]).astype(int)
        #     visualize_overlap_gt(image1, np_box1, gt_box1,
        #                          image2, np_box2, gt_box2, output)
        # else:
        visualize_overlap(image1, np_box1, image2, np_box2, output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input_pairs',
        type=str,
        default='assets/megadepth/pairs.txt',
        help='Path to the list of image pairs',
    )
    parser.add_argument(
        '--input_dir',
        type=str,
        default='assets/megadepth/',
        help='Path to the directory that contains the images',
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='outputs/',
        help='Path to the directory that contains the images',
    )
    parser.add_argument(
        '--resize',
        type=int,
        nargs='+',
        default=[640, 480],
        help='Resize the input image before running inference. If two numbers,'
        ' resize to the exact dimensions, if one number, resize the max '
        'dimension, if -1, do not resize',
    )
    parser.add_argument(
        '--resize_float',
        action='store_true',
        help='Resize the image after casting uint8 to float',
    )

    parser.add_argument('--conf',
                        type=str,
                        default='sadnet',
                        choices=list(confs.keys()))
    args = parser.parse_args()
    main(confs[args.conf], args)
