#
# Copyright (c) 2016-2019 Qualcomm Technologies, Inc.
# All Rights Reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#

import argparse
import numpy as np
import os

from PIL import Image


def create_alexnet_mean_npy(imagenet_mean, output_dir):
    imagenet_mean_npy = np.load(imagenet_mean) # should be (3, 256, 256)
    # crop to 3, 227, 227 that is smaller - usable by caffe
    mean_npy = np.ndarray(shape=(3, 227, 227))

    if imagenet_mean_npy.shape[0] < mean_npy.shape[0] \
        or imagenet_mean_npy.shape[1] < mean_npy.shape[1] \
        or imagenet_mean_npy.shape[2] < mean_npy.shape[2]:
        raise RuntimeError('Bad mean shape {} for alexnet mean shape {}'.format(imagenet_mean_npy.shape, mean_npy.shape))
    # cut to size
    mean_npy = imagenet_mean_npy[:, :227, :227]
    # transpose to 227, 227, 3 for snpe mean subtraction
    return np.transpose(mean_npy, (1, 2, 0))


def preprocess_img_for_alexnet(img_filepath, output_dir):
    img = Image.open(img_filepath)
    # convert image into RGB format
    if img.mode != 'RGB':
        img = img.convert(mode='RGB')
    # center crop to square
    width, height = img.size
    short_dim = min(height, width)
    crop_coord = (
        (width - short_dim) / 2,
        (height - short_dim) / 2,
        (width + short_dim) / 2,
        (height + short_dim) / 2
    )
    img = img.crop(crop_coord)
    # resize to alexnet size
    img = img.resize((227, 227), Image.ANTIALIAS)
    # save output
    preprocessed_image_filepath = os.path.join(output_dir, os.path.basename(img_filepath))
    img.save(preprocessed_image_filepath)

    return preprocessed_image_filepath


def img_to_snpe_raw(img_filepath, raw_filepath, mean_npy):
    img = Image.open(img_filepath)
    img_array = np.array(img) # read it
    img_ndarray = np.reshape(img_array, (227, 227, 3)) # reshape to alexnet size
    # reverse last dimension: rgb -> bgr
    img_out = img_ndarray[..., ::-1]
    # mean subtract
    img_out = img_out - mean_npy
    img_out = img_out.astype(np.float32)
    # save
    fid = open(raw_filepath, 'wb')
    img_out.tofile(fid)


def main():
    parser = argparse.ArgumentParser(description='Convert jpg images to SNPE alexnet input.')
    parser.add_argument('-i', '--input_dir',
                        help='Input directory with jpg files.', required=True)
    parser.add_argument('-o', '--output_dir',
                        help='Output directory.', required=True)
    parser.add_argument('-m', '--imagenet_mean',
                        help='Path to ilsvrc_2012_mean.npy', required=True)
    args = parser.parse_args()

    input_dir = os.path.abspath(args.input_dir)
    output_dir = os.path.abspath(args.output_dir)
    imagenet_mean = os.path.abspath(args.imagenet_mean)

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    mean_npy = create_alexnet_mean_npy(imagenet_mean, output_dir)

    for root, dirs, files in os.walk(input_dir):
        if root == input_dir:
            for file in files:
                if file.endswith('.jpg'):
                    img_filepath = os.path.join(os.path.abspath(root), file)
                    print('processing %s' % img_filepath)
                    if not os.path.isfile(img_filepath):
                        raise RuntimeError('Cannot access %s' % img_filepath)
                    img_basename = os.path.splitext(file)[0]
                    raw_filepath = os.path.join(output_dir, img_basename + '.raw')

                    preprocessed_img_filepath = preprocess_img_for_alexnet(img_filepath, output_dir)
                    img_to_snpe_raw(preprocessed_img_filepath, raw_filepath, mean_npy)

if __name__ == '__main__':
    main()