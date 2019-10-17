#
# Copyright (c) 2016-2018 Qualcomm Technologies, Inc.
# All Rights Reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#

'''
Helper script to download artifacts to run alexnet model with SNPE SDK.
'''
import caffe
import numpy as np
import os
import subprocess
import shutil
import hashlib
import argparse
import sys 

'''
Checksums of various assets required to be downloaded by user
'''
deploy_checksum = 'fdb525acaa8d3db0f87c5577901f53ee'
caffemodel_checksum = '29eb495b11613825c1900382f5286963'
ilsvrc12_checksum = 'f963098ea0e785a968ca1eb634003a90'

def wget(download_dir, file_url):
    cmd = ['wget', '-N', '--directory-prefix={}'.format(download_dir), file_url]
    subprocess.call(cmd)

def generateMd5(path):
    checksum = hashlib.md5()
    with open(path, 'rb') as data_file:
        while True:
            block = data_file.read(checksum.block_size)
            if not block:
                break
            checksum.update(block)
    return checksum.hexdigest()

def checkResource(alexnet_data_dir, filename, md5):
    filepath = os.path.join(alexnet_data_dir, filename)
    if not os.path.isfile(filepath):
        raise RuntimeError(filename + ' not found at the location specified by ' + alexnet_data_dir)
    else:
        checksum = generateMd5(filepath)
        if not checksum == md5:
            raise RuntimeError('Checksum of ' + filename + ' : ' + checksum + ' does not match checksum of file ' + md5)

def modifyPrototxt(directory, prototxt):
    deploy = open(directory+'/'+prototxt, 'r')
    batch1 = open(directory+'/'+prototxt.replace('.prototxt','_batch_1.prototxt'), 'w')
    for line in deploy:
        batch1.write(line.replace('dim: 10', 'dim: 1'))
    deploy.close()
    batch1.close()

def setup_assets(alexnet_data_dir, download):
    # print('Setup alexnet...')
    # print('You will need three caffe files for Alexnet. The file names and location to download them from are mentioned below')
    # print('deploy.prototxt - https://raw.githubusercontent.com/BVLC/caffe/master/models/bvlc_alexnet/deploy.prototxt')
    # print('bvlc_alexnet.caffemodel - http://dl.caffe.berkeleyvision.org/bvlc_alexnet.caffemodel')
    # print('caffe_ilsvrc12.tar.gz - http://dl.caffe.berkeleyvision.org/caffe_ilsvrc12.tar.gz')
    # print('Please point the environment variable $ALEXNET_DATA_DIR to a directory containing the following files.')

    if download:
        print("Downloading deploy.prototxt...")
        wget(alexnet_data_dir, 'https://raw.githubusercontent.com/BVLC/caffe/master/models/bvlc_alexnet/deploy.prototxt')
        print("Downloading dbvlc_alexnet.caffemodel...")
        wget(alexnet_data_dir, 'http://dl.caffe.berkeleyvision.org/bvlc_alexnet.caffemodel')
        print("Downloading caffe_ilsvrc12.tar.gz...")
        wget(alexnet_data_dir, 'http://dl.caffe.berkeleyvision.org/caffe_ilsvrc12.tar.gz')

    try:
        checkResource(alexnet_data_dir, 'deploy.prototxt', deploy_checksum)
        checkResource(alexnet_data_dir, 'bvlc_alexnet.caffemodel', caffemodel_checksum)
        checkResource(alexnet_data_dir, 'caffe_ilsvrc12.tar.gz', ilsvrc12_checksum)
    except Exception as err:
        sys.stderr.write('ERROR: %s\n' % str(err))
        sys.exit(0)

    if 'SNPE_ROOT' not in os.environ:
        raise RuntimeError('SNPE_ROOT not setup.  Please run the SDK env setup script.')
    snpe_root = os.path.abspath(os.environ['SNPE_ROOT'])
    if not os.path.isdir(snpe_root):
        raise RuntimeError('SNPE_ROOT (%s) is not a dir' % snpe_root)
    # this changes with SDK bom
    model_dir = os.path.join(snpe_root, 'models', 'alexnet')
    if not os.path.isdir(model_dir):
        raise RuntimeError('%s does not exist.  Your SDK may be faulty.' % model_dir)

    print('Copying Caffe model')
    caffe_dir = os.path.join(model_dir, 'caffe')
    if not os.path.isdir(caffe_dir):
        os.makedirs(caffe_dir)
    shutil.copy(os.path.join(alexnet_data_dir, 'deploy.prototxt'), caffe_dir)
    shutil.copy(os.path.join(alexnet_data_dir, 'bvlc_alexnet.caffemodel'), caffe_dir)

    print ("Modiying prototxt to use a batch size of 1")
    modifyPrototxt(caffe_dir, 'deploy.prototxt')

    print('Creating DLC')
    dlc_dir = os.path.join(model_dir, 'dlc')
    if not os.path.isdir(dlc_dir):
        os.makedirs(dlc_dir)
    cmd = ['snpe-caffe-to-dlc',
           '--caffe_txt', os.path.join(caffe_dir, 'deploy_batch_1.prototxt'),
           '--caffe_bin', os.path.join(caffe_dir, 'bvlc_alexnet.caffemodel'),
           '--output_path', os.path.join(dlc_dir, 'bvlc_alexnet.dlc')]
    subprocess.call(cmd)

    print('Getting imagenet aux data')
    data_dir = os.path.join(model_dir, 'data')
    if not os.path.isdir(data_dir):
        os.makedirs(data_dir)
    shutil.copy(os.path.join(alexnet_data_dir, 'caffe_ilsvrc12.tar.gz'), data_dir)
    cmd = ['tar', '-C', data_dir, '-xf', os.path.join(data_dir, 'caffe_ilsvrc12.tar.gz')]
    subprocess.call(cmd)

    mean_npy = 'ilsvrc_2012_mean.npy'
    print('Creating %s' % mean_npy)
    imagenet_mean = os.path.join(data_dir, 'imagenet_mean.binaryproto')
    if not os.path.isfile(imagenet_mean):
        raise RuntimeError('Missing %s' % imagenet_mean)
    blob = caffe.proto.caffe_pb2.BlobProto()
    mean_npy = os.path.join(data_dir, mean_npy)
    with open(imagenet_mean, 'rb') as f:
        blob.ParseFromString(f.read())
        mean_arr = np.array(caffe.io.blobproto_to_array(blob)) # should be (1, 3, 256, 256)
        np.save(mean_npy, mean_arr[0]) # save (3, 256, 256)

        imagenet_mean_cropped = os.path.join(data_dir, 'ilsvrc_2012_mean_cropped.bin')
        print('Creating %s', os.path.basename(imagenet_mean_cropped))
        mean_array_cropped = mean_arr[..., :227, :227]
        mean_array_cropped = np.transpose(mean_array_cropped, (0, 2, 3, 1))
        mean_array_cropped = mean_array_cropped.astype(np.float32)
        mean_array_cropped.tofile(imagenet_mean_cropped)

    labels_txt = 'ilsvrc_2012_labels.txt'
    print('Creating %s' % labels_txt)
    with open(os.path.join(data_dir, 'synset_words.txt'), 'r') as f:
        # get rid of first part which is categories' id
        labels_with_id = [' '.join(line.strip().split()[1:]) for line in f.readlines()]
        with open(os.path.join(data_dir, labels_txt), 'w') as labels_file:
            labels_file.write('\n'.join(labels_with_id))

    print('Create SNPE alexnet input')
    scripts_dir = os.path.join(model_dir, 'scripts')
    create_alexnet_raws_script = os.path.join(scripts_dir, 'create_alexnet_raws.py')
    data_cropped_dir = os.path.join(data_dir, 'cropped')
    cmd = ['python', create_alexnet_raws_script,
           '-i', data_dir,
           '-o', data_cropped_dir,
           '-m', mean_npy]
    subprocess.call(cmd)

    print('Create file lists')
    create_file_list_script = os.path.join(scripts_dir, 'create_file_list.py')
    cmd = ['python', create_file_list_script,
           '-i', data_cropped_dir,
           '-o', os.path.join(data_cropped_dir, 'raw_list.txt'),
           '-e', '*.raw']
    subprocess.call(cmd)
    cmd = ['python', create_file_list_script,
           '-i', data_cropped_dir,
           '-o', os.path.join(data_dir, 'target_raw_list.txt'),
           '-e', '*.raw',
           '-r']
    subprocess.call(cmd)

    print('Setup alexnet completed.')

def getArgs():

    parser = argparse.ArgumentParser(
        prog=__file__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=
        '''Prepares the AlexNet assets for tutorial examples.''')
    
    parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')
    optional = parser.add_argument_group('optional arguments')

    required.add_argument('-a', '--assets_dir', type=str, required=True,
                        help='directory containing the AlexNet assets')
    optional.add_argument('-d', '--download', action="store_true", required=False,
                        help='Download AlexNet assets to AlexNet assets directory')
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = getArgs()

    try:
        setup_assets(args.assets_dir, args.download)
    except Exception as err:
        sys.stderr.write('ERROR: %s\n' % str(err))
