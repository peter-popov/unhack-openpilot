#
# Copyright (c) 2017-2019 Qualcomm Technologies, Inc.
# All Rights Reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#

'''
Helper script to download artifacts to run inception_v3 model with SNPE SDK.
'''
import tensorflow as tf
import numpy as np
import os
import subprocess
import shutil
import hashlib
import argparse
import sys
import glob

INCEPTION_V3_ARCHIVE_CHECKSUM       = 'a904ddf15593d03c7dd786d552e22d73'
INCEPTION_V3_ARCHIVE_FILE           = 'inception_v3_2016_08_28_frozen.pb.tar.gz'
INCEPTION_V3_ARCHIVE_URL            = 'https://storage.googleapis.com/download.tensorflow.org/models/' + INCEPTION_V3_ARCHIVE_FILE
INCEPTION_V3_PB_FILENAME            = 'inception_v3_2016_08_28_frozen.pb'
INCEPTION_V3_PB_OPT_FILENAME        = 'inception_v3_2016_08_28_frozen_opt.pb'
INCEPTION_V3_DLC_FILENAME           = 'inception_v3.dlc'
INCEPTION_V3_DLC_QUANTIZED_FILENAME = 'inception_v3_quantized.dlc'
INCEPTION_V3_LBL_FILENAME           = 'imagenet_slim_labels.txt'
OPT_4_INFERENCE_SCRIPT              = 'optimize_for_inference.py'
RAW_LIST_FILE                       = 'raw_list.txt'
TARGET_RAW_LIST_FILE                = 'target_raw_list.txt'

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

def checkResource(inception_v3_data_dir, filename, md5):
    filepath = os.path.join(inception_v3_data_dir, filename)
    if not os.path.isfile(filepath):
        raise RuntimeError(filename + ' not found at the location specified by ' + inception_v3_data_dir + '. Re-run with download option.')
    else:
        checksum = generateMd5(filepath)
        if not checksum == md5:
            raise RuntimeError('Checksum of ' + filename + ' : ' + checksum + ' does not match checksum of file ' + md5)

def find_optimize_for_inference():
    tensorflow_root = os.path.abspath(os.environ['TENSORFLOW_HOME'])
    for root, dirs, files in os.walk(tensorflow_root):
        if OPT_4_INFERENCE_SCRIPT in files:
            return os.path.join(root, OPT_4_INFERENCE_SCRIPT)

def optimize_for_inference(model_dir, tensorflow_dir):
    # Try to optimize the inception v3 PB for inference
    opt_4_inference_file = find_optimize_for_inference()

    pb_filename = ""

    if not opt_4_inference_file:
        print("\nWARNING: cannot find " + OPT_4_INFERENCE_SCRIPT + " script. Skipping inference optimization.\n")
        pb_filename = INCEPTION_V3_PB_FILENAME
    else:
        print('INFO: Optimizing for inference Inception v3 using ' + opt_4_inference_file)
        print('      Please wait. It could take a while...')
        dlc_dir = os.path.join(model_dir, 'dlc')
        if not os.path.isdir(dlc_dir):
            os.makedirs(dlc_dir)
        cmd = ['python', opt_4_inference_file,
               '--input', os.path.join(tensorflow_dir, INCEPTION_V3_PB_FILENAME),
               '--output', os.path.join(tensorflow_dir, INCEPTION_V3_PB_OPT_FILENAME),
               '--input_names', 'input',
               '--output_names', 'InceptionV3/Predictions/Reshape_1']
        subprocess.call(cmd)
        pb_filename = INCEPTION_V3_PB_OPT_FILENAME

    return pb_filename

def prepare_data_images(snpe_root, model_dir, tensorflow_dir):
    # make a copy of the image files from the alexnet model data dir
    src_img_files = os.path.join(snpe_root, 'models', 'alexnet', 'data', '*.jpg')
    data_dir = os.path.join(model_dir, 'data')
    if not os.path.isdir(data_dir):
        os.makedirs(data_dir)
        os.makedirs(data_dir + '/cropped')
    for file in glob.glob(src_img_files):
        shutil.copy(file, data_dir)

    # copy the labels file to the data directory
    src_label_file = os.path.join(tensorflow_dir, INCEPTION_V3_LBL_FILENAME)
    shutil.copy(src_label_file, data_dir)

    print('INFO: Creating SNPE inception_v3 raw data')
    scripts_dir = os.path.join(model_dir, 'scripts')
    create_raws_script = os.path.join(scripts_dir, 'create_inceptionv3_raws.py')
    data_cropped_dir = os.path.join(data_dir, 'cropped')
    cmd = ['python', create_raws_script,
           '-i', data_dir,
           '-d',data_cropped_dir]
    subprocess.call(cmd)

    print('INFO: Creating image list data files')
    create_file_list_script = os.path.join(scripts_dir, 'create_file_list.py')
    cmd = ['python', create_file_list_script,
           '-i', data_cropped_dir,
           '-o', os.path.join(data_cropped_dir, RAW_LIST_FILE),
           '-e', '*.raw']
    subprocess.call(cmd)
    cmd = ['python', create_file_list_script,
           '-i', data_cropped_dir,
           '-o', os.path.join(data_dir, TARGET_RAW_LIST_FILE),
           '-e', '*.raw',
           '-r']
    subprocess.call(cmd)

def convert_to_dlc(pb_filename, model_dir, tensorflow_dir, runtime):
    print('INFO: Converting ' + pb_filename +' to SNPE DLC format')
    dlc_dir = os.path.join(model_dir, 'dlc')
    if not os.path.isdir(dlc_dir):
        os.makedirs(dlc_dir)
    cmd = ['snpe-tensorflow-to-dlc',
           '--graph', os.path.join(tensorflow_dir, pb_filename),
           '--input_dim', 'input', '1,299,299,3',
           '--out_node', 'InceptionV3/Predictions/Reshape_1',
           '--dlc', os.path.join(dlc_dir, INCEPTION_V3_DLC_FILENAME),
           '--allow_unconsumed_nodes']
    subprocess.call(cmd)

    # Further optimize the model with quantization for fixed-point runtimes if required.
    if ('dsp' == runtime or 'aip' == runtime):
        print('INFO: Creating ' + INCEPTION_V3_DLC_QUANTIZED_FILENAME + ' quantized model')
        data_cropped_dir = os.path.join(os.path.join(model_dir, 'data'), 'cropped')
        cmd = ['snpe-dlc-quantize',
               '--input_dlc', os.path.join(dlc_dir, INCEPTION_V3_DLC_FILENAME),
               '--input_list', os.path.join(data_cropped_dir, RAW_LIST_FILE),
               '--output_dlc', os.path.join(dlc_dir, INCEPTION_V3_DLC_QUANTIZED_FILENAME)]
        if ('aip' == runtime):
            print ('INFO: Compiling HTA metadata for AIP runtime.')
            # Enable compilation on the HTA after quantization
            cmd.append ('--enable_hta')
        subprocess.call(cmd)

def setup_assets(inception_v3_data_dir, download, runtime):

    if 'SNPE_ROOT' not in os.environ:
        raise RuntimeError('SNPE_ROOT not setup.  Please run the SDK env setup script.')

    snpe_root = os.path.abspath(os.environ['SNPE_ROOT'])
    if not os.path.isdir(snpe_root):
        raise RuntimeError('SNPE_ROOT (%s) is not a dir' % snpe_root)

    if None == runtime:
        # No runtime specified. Use cpu as default runtime
        runtime = 'cpu'
    runtimes_list = ['cpu', 'gpu', 'dsp', 'aip']
    if runtime not in runtimes_list:
        raise RuntimeError('%s not a valid runtime. See help.' % runtime)

    if download:
        url_path = INCEPTION_V3_ARCHIVE_URL;
        print("INFO: Downloading inception_v3 TensorFlow model...")
        wget(inception_v3_data_dir, url_path)

    try:
        checkResource(inception_v3_data_dir, INCEPTION_V3_ARCHIVE_FILE, INCEPTION_V3_ARCHIVE_CHECKSUM)
    except Exception as err:
        sys.stderr.write('ERROR: %s\n' % str(err))
        sys.exit(0)

    model_dir = os.path.join(snpe_root, 'models', 'inception_v3')
    if not os.path.isdir(model_dir):
        raise RuntimeError('%s does not exist.  Your SDK may be faulty.' % model_dir)

    print('INFO: Extracting TensorFlow model')
    tensorflow_dir = os.path.join(model_dir, 'tensorflow')
    if not os.path.isdir(tensorflow_dir):
        os.makedirs(tensorflow_dir)
    cmd = ['tar', '-xzf',  os.path.join(inception_v3_data_dir, INCEPTION_V3_ARCHIVE_FILE), '-C', tensorflow_dir]
    subprocess.call(cmd)

    pb_filename = optimize_for_inference(model_dir, tensorflow_dir)

    prepare_data_images(snpe_root, model_dir, tensorflow_dir)

    convert_to_dlc(pb_filename, model_dir, tensorflow_dir, runtime)

    print('INFO: Setup inception_v3 completed.')

def getArgs():

    parser = argparse.ArgumentParser(
        prog=__file__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=
        '''Prepares the inception_v3 assets for tutorial examples.''')

    parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')
    optional = parser.add_argument_group('optional arguments')

    required.add_argument('-a', '--assets_dir', type=str, required=True,
                        help='directory containing the inception_v3 assets')
    optional.add_argument('-d', '--download', action="store_true", required=False,
                        help='Download inception_v3 assets to inception_v3 example directory')
    optional.add_argument('-r', '--runtime', type=str, required=False,
                        help='Choose a runtime to set up tutorial for. Choices: cpu, gpu, dsp, aip')
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = getArgs()

    try:
        setup_assets(args.assets_dir, args.download, args.runtime)
    except Exception as err:
        sys.stderr.write('ERROR: %s\n' % str(err))
