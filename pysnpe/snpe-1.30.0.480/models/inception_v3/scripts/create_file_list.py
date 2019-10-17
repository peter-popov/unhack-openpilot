#
# Copyright (c) 2016 Qualcomm Technologies, Inc.
# All Rights Reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#

import argparse
import glob
import os

def create_file_list(input_dir, output_filename, ext_pattern, print_out=False, rel_path=False):
    input_dir = os.path.abspath(input_dir)
    output_filename = os.path.abspath(output_filename)
    output_dir = os.path.dirname(output_filename)

    if not os.path.isdir(input_dir):
        raise RuntimeError('input_dir %s is not a directory' % input_dir)

    if not os.path.isdir(output_dir):
        raise RuntimeError('output_filename %s directory does not exist' % output_dir)

    glob_path = os.path.join(input_dir, ext_pattern)
    file_list = glob.glob(glob_path)

    if rel_path:
        file_list = [os.path.relpath(file_path, output_dir) for file_path in file_list]

    if len(file_list) <= 0:
        if print_out: print('No results with %s' % glob_path)
    else:
        with open(output_filename, 'w') as f:
            f.write('\n'.join(file_list))
            if print_out: print('%s created listing %d files.' % (output_filename, len(file_list)))

def main():
    parser = argparse.ArgumentParser(description="Create file listing matched extension pattern.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i', '--input_dir',
                        help='Input directory to look up files.',
                        default='.')
    parser.add_argument('-o', '--output_filename',
                        help='Output filename - will overwrite existing file.',
                        default='file_list.txt')
    parser.add_argument('-e', '--ext_pattern',
                        help='Lookup extension pattern, e.g. *.jpg, *.raw',
                        default='*.jpg')
    parser.add_argument('-r', '--rel_path',
                        help='Listing to use relative path',
                        action='store_true')
    args = parser.parse_args()

    create_file_list(args.input_dir, args.output_filename, args.ext_pattern, print_out=True, rel_path=args.rel_path)

if __name__ == '__main__':
    main()
