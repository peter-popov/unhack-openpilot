#
# Copyright (c) 2016, 2018 Qualcomm Technologies, Inc.
# All Rights Reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#

#############################################################
# Alexnet setup
#############################################################

mkdir -p alexnet
mkdir -p alexnet/images

cd alexnet

cp -R ../../../../models/alexnet/data/cropped/*.jpg images
cp -R ../../../../models/alexnet/dlc/bvlc_alexnet.dlc model.dlc
cp -R ../../../../models/alexnet/data/ilsvrc_2012_labels.txt labels.txt
cp -R ../../../../models/alexnet/data/ilsvrc_2012_mean_cropped.bin mean_image.bin

zip -r alexnet.zip ./*
mkdir ../app/src/main/res/raw/
cp alexnet.zip ../app/src/main/res/raw/

cd ..
rm -rf ./alexnet
