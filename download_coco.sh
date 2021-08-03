#!/bin/bash

download_dir='/data'

wget -P ${download_dir} http://images.cocodataset.org/annotations/annotations_trainval2017.zip
wget -P ${download_dir} http://images.cocodataset.org/zips/train2017.zip
wget -P ${download_dir} http://images.cocodataset.org/zips/val2017.zip

unzip ${download_dir}/annotations_trainval2017.zip -d ${download_dir}
unzip ${download_dir}/train2017.zip -d ${download_dir}
unzip ${download_dir}/val2017.zip -d ${download_dir}

rm ${download_dir}/annotations_trainval2017.zip
rm ${download_dir}/train2017.zip
rm ${download_dir}/val2017.zip

python3 coco.py --datadir=${download_dir}