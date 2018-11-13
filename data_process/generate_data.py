# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import cv2
import argparse
import os
import tqdm

if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    
    parse.add_argument('--images-dir', '-i', required=True, help='the images path')
    parse.add_argument('--label-file', '-l', required=True, help='the .txt label file of images')
    parse.add_argument('--out-dir', '-o', default='./out/', help='the output dir')
    parse.add_argument('--resize', '-r', type=int)
    
    args = parse.parse_args()
    
    # 创建输出目录
    os.makedirs(os.path.dirname(args.out_dir), exist_ok=True)
    
     # Scan all images
    all_images = [name for name in os.listdir(args.images_dir)]
    tqdm.write('Found total {} images in "{}".'.format(len(all_images), args.images_dir))
    
    # read label file
    names = ['image', 'label']
    cols = [0, 1]
    types = dict(zip(names, ['str', 'int']))    
    txt_df = pd.read_csv(args.label_file, sep='\t', names=names, usecols=cols, dtype=types)
    tqdm.write('Found {} labels in {}'.format(txt_df.shape[0], args.label_file))
    txt_df = txt_df[txt_df['image'].isin(all_images)]
    txt_df.reset_index(drop=True, inplace=True)
    
    #read images and save to .npy
    image_size = args.resize or None
    labels = []
    images = []
    for i, row in tqdm(txt_df.iterrows(), total=txt_df.shape[0], desc='reading'):
        path = os.path.join(args.images_dir, row['image'])
        image = cv2.imread(path)
        if image_size:
            image = cv2.resize(image, (image_size, image_size))
        labels.append(row['label'])
        images.append(image)
        
    np.save(args.out_dir+'images.npy', images)
    labels = pd.DataFrame(labels, columns=names)
	labels.to_csv(args.out_dir+'labels.csv', index=False)
    print('Done...')
