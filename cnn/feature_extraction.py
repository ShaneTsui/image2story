"""
Extract feature from original image using InceptionV3
"""

from keras.applications.vgg19 import VGG19
from keras.applications.inception_v3 import InceptionV3
from keras.models import Model
from keras import backend as K
from keras.preprocessing import image as kimage


import os
import json
import numpy as np
from collections import defaultdict

from cnn.coco_dataset import CocoDataset


def parse_annotation(filepath, year, type='train'):
    with open(os.path.join(filepath, f"captions_{type}{year}.json")) as f:
        annotation = json.load(f)
        id2imgname = {record['id']: record['file_name'] for record in annotation['images']}
        id2captions = defaultdict(list)
        for record in annotation['annotations']:
            id2captions[record['image_id']].append(record['caption'])
    return id2imgname, id2captions

def extract_feature(model, dataset, id2captions, type, year):
    all_feats = []
    with open(f"../data/coco/coco{year}_{type}_caps.txt", 'w') as f_caps:
        for (ids, images) in dataset:
            features = model.predict(images)

            for idx, id in enumerate(ids):
                for caption in id2captions[id]:
                    all_feats.append(features[idx])
                    f_caps.write(caption.replace("\n", "") + '\n')

        with open(f"../data/coco/coco{year}_{type}_ims.npy", 'wb') as f_feats:
            np.save(f_feats, all_feats)


def extract_feature_caption(year, dataset_path='D:\Dataset\Coco', batch_size=128):

    # Parse annotation file
    train_path = os.path.join(dataset_path, f'train{year}')
    val_path = os.path.join(dataset_path, f'val{year}')
    annotation_path = os.path.join(dataset_path, 'annotations')

    train_id2imgname, train_id2captions = parse_annotation(annotation_path, year, 'train')
    val_id2imgname, val_id2captions = parse_annotation(annotation_path, year, 'val')
    train_imgs, val_imgs = list(train_id2captions.keys()), list(val_id2captions.keys())

    train_dataset = CocoDataset(train_path, train_imgs, train_id2imgname, batch_size=batch_size)
    val_dataset = CocoDataset(val_path, val_imgs, val_id2imgname, batch_size=batch_size)

    # Build inception model
    # model = InceptionV3(weights='imagenet')
    model = VGG19(weights='imagenet')
    base_input = model.input
    hidden_layer = model.layers[-2].output
    hidden_model = Model(base_input, hidden_layer)
    print(model.summary())
    print(hidden_model.summary())
    # print()

    # extract feature
    extract_feature(model=hidden_model, dataset=train_dataset, id2captions=train_id2captions, type='train', year=year)
    extract_feature(model=hidden_model, dataset=val_dataset, id2captions=val_id2captions, type='val', year=year)

def extract_image_feature(image_path):
    # Build inception model
    # model = InceptionV3(weights='imagenet')
    model = VGG19(weights='imagenet')
    base_input = model.input
    hidden_layer = model.layers[-2].output
    hidden_model = Model(base_input, hidden_layer)

    img = kimage.load_img(image_path, target_size=(224, 224))
    img = kimage.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    features = hidden_model.predict(img)
    with open(f"{image_path}_feat.npy".replace(".jpg", ""), 'wb') as f:
        np.save(f, features)

if __name__ == '__main__':
    print("gpu: ", K.tensorflow_backend._get_available_gpus())
    extract_image_feature('../data/coco/gary_crop.jpg')
    # for year in [2014, 2017]:
    #     features = extract_feature_caption(year)

