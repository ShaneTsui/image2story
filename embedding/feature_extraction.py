"""
Extract feature from original image using InceptionV3
"""

from keras.applications.inception_v3 import InceptionV3
from keras.models import Model
from keras.layers import Dense
from keras.preprocessing import image
from keras import backend as K
import matplotlib.pyplot as plt
import numpy as np
import glob
from PIL import Image


def feature_extraction(image_path='../coco/val2017/'):

    img = glob.glob(image_path + '*.jpg')
    # Image.open(img[0])

    model = InceptionV3(weights='imagenet')
    base_input = model.input
    hidden_layer = model.layers[-2].output
    base_model = Model(base_input, hidden_layer)

    x = base_model.output
    preds = Dense(4096, activation='softmax')(x)  # final layer with softmax activation
    new_model = Model(inputs=base_model.input, outputs=preds)
    print(new_model.summary())

    # plt.imshow(np.squeeze(preprocess(img[0])))
    # tryi = new_model.predict(preprocess(img[0]))
    # print(tryi.type)

    features = new_model.predict(preprocess(img[0]))
    for image in img[1:]:
        features = np.concatenate((features, new_model.predict(preprocess(image))))

    print("feature extracted: ", features.shape)

    return features


def preprocess_input(x):
    x /= 255.
    return x


def preprocess(image_path):
    img = image.load_img(image_path, target_size=(299, 299))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x


if __name__ == '__main__':
    print("gpu: ", K.tensorflow_backend._get_available_gpus())
    features = feature_extraction()

    np.save('test.npy', features)
    test_load = np.load('test.npy')
    print(test_load.shape)


