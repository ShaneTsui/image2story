from torch.utils.data import Dataset
import numpy as np
import os
from keras.preprocessing import image

class CocoDataset:

    def __init__(self, image_dir, image_ids, id2imgname, batch_size=128):
        self.image_dir = image_dir
        self.id2imgname = id2imgname
        self.image_ids = image_ids
        self.cur = 0
        self.batch_size = batch_size

    def _preprocess_input(self, x):
        x /= 255.
        return x

    def _get_image(self, image_path):
        try:
            img = image.load_img(image_path, target_size=(299, 299))
            x = image.img_to_array(img)
            # x = np.expand_dims(x, axis=0)
            # x = self._preprocess_input(x)
            return x
        except:
            raise

    def __iter__(self):
        return self

    def __next__(self):
        if self.cur >= len(self.image_ids):
            raise StopIteration
        ids, images = [], []
        for id in self.image_ids[self.cur : self.cur + self.batch_size]:
            try:
                images.append(self._get_image(os.path.join(self.image_dir, self.id2imgname[id])))
                ids.append(id)
            except Exception as ex:
                print(ex)
        self.cur += self.batch_size
        return ids, np.array(images)