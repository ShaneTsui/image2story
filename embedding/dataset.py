"""
Dataset loading
"""
import numpy

path_to_data = '../data/'

def load_dataset(name, load_test=False, cap_only=False):
    """
    Load captions and image features
    """
    loc = path_to_data + name + '/'

    if load_test:
        # Captions
        test_caps = []
        with open(loc + 'coco_test_caps.txt', 'rb') as f:
            for line in f:
                test_caps.append(line.strip())
        # Image features
        test_ims = numpy.load(loc + 'coco_test_ims.npy')
        return (test_caps, test_ims)
    else:
        # Captions
        train_caps, dev_caps = [], []
        with open(loc + 'coco_train_caps.txt', 'rb') as f:
            for line in f:
                train_caps.append(line.strip())

        with open(loc + 'coco_dev_caps.txt', 'rb') as f:
            for line in f:
                dev_caps.append(line.strip())

        if cap_only:
            return train_caps, dev_caps
        else:
            # Image features
            train_ims = numpy.load(loc + 'coco_train_ims.npy')
            dev_ims = numpy.load(loc + 'coco_dev_ims.npy')

            return (train_caps, train_ims), (dev_caps, dev_ims)
