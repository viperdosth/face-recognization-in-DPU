#!/usr/bin/env python3

import cv2
import os
import glob
from sklearn.utils import shuffle
import numpy as np


#resize picture and label them, with 0 stands for me, 1 stands for others
def load_train(train_path,image_size,classes):
    images = []
    labels = []
    img_names = []
    cls = []

    print('Going to read training images')
    for fields in classes:
        index = classes.index(fields)
        print('Now going to read {} file (Index: {})'.format(fields,index))
        path = os.path.join(train_path,fields,'*g')
        files = glob.glob(path)
        for f1 in files:
            image = cv2.imread(f1)
            image = cv2.resize(image,(image_size,image_size),0,0,cv2.INTER_LINEAR)
            image = image.astype(np.float32)
            image = np.multiply(image,1.0 / 255.0)
            images.append(image)
            label = np.zeros(len(classes))
            label[index] = 1.0
            labels.append(label)
            flbase = os.path.basename(f1)
            img_names.append(flbase)
            cls.append(fields)
    images = np.array(images)
    labels = np.array(labels)
    img_names = np.array(img_names)
    cls = np.array(cls)

    return images,labels,img_names,cls


#define parameter for the input train/test/validation set
class DataSet(object):
    def __init__(self,images,labels,img_names,cls):
        self._num_examples = 32 #images.shape[0]

        self._images = images
        self._labels = labels
        self._img_names = img_names
        self._cls = cls
        self._epochs_done = 0
        self._index_in_epoch = 0
    @property
    def images(self):
        return self._images
    @property
    def labels(self):
        return self._labels
    @property
    def img_names(self):
        return self._img_names
    @property
    def cls(self):
        return self._cls
    @property
    def num_examples(self):
        return self._num_examples
    @property
    def epochs_done(self):
        return self._epochs_done

    def next_batch(self,batch_size):
        start = self._index_in_epoch
        self._index_in_epoch += batch_size

        if self._index_in_epoch > self._num_examples:
            self._epochs_done += 1
            start = 0
            self._index_in_epoch = batch_size
            assert  batch_size <= self._num_examples
        end = self._index_in_epoch

        return self._images[start:end],self._labels[start:end],self._img_names[start:end],self.cls[start:end]



#define train and test/validation set as well as the label
def read_train_sets(train_path,image_size,classes,validation_size):
    class DataSets(object):
        pass
    data_sets = DataSets()

    images, labels, img_names, cls = load_train(train_path, image_size, classes)
    images, labels, img_names, cls = shuffle(images, labels, img_names, cls)

    if isinstance(validation_size, float):
        validation_size = int(validation_size * images.shape[0])

    validation_images = images[:validation_size]
    validation_labels = labels[:validation_size]
    validation_img_names = img_names[:validation_size]
    validation_cls = cls[:validation_size]

    train_images = images[validation_size:]
    train_labels = labels[validation_size:]
    train_img_names = img_names[validation_size:]
    train_cls = cls[validation_size:]

    data_sets.train = DataSet(train_images,train_labels,train_img_names,train_cls)
    data_sets.valid = DataSet(validation_images,validation_labels,validation_img_names,validation_cls)

    return data_sets









