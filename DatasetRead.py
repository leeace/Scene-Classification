# -*- coding: utf-8 -*-
import os
import numpy as np
import tensorflow as tf
import csv
import glob

import scipy.io as sio



IMG_MEAN = np.array((103.939,116.779,123.68), dtype=np.float32)
BATCH_SIZE = 32
INPUT_SIZE=[224,224]
RANDOM_SEED = 1234
CLASS_NUM=20

class DatasetList(object):

    def __init__(self,path,istraining=False,rate=0.7):

        self.imagepath=os.path.join(path,'data')
        self.istraining = istraining
        self.datatag = os.listdir(self.imagepath)
        self.datatag.sort()
        self.imagename_list = glob.glob(os.path.join(path, 'data', '*.jpg'))
        self.labelpath = os.path.join(path, 'list.csv')
        if istraining:
            self.imagename_list.sort()
            self.label=readlabel(self.labelpath,self.istraining)
            assert len(self.datatag)==len(self.label),'training dataset is not correct'
            train_inices, val_inices=splitdataset(self.datatag,rate)
            ###training
            self.traing_data_dir=np.array(self.imagename_list)[train_inices]
            self.traing_label=np.array(self.label)[train_inices]
            ###validation
            self.val_data_dir = np.array(self.imagename_list)[val_inices]
            self.val_label = np.array(self.label)[val_inices]
        else:
            self.test_data_dir=self.imagename_list
            self.label = readlabel(self.labelpath, self.istraining)

    def training_data(self):
        return self.traing_data_dir.tolist(),self.traing_label.tolist()
    def val_data(self):
        return self.val_data_dir.tolist(), self.val_label.tolist()
    def test_data(self):
        return self.test_data_dir,self.label


class ImageReader(object):

    def __init__(self,image_list,input_size,img_mean,coord,istraining=False,aug_tag=False,label_list=None):
        self.input_size = input_size
        self.coord = coord
        self.istraining=istraining
        self.image_list = image_list
        self.images = tf.convert_to_tensor(self.image_list, dtype=tf.string)
        if self.istraining:
            self.label_list=label_list
            self.labels = tf.convert_to_tensor(self.label_list, dtype=tf.int32)
            self.queue = tf.train.slice_input_producer([self.images, self.labels],shuffle=False)
            self.image, self.label = read_images_from_disk(self.queue, self.input_size, aug_tag, img_mean,self.istraining)
        else:
            self.queue = tf.train.slice_input_producer([self.images],shuffle=False)
            self.image= read_images_from_disk(self.queue, self.input_size, aug_tag, img_mean,self.istraining)


    def dequeue(self, num_elements):
        if self.istraining:
            image_batch, label_batch = tf.train.batch([self.image, self.label],num_elements)
            return image_batch, label_batch
        else:
            image_batch = tf.train.batch([self.image], num_elements)
            return  image_batch


def read_images_from_disk(input_queue, input_size, aug_flag,img_mean,istraining=False):
    # optional pre-processing arguments
    """Read one image and its corresponding label.

    Args:
      input_queue: tf queue with path of the image and its label.
      input_size: a tuple with (height, width) values.
      random_scale: whether to randomly scale the images prior
                    to random crop.
      random_mirror: whether to randomly mirror the images prior
                    to random crop.
      img_mean: vector of mean colour values."""

    channel_number=tf.constant(3)
    img_contents = tf.read_file(input_queue[0])
    img = tf.image.decode_jpeg(img_contents)
    input_image=tf.cond(tf.shape(img)[2]>=channel_number,lambda:img,lambda:tf.concat(axis=2,values=[img,img,img]))
    img_r, img_g, img_b = tf.split(axis=2, num_or_size_splits=3, value=input_image)
    img = tf.cast(tf.concat(axis=2, values=[img_b, img_g, img_r]), dtype=tf.float32)
    # Extract mean.

    ###Todo:data augmentation
    if aug_flag:
        img= tf.image.random_flip_left_right(img)
        size=tf.cast(tf.multiply(tf.cast(tf.shape(img),dtype=tf.float32),tf.constant([0.7,0.7,1])),dtype=tf.int32)
        img=tf.random_crop(img,size)
        img=tf.image.random_contrast(img,lower=0.3,upper=1.0)
        img=tf.image.random_brightness(img,max_delta=0.3)
        img=tf.image.random_saturation(img,lower=0.0,upper=2.0)
    img -= img_mean
    img = tf.image.resize_images(img, input_size, method=0)
    img.set_shape((input_size[0], input_size[1], 3))

    if istraining:
        label = input_queue[1]
        label = tf.one_hot(label, CLASS_NUM, on_value=1, off_value=None, axis=0)
        return img, label
    else:
        return img

def readlabel(path,istraining):
    with open( path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        rows = [row for row in reader]
        labeldata=rows[1:]
        if istraining:
            labeldata.sort()
            label=[int(index[1]) for index in labeldata]
        else:
            label=[index[0] for index in labeldata]
    return label


def splitdataset(tag,rate):
    indices=list(range(len(tag)))
    np.random.shuffle(indices)
    train_inices=indices[:int(len(tag)*rate)]
    val_inices = indices[int(len(tag) *rate):]
    return train_inices,val_inices


def writelabel(target_file,imagefile,data,header=[]):
    with open(target_file, 'w') as fh:
        csv_writer = csv.writer(fh)
        if (len(header)):
            csv_writer.writerow(header)
        for row in imagefile:
            csv_writer.writerow([row,data[row][0],data[row][1],data[row][2]])


if __name__=='__main__':
    dataset_dir='/data1/dxj_data/image_scene_training_v1/image_scene_training'
    datasetread = DatasetList(dataset_dir, istraining=True)
    X_TRAIN, Y_TRAIN = datasetread.training_data()
    X_VAL, Y_VAL = datasetread.val_data()
    sio.savemat('x_train.mat',{'X_TRAIN':X_TRAIN})
    sio.savemat('y_train.mat', {'Y_TRAIN': Y_TRAIN})
    sio.savemat('x_val.mat',{'X_VAL':X_VAL})
    sio.savemat('y_val.mat', {'Y_VAL': Y_VAL})



















