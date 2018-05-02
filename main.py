# -*- coding: utf-8 -*-
import os
import keras
import time
import argparse
import DatasetRead as DR
import sys
sys.path.append('./network')
import tensorflow as tf
import importlib
from keras.optimizers import SGD

import tensorflow as tf
import keras.backend.tensorflow_backend as KTF

import functools

import scipy.io as sio


IMG_ROWS,IMG_COLS = 224,224
NUM_CLASSES = 20
EPOCH_EVERY_INDEX=10
BATCH_SIZE=16
LEARNING_RATE=1e-3
ISSCRATCH=True
ISLOAD=True



def train(dataset_dir,network_used,max_epoch=100, model_weight_dir=None):

    if not ISLOAD:
        datasetread=DR.DatasetList(dataset_dir,istraining=True)
        X_TRAIN,Y_TRAIN=datasetread.training_data()
        X_VAL,Y_VAL=datasetread.val_data()
    else:
        X_TRAIN=sio.loadmat('x_train.mat')['X_TRAIN']
        Y_TRAIN = sio.loadmat('y_train.mat')['Y_TRAIN']
        X_VAL = sio.loadmat('x_val.mat')['X_VAL']
        Y_VAL = sio.loadmat('y_val.mat')['Y_VAL']


    train_dataset_size=len(X_TRAIN)
    val_dataset_size=len(Y_VAL)

    tf.set_random_seed(DR.RANDOM_SEED)
    train_coord = tf.train.Coordinator()
    val_coord = tf.train.Coordinator()
    train_reader = DR.ImageReader(X_TRAIN,(IMG_ROWS,IMG_COLS),DR.IMG_MEAN,train_coord,istraining=True,aug_tag=True,label_list=Y_TRAIN)
    val_reader = DR.ImageReader(X_VAL, (IMG_ROWS,IMG_COLS),DR.IMG_MEAN, val_coord,istraining=True,aug_tag=True,label_list=Y_VAL)

    train_image_batch, train_label_batch = train_reader.dequeue(train_dataset_size)
    val_image_batch, val_label_batch = val_reader.dequeue(val_dataset_size)

    model = network_used.network(img_rows=IMG_ROWS, img_cols=IMG_COLS,color_type=3, num_classes=NUM_CLASSES,isscratch=ISSCRATCH)

    if not ISSCRATCH:
        if os.path.exists(model_weight_dir):
           model.load_weights(model_weight_dir, by_name=True)
        elif model_weight_dir !=None:
           print(model_weight_dir+'  is not correct')

    top3_acc = functools.partial(keras.metrics.top_k_categorical_accuracy, k=3)

    top3_acc.__name__ = 'top3_acc'


    # Todo:learning rate
    # Todo:loss fuction
    sgd = SGD(lr=LEARNING_RATE, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy', top3_acc])

    config=tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction=0.2
    config.gpu_options.allow_growth=True
    config.allow_soft_placement=True
    config.log_device_placement=False
    train_sess = tf.Session(config=config)
    val_sess = tf.Session(config=config)
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    train_sess.run(init_op)
    val_sess.run(init_op)
    train_threads = tf.train.start_queue_runners(coord=train_coord, sess=train_sess)
    val_threads = tf.train.start_queue_runners(coord=val_coord, sess=val_sess)

    for index in range(max_epoch):
        train_images, train_labels=train_sess.run([train_image_batch,train_label_batch])
        val_image, val_label = val_sess.run([val_image_batch, val_label_batch])
        print(str(index)+'/'+str(max_epoch)+' load done!')

        os.makedirs('./model/'+network.__name__,exist_ok=True)

        callbacks = [
            keras.callbacks.ModelCheckpoint('./model/'+network.__name__+'/'+'{:04d}'.format(index)+'.h5',
                                            verbose=0, save_weights_only=True,period=10)]
        model.fit(train_images, train_labels,
                  batch_size=BATCH_SIZE,
                  epochs=EPOCH_EVERY_INDEX,
                  shuffle=True,
                  verbose=1,
                  callbacks=callbacks,
                  validation_data=(val_image, val_label),
                  )
    train_coord.request_stop()
    train_coord.join(train_threads)
    val_coord.request_stop()
    val_coord.join(val_threads)


def test(dataset_dir,network_used, model_weight_dir,target_file):
    datasetread = DR.DatasetList(dataset_dir, istraining=False)
    X_TEST, Y_TEST = datasetread.test_data()

    Imagename=[os.path.split(index)[-1].split('.')[0] for index in X_TEST]

    test_dataset_size = len(X_TEST)

    tf.set_random_seed(DR.RANDOM_SEED)
    test_coord = tf.train.Coordinator()

    test_reader = DR.ImageReader(X_TEST, (IMG_ROWS, IMG_COLS), DR.IMG_MEAN, test_coord, istraining=False,
                                  aug_tag=False)

    test_image_batch = test_reader.dequeue(test_dataset_size)

    model = network_used.network(img_rows=IMG_ROWS, img_cols=IMG_COLS, color_type=3, num_classes=NUM_CLASSES)

    assert os.path.exists(model_weight_dir),model_weight_dir+' is not exist'
    model.load_weights(model_weight_dir)

    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.2
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    config.log_device_placement = False
    sess = tf.Session(config=config)
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init_op)
    test_threads = tf.train.start_queue_runners(coord=test_coord, sess=sess)
    test_images= sess.run([test_image_batch])
    print('load data done!')

    preds=model.predict(test_images,batch_size=BATCH_SIZE)
    top_indices_array=preds.argsort().T[-3:][::-1].T
    top_indices=top_indices_array.tolist()

    result=dict(zip(Imagename,top_indices))

    DR.writelabel(target_file, Y_TEST,result, header=['FILE_ID', 'CATEGORY_ID0', 'CATEGORY_ID1', 'CATEGORY_ID2'])


    print("Write results to file %s" % target_file)








if __name__ == '__main__':

    parser=argparse.ArgumentParser()
    parser.add_argument('--network', type=str, default='densenet161',
                        help='Network we used.')
    parser.add_argument('--mode', type=str, default='train',
                        help="Define the running mode as 'training' or 'test'.")
    parser.add_argument('--dataset_dir', type=str,default='/data1/dxj_data/image_scene_training_v1/image_scene_training',
                        help="Path to directory of training set or test set, depends on the running mode.")
    parser.add_argument('--model_weight_dir', type=str, default='./model/densenet121/00024.h5',
                        help="Path to directory of model.")
    parser.add_argument('--max_epoch', type=int, default=100,
                        help="Maximum training steps.")
    parser.add_argument('--target_file', type=str, default='./test_results.csv',
                        help='Path to test result file.')
    parser.add_argument('--gpu_num', type=str, default=str(0),
                        help='GPU_num we used.')

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_num

    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.9
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    config.log_device_placement = False
    session = tf.Session(config=config)
    KTF.set_session(session)

    network=importlib.import_module(args.network)



    if args.mode == 'train':
        train(args.dataset_dir, network,args.max_epoch, args.model_weight_dir)
    elif args.mode == 'test':
        test(args.dataset_dir,network, args.model_weight_dir, args.target_file)
    else:
        raise Exception('--mode can be train or test only')




