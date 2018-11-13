# -*- coding: utf-8 -*-
# from keras.utils.vis_utils import plot_model
from keras.utils import to_categorical
from keras.callbacks import *
from keras.optimizers import *
from keras.layers import *
from keras.models import *
import numpy as np
import pandas as pd
import os
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
import pickle
from metrics import precision, recall, fmeasure
# from keras.applications import resnet50, densenet
import keras.backend as K
from vggface import VGGFace
import argparse

# 指定GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
#进行配置，使用70%的GPU
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.7
session = tf.Session(config=config)
# 设置session
KTF.set_session(session)

# K.set_image_dim_ordering("th")

class decay_lr(Callback):
    '''
    n_epoch = no of epochs after decay should happen
    decay = decay rate
    '''
    def __init__(self, n_epoch, decay):
        super(decay_lr, self).__init__()
        self.n_epoch = n_epoch
        self.decay = decay

    def on_epoch_begin(self, epoch, logss={}):
        old_lr = K.get_value(self.model.optimizer.lr)
        if epoch > 1 and epoch % self.n_epoch == 0:
            new_lr = self.decay * old_lr
            K.set_value(self.model.optimizer.lr, new_lr)
        else:
            K.set_value(self.model.optimizer.lr, old_lr)
            
def main(args):
#    lr_decay = decay_lr(10, 0.5)
#    opt = Adam(lr=1e-4, decay=1e-5)
    
    # read label
    train_csv = pd.read_csv(args.train_csv)
    train_label = [i for i in train_csv['lianxing']]

    # read images
    train_data = np.load(open(args.train, 'rb'))
    train_data = train_data / 255.0
    train_data = np.transpose(train_data, (0, 3, 1, 2))
    
    print('train data and label shape: ', train_data.shape, np.shape(train_label))
    print('Train Data is done!')

    train_label = to_categorical(train_label, num_classes=args.num_classes)
    
    if args.val:
        val_csv = pd.read_csv(args.val_csv)
        val_label = [i for i in val_csv['lianxing']]
        val_data = np.load(open(args.val, 'rb'))
        val_data = val_data / 255.0
        val_data = np.transpose(val_data, (0, 3, 1, 2))
        val_label = to_categorical(val_label, num_classes=args.num_classes)
        print('val data and label shape: ', val_data.shape, np.shape(val_label))
        print('Val Data is done!')
    # model
    print('get model....')
    model_name = args.model
    
    model = VGGFace(include_top=False, model=model_name, weights='vggface', 
                    pooling='avg', input_shape=(224, 224, 3), classes=args.num_classes)
    
    # if you want to change the layers of model
#    fc5 = model.layers[-8].output
#    fc6 = Flatten()(fc5)
#    fc7_1 = Dense(256, activation='relu', name='fc7_1')(fc6)
#    dropout7_1 = Dropout(0.3)(fc7_1)
#    fc7_2 = Dense(128, activation='relu', name='fc7_2')(dropout7_1)
#    prediction = Dense(classes, activation='softmax')(fc7_2)
#    model = Model(inputs=model.input, outputs=prediction)
    
    model.summary()
    model.compile(optimizer=args.opt, loss='categorical_crossentropy', metrics=['accuracy', precision, recall, fmeasure])
    
    # callbacks
    filepath = args.out_dir + model_name + '_model/' + model_name + "-weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
    model_checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True)
    tensorboard_log = TensorBoard(args.out_dir + model_name + '_tensorboard/', write_graph=True, histogram_freq=0)
    csv = CSVLogger(args.out_dir + model_name + '_csv/' + model_name + '.csv')
    
    # train
    print('Trainning model......')
    if args.val:
        model.fit(train_data, train_label, batch_size=args.batch_size, epochs=args.epochs, callbacks=[model_checkpoint, csv, tensorboard_log],
                  verbose=1, shuffle=True, validation_data=(val_data, val_label))
    else:
        model.fit(train_data, train_label, batch_size=args.batch_size, epochs=args.epochs, callbacks=[model_checkpoint, csv, tensorboard_log],
                  verbose=1, shuffle=True)
    
if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument('--train', '-t', required=True, help='the .pkl file of train images')
    parse.add_argument('--val', '-v', help='the .pkl file of validation images')
    parse.add_argument('--train-file', '-tl', required=True, help='the .csv file of train labels')
    parse.add_argument('--val-file', '-vl', help='the .csv file of validation labels')
    parse.add_argument('--num-classes', '-c', type=int, required=True, help='the number of classes')
    parse.add_argument('--batch-size', '-batch', type=int, default=32)
    parse.add_argument('--epochs', '-e', type=int, default=100)
    parse.add_argument('--model', '-m', default='vgg16', help='the model for train, vgg16, resnet50 and senet50')
    parse.add_argument('--out-dir', '-o', default='./', help='the output path')
    parse.add_argument('--optimizer', '-opt', default='adam')
    
    args = parse.parse_args()
    
    main(args)
    
