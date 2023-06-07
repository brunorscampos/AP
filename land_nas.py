# Copyright (c) Microsoft Corporation
# All rights reserved.
#
# MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and
# to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED *AS IS*, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import argparse
import logging

import os
import random
import keras
import numpy as np
from keras import backend as K
from keras.callbacks import TensorBoard
from keras.datasets import mnist
from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D,Conv2DTranspose
from keras.models import Sequential

import tensorflow as tf


import tensorflow as tf
tf.debugging.set_log_device_placement(True)


import nni

LOG = logging.getLogger('land_keras')
LOG.setLevel(logging.DEBUG)

# Criar um manipulador de arquivo
log_file = 'land_keras.log'
file_handler = logging.FileHandler(log_file)

# Definir o formato do log
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

# Adicionar o manipulador de arquivo ao logger
LOG.addHandler(file_handler)

#print(f"Tensorflow version: {tf.__version__}")
#print(f"Keras Version: {tf.keras.__version__}")
#print("GPU is", "available" if tf.config.list_physical_devices('GPU') else "NOT AVAILABLE")

K.set_image_data_format('channels_last')
TENSORBOARD_DIR = current_directory = os.getcwd() + '/logs'

H, W =64,64
NUM_CLASSES = 21

def create_mnist_model(hyper_params, input_shape=(H, W, 1), num_classes=NUM_CLASSES):
    
    
    '''
    Create simple convolutional model
    '''
    #layers = []
    #layers.append(Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    #layers.append(Conv2D(64, (3, 3), activation='relu'))
    #layers.append(Conv2D(64, (3, 3), activation='relu'))
    #layers.append(MaxPooling2D(pool_size=(2, 2)))
    #layers.append(Flatten())
    #
    #for _ in range(hyper_params['dense_layers']):
    #    layers.append(Dense(hyper_params['dense_nodes'], activation='relu'))
    #    
    #layers.append(Dense(num_classes, activation='softmax'))
    #    model = Sequential(layers)
    
    model = Sequential()

    model.add(Conv2D(64, kernel_size=3, input_shape=(H,W,1)))
    
    for _ in range(hyper_params['conv1']):
        model.add(Conv2D(32, kernel_size=3, activation='relu'))
        if(hyper_params['conv1_aux'] == 1):
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Flatten())
    for _ in range(hyper_params['conv2']):
        model.add(Conv2D(64, kernel_size=3, activation='relu'))
        if(hyper_params['conv2_aux'] == 1):
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Flatten())
            
    for _ in range(hyper_params['dense_layers']):
        model.append(Dense(100, activation='relu'))
        
    model.add(Conv2D(hyper_params['conv_nodes'], kernel_size=3, activation='relu'))
    model.add(Conv2D(hyper_params['conv_nodes'], kernel_size=3, activation='relu'))
    
    

        
    model.add(Dense(12544, input_shape=(hyper_params['dense_nodes'],)))
    
    model.add(Dense(num_classes, activation='softmax'))
    

    if hyper_params['optimizer'] == 'Adam':
        optimizer = keras.optimizers.Adam(learning_rate=hyper_params['learning_rate'])
    else:
        optimizer = keras.optimizers.SGD(learning_rate=hyper_params['learning_rate'], momentum=0.9)
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=optimizer, metrics=['accuracy'])
    return model


import glob
import cv2

def load_mnist_data(hyper_params):
    land_path = './archive/images/'  # Altere para o caminho correto do diretório MedNIST

    # Carregar imagens de treinamento
    train_images = []
    train_labels = []
    test_images = []
    test_labels = []

    for class_name in os.listdir(land_path):
        class_path = os.path.join(land_path, class_name)
        if os.path.isdir(class_path):
            images = glob.glob(os.path.join(class_path, '*.png'))
            num_images = len(images)
            num_test_images = int(num_images * 0.3)  # 30% para teste

            for i, image_path in enumerate(images):
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                image = cv2.resize(image, (W,H))

                if i < num_test_images:
                    test_images.append(image)
                    test_labels.append(class_name)
                else:
                    train_images.append(image)
                    train_labels.append(class_name)

    x_train = np.expand_dims(np.array(train_images), -1).astype(float) / 255.
    x_test = np.expand_dims(np.array(test_images), -1).astype(float) / 255.

    label_to_index = {label: index for index, label in enumerate(set(train_labels))}
    y_train = np.array([label_to_index[label] for label in train_labels])
    y_test = np.array([label_to_index[label] for label in test_labels])

    num_classes = len(label_to_index)
    
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    LOG.debug('x_train shape: %s', (x_train.shape,))
    LOG.debug('x_test shape: %s', (x_test.shape,))
    LOG.debug('Number of training samples: %d', len(train_images))
    LOG.debug('Number of test samples: %d', len(test_images))

    return x_train, y_train, x_test, y_test

class SendMetrics(keras.callbacks.Callback):
    '''
    Keras callback to send metrics to NNI framework
    '''
    def on_epoch_end(self, epoch, logs={}):
        '''
        Run on end of each epoch
        '''
        LOG.debug(logs)
        # TensorFlow 2.0 API reference claims the key is `val_acc`, but in fact it's `val_accuracy`
        if 'val_acc' in logs and 'val_loss' in logs:
            nni.report_intermediate_result({'accuracy': logs['val_acc'], 'loss': logs['val_loss']})
        elif 'val_accuracy' in logs and 'val_loss' in logs:
            nni.report_intermediate_result({'accuracy': logs['val_accuracy'], 'loss': logs['val_loss']})
        else:
            LOG.warning("Accuracy or Loss not found in logs.")
            
from keras.callbacks import ModelCheckpoint


def train(args, params):
    '''
    Train model
    '''
    current_directory = os.getcwd()

    model_checkpoint_path = os.path.join(current_directory, 'best_model_land.h5')
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, mode='min')
    checkpoint_callback = ModelCheckpoint(model_checkpoint_path, monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)

    x_train, y_train, x_test, y_test = load_mnist_data(params)
    model = create_mnist_model(params)

    model.fit(x_train, y_train, batch_size=args.batch_size, epochs=args.epochs, verbose=1,
        validation_data=(x_test, y_test), callbacks=[SendMetrics(), TensorBoard(log_dir=TENSORBOARD_DIR), checkpoint_callback])

    _, acc = model.evaluate(x_test, y_test, verbose=0)
    LOG.debug('Final result is: %d', acc)
    nni.report_final_result(acc)

def generate_default_params():
    '''
    Generate default hyper parameters
    '''
    return {
        'optimizer': 'Adam',
        'learning_rate': 0.0001,
        'dense_nodes':32,
        'conv_nodes':32,
    }

if __name__ == '__main__':
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument("--batch_size", type=int, default=64, help="batch size", required=False)
    PARSER.add_argument("--epochs", type=int, default=10, help="Train epochs", required=False)
    PARSER.add_argument("--num_train", type=int, default=60000, help="Number of train samples to be used, maximum 60000", required=False)
    PARSER.add_argument("--num_test", type=int, default=10000, help="Number of test samples to be used, maximum 10000", required=False)

    ARGS, UNKNOWN = PARSER.parse_known_args()

    try:
        # get parameters from tuner
        RECEIVED_PARAMS = nni.get_next_parameter()
        LOG.debug(RECEIVED_PARAMS)
        PARAMS = generate_default_params()
        PARAMS.update(RECEIVED_PARAMS)
        # train
        train(ARGS, PARAMS)
    except Exception as e:
        LOG.exception(e)
        raise
