import argparse
import logging
import os
import random
import keras
import numpy as np
from keras import backend as K
from keras.callbacks import TensorBoard
from keras.datasets import mnist
from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D,Dropout
from keras.models import Sequential
import tensorflow as tf
tf.debugging.set_log_device_placement(True)
import nni

LOG = logging.getLogger('land_nas_keras')
LOG.setLevel(logging.DEBUG)

# Criar um manipulador de arquivo
log_file = 'land_nas_keras.log'
file_handler = logging.FileHandler(log_file)

# Definir o formato do log
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

# Adicionar o manipulador de arquivo ao logger
LOG.addHandler(file_handler)

K.set_image_data_format('channels_last')
TENSORBOARD_DIR = current_directory = os.getcwd() + '/logs'

H, W =64,64
NUM_CLASSES = 21

#Criar um modelo onde a arquitetura é alteravel
def create_model(hyper_params, input_shape=(H, W, 1), num_classes=NUM_CLASSES):
    '''
    Create simple convolutional model
    '''
    model = Sequential()

    model.add(Conv2D(64, kernel_size=3, input_shape=input_shape))
    for _ in range(hyper_params['camadas']):
        for _ in range(hyper_params['conv1']):
            model.add(Conv2D(32, kernel_size=3, activation='relu'))

            if(hyper_params['drop1'] == 1):
                model.add(Dropout(0.3))
                
        for _ in range(hyper_params['conv2']):
            model.add(Conv2D(64, kernel_size=3, activation='relu'))
   
            if(hyper_params['drop2'] == 1):
                model.add(Dropout(0.3))
                
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
   
    for _ in range(hyper_params['dense_layers1']):
        model.add(Dense(2048, activation='relu'))
    
    for _ in range(hyper_params['dense_layers2']):
        model.add(Dense(512, activation='relu'))

    model.add(Dense(num_classes, activation='softmax'))
    
    
    optimizer = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=optimizer, metrics=['accuracy'])
    return model


import glob
import cv2

def load_data(hyper_params):
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

    #Normalização dos dados
    x_train = np.expand_dims(np.array(train_images), -1).astype(float) / 255.
    x_test = np.expand_dims(np.array(test_images), -1).astype(float) / 255.
    
    #Atribuir a classe aos dados
    label_to_index = {label: index for index, label in enumerate(set(train_labels))}
    y_train = np.array([label_to_index[label] for label in train_labels])
    y_test = np.array([label_to_index[label] for label in test_labels])

    num_classes = len(label_to_index)
    
    #Label Encoder
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    #Regista dados no Log para debug
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
        #Enviar os valores de accuracy e loss para o frontend
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
    
    #Guardar o melhor modelo obtido durante o treino
    model_checkpoint_path = os.path.join(current_directory, 'best_model_land_nas.h5')
    
    #Callback para terminar o treino mais cedo caso a loss minimo não alterar durante o periodo patience
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, mode='min')
    
    checkpoint_callback = ModelCheckpoint(model_checkpoint_path, monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)

    x_train, y_train, x_test, y_test = load_data(params)
    model = create_model(params)

    model.fit(x_train, y_train, batch_size=args.batch_size, epochs=args.epochs, verbose=1,
        validation_data=(x_test, y_test), callbacks=[SendMetrics(), TensorBoard(log_dir=TENSORBOARD_DIR), checkpoint_callback])

    _, acc = model.evaluate(x_test, y_test, verbose=0)
    
    #Enviar para o frontend o resultado final
    LOG.debug('Final result is: %d', acc)
    nni.report_final_result(acc)

def generate_default_params():
    '''
    Generate default hyper parameters
    '''
    return {
        'camadas': 1,
        'conv1': 1,
        'conv1_aux':1,
        'drop1':1,
        'conv2': 1,
        'conv2_aux': 1,
        'drop2':1,
        'dense_layers1':2,
        'dense_layers2': 2
    }

if __name__ == '__main__':
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument("--batch_size", type=int, default=64, help="batch size", required=False)
    PARSER.add_argument("--epochs", type=int, default=10, help="Train epochs", required=False)

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
