import argparse
import logging
import os
import keras
import numpy as np
from keras import backend as K
from keras.callbacks import TensorBoard
from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D,Conv2DTranspose
from keras.models import Sequential
import tensorflow as tf
tf.debugging.set_log_device_placement(True)
import nni
from keras import layers
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from keras.callbacks import ModelCheckpoint
np.random.seed(0)

LOG = logging.getLogger('reg_keras')
LOG.setLevel(logging.DEBUG)

# Criar um manipulador de arquivo
log_file = 'reg_keras.log'
file_handler = logging.FileHandler(log_file)

# Definir o formato do log
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

# Adicionar o manipulador de arquivo ao logger
LOG.addHandler(file_handler)

K.set_image_data_format('channels_last')
TENSORBOARD_DIR = current_directory = os.getcwd() + '/logs'

#Criar um modelo onde os hiperparametros são alteraveis
def create_model(hyper_params):
    '''
    Create convolutional model
    '''
    model = keras.Sequential([
        layers.Dense(hyper_params['dense_nodes1'], activation=hyper_params['activation'], input_shape=[8]),
        layers.Dense(hyper_params['dense_nodes2'], activation=hyper_params['activation']),
        layers.Dense(hyper_params['dense_nodes3'], activation=hyper_params['activation']),
        layers.Dense(hyper_params['dense_nodes4'], activation=hyper_params['activation']),
        layers.Dense(1)
    ])

    if hyper_params['optimizer'] == 'Adam':
        optimizer = keras.optimizers.Adam(learning_rate=hyper_params['learning_rate'])
    else:
        optimizer = keras.optimizers.SGD(learning_rate=hyper_params['learning_rate'], momentum=0.9)
    
    model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae', 'mse'])
    return model

#Função para dar load ao dataset
def load_data(hyper_params):
    df = pd.read_csv('usa_real_estate/realtor-data.csv')
    df.drop(['prev_sold_date'],axis=1,inplace=True)
    for i in df.columns:
        df[i].fillna(df[i].mode()[0], inplace=True)
    lb_make = LabelEncoder()
    df["status"] = lb_make.fit_transform(df['status'])
    df["city"] = lb_make.fit_transform(df['city'])
    df["state"] = lb_make.fit_transform(df['state'])
    x = df[["status","bed","bath","acre_lot","city","state","zip_code","house_size"]].to_numpy()
    y = df["price"].to_numpy()
    X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=2022)

    return X_train, y_train, X_test, y_test

class SendMetrics(keras.callbacks.Callback):
    '''
    Keras callback to send metrics to NNI framework
    '''
    def on_epoch_end(self, epoch, logs={}):
        '''
        Run on end of each epoch
        '''
        LOG.debug(logs)
        
        # Enviar os valores de mae,mse e loss para o frontend
        if 'val_mae' in logs and 'val_loss' in logs:
            nni.report_intermediate_result({'mae': logs['val_mae'], 'mse': logs['val_mse'], 'loss': logs['val_loss']})
        else:
            LOG.warning("Accuracy or Loss not found in logs.")
            
def train(args, params):
    '''
    Train model
    '''
    current_directory = os.getcwd()

    #Guardar o melhor modelo obtido durante o treino
    model_checkpoint_path = os.path.join(current_directory, 'best_model_land.h5')
    
    #Callback para terminar o treino mais cedo caso a loss minimo não alterar durante o periodo patience
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, mode='min')
    
    checkpoint_callback = ModelCheckpoint(model_checkpoint_path, monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)

    x_train, y_train, x_test, y_test = load_data(params)
    model = create_model(params)

    model.fit(x_train, y_train, batch_size=args.batch_size, epochs=args.epochs, verbose=1,
        validation_data=(x_test, y_test), callbacks=[SendMetrics(), TensorBoard(log_dir=TENSORBOARD_DIR), checkpoint_callback])

    _, mae, mse = model.evaluate(x_test, y_test, verbose=0)
    
    #Enviar para o frontend o resultado final
    LOG.debug('Final result is: MAE %d MSE %d', mae,mse)
    nni.report_final_result(mae)

def generate_default_params():
    '''
    Generate default hyper parameters
    '''
    return {
        'optimizer': 'Adam',
        'learning_rate': 0.0001,
        'dense_nodes1':32,
        'dense_nodes2':32,
        'dense_nodes3':32,
        'dense_nodes4':32,
        'activation':'relu',
    }

if __name__ == '__main__':
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument("--batch_size", type=int, default=64, help="batch size", required=False)
    PARSER.add_argument("--epochs", type=int, default=30, help="Train epochs", required=False)

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