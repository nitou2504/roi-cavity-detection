import numpy as np
import os
import shutil
import pandas as pd
import cv2
import csv
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import RepeatedStratifiedKFold
from constants import EPOCHS, BATCH_SIZE, REPETITION, FOLDS, CHECKPOINT_FREQ, USE_AUGMENTATION

def create_directories():
    # Define directory paths
    directories = [
        'histories/CNN',
        'histories/DCNN1',
        'histories/DCNN2',
        'histories/DCNN3',
        'checkpoint',
        'results',
        'train_results'
    ]

    # Create directories if they don't exist
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
        else:
            # Clean directory if it already exists
            shutil.rmtree(directory)
            os.makedirs(directory)

    print("Directories created and cleaned successfully.")



def get_labels(file_path):
    data = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            data.append(row)

    flattened_array = np.array(data).flatten()
    labels = flattened_array.astype(int)  # Convert labels to integer data type
    return labels

def load_image(file_path):
    image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)  # Load image as is
    image = image / 255.0  # Normalize pixel values
    return image

def load_data(data_path):
    trainX, trainY = list(), list()
    data, labels = list(), list()
    count = 0
    dim = 100

    # Load training data
    train_files = [f for f in sorted(os.listdir(data_path)) if f.endswith(".jpg")]
    for file in train_files:
        image = load_image(os.path.join(data_path, file))  # Replace with your image loading code
        data.append(image)

    # Load training labels
    train_labels = get_labels(os.path.join(data_path, 'labels.csv'))

    # Combine data and labels
    data = np.array(data)

    # Split data into training sets
    trainX = data
    trainY = train_labels

    num_classes = np.max(train_labels) + 1

    return trainX, trainY, num_classes, dim

# Define the F1Score metric class
class F1Score(tf.keras.metrics.Metric):
    def __init__(self, name='f1_score', **kwargs):
        super(F1Score, self).__init__(name=name, **kwargs)
        self.true_positives = self.add_weight(name='tp', initializer='zeros')
        self.false_positives = self.add_weight(name='fp', initializer='zeros')
        self.false_negatives = self.add_weight(name='fn', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(tf.round(y_pred), tf.float32)

        true_positives = tf.reduce_sum(y_true * y_pred)
        false_positives = tf.reduce_sum((1 - y_true) * y_pred)
        false_negatives = tf.reduce_sum(y_true * (1 - y_pred))

        self.true_positives.assign_add(true_positives)
        self.false_positives.assign_add(false_positives)
        self.false_negatives.assign_add(false_negatives)

    def result(self):
        precision = self.true_positives / (self.true_positives + self.false_positives + tf.keras.backend.epsilon())
        recall = self.true_positives / (self.true_positives + self.false_negatives + tf.keras.backend.epsilon())
        f1_score = 2 * ((precision * recall) / (precision + recall + tf.keras.backend.epsilon()))
        return f1_score

    def reset_states(self):
        self.true_positives.assign(0.0)
        self.false_positives.assign(0.0)
        self.false_negatives.assign(0.0)

# define CNN
def define_CNN(num_classes, dim):
    filter_size = (3, 3)
    pool_size = (2, 2)
    CNN = Sequential([Conv2D(4, filter_size, 
                        input_shape=(dim, dim, 1)),
                      MaxPooling2D(pool_size),
                      Flatten(),
                      Dense(num_classes, activation='softmax'),
                      ])
    
    return CNN

# define DCNN1
def define_DCNN1(num_classes, dim):
    filter_size = (3, 3)
    pool_size = (2, 2)
    DCNN1 = Sequential([
        Conv2D(8, (filter_size), activation='relu', 
                    input_shape=(dim, dim, 1)),
        MaxPooling2D(pool_size=pool_size),
        #
        Conv2D(16, filter_size, activation='relu'),
        MaxPooling2D(pool_size=pool_size),
        #
        Flatten (),
        #
        Dense(num_classes, activation='softmax')

    ])
    return DCNN1

# define DCNN2
def define_DCNN2(num_classes, dim):
    filter_size = (3, 3)
    pool_size = (2, 2)
    DCNN2 = Sequential([
        Conv2D(8, (filter_size), activation='relu',
                    input_shape=(dim, dim, 1)),
        MaxPooling2D(pool_size=pool_size),
        #
        Conv2D(16, filter_size, activation='relu'),
        MaxPooling2D(pool_size=pool_size),
        #
        Conv2D(32, filter_size, activation='relu'),
        MaxPooling2D(pool_size=pool_size),
        #
        Flatten(),
        #
        Dense(num_classes, activation='softmax')

    ])
    
    return DCNN2

# define DCNN3
def define_DCNN3(num_classes, dim):
    filter_size = (3, 3)
    pool_size = (2, 2)
    DCNN3 = Sequential([
        Conv2D(8, (filter_size), activation='relu',
                    input_shape=(dim, dim, 1)),
        MaxPooling2D(pool_size=pool_size),
        #
        Conv2D(16, filter_size, activation='relu'),
        MaxPooling2D(pool_size=pool_size),
        #
        Conv2D(32, filter_size, activation='relu'),
        MaxPooling2D(pool_size=pool_size),
        #
        Conv2D(64, filter_size, activation='relu'),
        MaxPooling2D(pool_size=pool_size),
        #
        Flatten(),
        #
        Dense(num_classes, activation='softmax')

    ])
    
    return DCNN3

# model callbacks
def model_callbacks(filepath):
    checkpoint = ModelCheckpoint(filepath, 
                                    verbose=1, 
                                    monitor='val_auc', 
                                    save_best_only=False, 
                                    save_weights_only=False, 
                                    mode='auto',
                                    period=CHECKPOINT_FREQ) # checkpoint freq
    return checkpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# training the model using a repeated stratified k-fold cross-validation
def training(dataX, dataY, num_classes, dim):
    histories_CNN, histories_DCNN1, histories_DCNN2, histories_DCNN3 = list(), list(), list(), list()
    e = EPOCHS
    bs = BATCH_SIZE
    repetition = REPETITION
    folds = FOLDS

    # Data augmentation settings
    datagen = ImageDataGenerator(
        horizontal_flip=True,
        vertical_flip=True,
        rotation_range=45
    )

    # prepare cross validation
    rkfold = RepeatedStratifiedKFold(n_splits=folds, n_repeats=repetition, random_state=1)

    # enumerate splits
    for i, (train_ix, test_ix) in enumerate(rkfold.split(dataX, dataY)):
        # select rows for train and test
        trainX, trainY, valX, valY = dataX[train_ix], dataY[train_ix], dataX[test_ix], dataY[test_ix]
        
        # Reshape the input data
        trainX = trainX.reshape(trainX.shape[0], trainX.shape[1], trainX.shape[2], 1)
        valX = valX.reshape(valX.shape[0], valX.shape[1], valX.shape[2], 1)

        # apply data augmentation
        train_generator = datagen.flow(trainX, to_categorical(trainY), batch_size=bs)

        # define and compile the models
        CNN = define_CNN(num_classes, dim)
        DCNN1 = define_DCNN1(num_classes, dim)
        DCNN2 = define_DCNN2(num_classes, dim)
        DCNN3 = define_DCNN3(num_classes, dim)

        # compile models
        # f1_weighted aumentar
        CNN.compile(
            loss='categorical_crossentropy',
            optimizer=Adam(learning_rate=0.0001),
            metrics=['accuracy', 'Precision', 'Recall', 'AUC', F1Score()]
        )
        DCNN1.compile(
            optimizer=Adam(learning_rate=0.0001),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'Precision', 'Recall', 'AUC', F1Score()]
        )
        DCNN2.compile(
            loss='categorical_crossentropy',
            optimizer=Adam(learning_rate=0.0001),
            metrics=['accuracy', 'Precision', 'Recall', 'AUC', F1Score()]
        )
        DCNN3.compile(
            loss='categorical_crossentropy',
            optimizer=Adam(learning_rate=0.0001),
            metrics=['accuracy', 'Precision', 'Recall', 'AUC', F1Score()]
        )

        # model checkpoint
        print(f"**************** fold - {i}")
        CNN_filepath = "checkpoint/CNN_weights_best_epoch_{epoch:02d}" + f"_fold_{i}" + "_auc_{val_auc:.2f}.hdf5"
        DCNN1_filepath = "checkpoint/DCNN1_weights_best_epoch_{epoch:02d}" + f"_fold_{i}" + "_auc_{val_auc:.2f}.hdf5"
        DCNN2_filepath = "checkpoint/DCNN2_weights_best_epoch_{epoch:02d}" + f"_fold_{i}" + "_auc_{val_auc:.2f}.hdf5"
        DCNN3_filepath = "checkpoint/DCNN3_weights_best_epoch_{epoch:02d}" + f"_fold_{i}" + "_auc_{val_auc:.2f}.hdf5"

        CNN_checkpoint = model_callbacks(CNN_filepath)
        DCNN1_checkpoint = model_callbacks(DCNN1_filepath)
        DCNN2_checkpoint = model_callbacks(DCNN2_filepath)
        DCNN3_checkpoint = model_callbacks(DCNN3_filepath)
        callbacks_list = [CNN_checkpoint, DCNN1_checkpoint, DCNN2_checkpoint, DCNN3_checkpoint]

        if USE_AUGMENTATION:
            # fit model with data augmentation
            history_CNN = CNN.fit(
                train_generator,
                steps_per_epoch=len(trainX) // bs,
                epochs=e,
                validation_data=(valX, to_categorical(valY)),
                verbose=2,
                callbacks=[callbacks_list[0]]
            )
            history_DCNN1 = DCNN1.fit(
                train_generator,
                steps_per_epoch=len(trainX) // bs,
                epochs=e,
                validation_data=(valX, to_categorical(valY)),
                verbose=2,
                callbacks=[callbacks_list[1]]
            )
            history_DCNN2 = DCNN2.fit(
                train_generator,
                steps_per_epoch=len(trainX) // bs,
                epochs=e,
                validation_data=(valX, to_categorical(valY)),
                verbose=2,
                callbacks=[callbacks_list[2]]
            )
            history_DCNN3 = DCNN3.fit(
                train_generator,
                steps_per_epoch=len(trainX) // bs,
                epochs=e,
                validation_data=(valX, to_categorical(valY)),
                verbose=2,
                callbacks=[callbacks_list[3]]
            )
        else:
            # fit model without augmentation

            history_CNN = CNN.fit(trainX, to_categorical(trainY), epochs=e, batch_size=bs,
                            validation_data=(valX, to_categorical(valY)), verbose=2, callbacks=[callbacks_list[0]])
            history_DCNN1 = DCNN1.fit(trainX, to_categorical(trainY), epochs=e, batch_size=bs, 
                             validation_data=(valX, to_categorical(valY)), verbose=2, callbacks=[callbacks_list[1]])
            history_DCNN2 = DCNN2.fit(trainX, to_categorical(trainY), epochs=e, batch_size=bs, 
                             validation_data=(valX, to_categorical(valY)), verbose=2, callbacks=[callbacks_list[2]])
            history_DCNN3 = DCNN3.fit(trainX, to_categorical(trainY), epochs=e, batch_size=bs, 
                             validation_data=(valX, to_categorical(valY)), verbose=2, callbacks=[callbacks_list[3]])


        # export history to csv file
        history_CNN_DF = pd.DataFrame.from_dict(history_CNN.history)
        history_CNN_DF.to_csv('histories/CNN/CNN_historyCSV_fold_' + str(i) + '.csv')
        history_DCNN1_DF = pd.DataFrame.from_dict(history_DCNN1.history)
        history_DCNN1_DF.to_csv('histories/DCNN1/DCNN1_historyCSV_fold_' + str(i) + '.csv')
        history_DCNN2_DF = pd.DataFrame.from_dict(history_DCNN2.history)
        history_DCNN2_DF.to_csv('histories/DCNN2/DCNN2_historyCSV_fold_' + str(i) + '.csv')
        history_DCNN3_DF = pd.DataFrame.from_dict(history_DCNN3.history)
        history_DCNN3_DF.to_csv('histories/DCNN3/DCNN3_historyCSV_fold_' + str(i) + '.csv')


# run the experiment
if __name__ == "__main__":
    # Call the function to create and clean directories
    create_directories()

    # load training dataset
    trainX, trainY, num_classes, dim = load_data('train_dataset') #default value has the path

    # train the models
    training(trainX, trainY, num_classes, dim)

    print("Finished succesfully")
