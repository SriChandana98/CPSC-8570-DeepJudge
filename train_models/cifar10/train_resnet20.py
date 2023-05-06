from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.datasets import cifar10
import numpy as np
import tensorflow as tf
#tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import os

import sys
sys.path.append('..')
from utils.resnet20 import resnet_v1

import warnings
warnings.filterwarnings("ignore")

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
        
        
def lr_schedule(epoch):
    lr = 1e-3
    if epoch > 70:
        lr *= 1e-3
    if epoch > 60:
        lr *= 1e-2
    elif epoch > 40:
        lr *= 1e-1 
    print('Learning rate: ', lr)
    return lr



if __name__ == '__main__':
    # TRAINING PARAMS
    batch_size = 64
    epochs = 20
    data_augmentation = True
    num_classes = 10
    n = 3
    version = 1
    train_nums = 25000
    save_dir = './models/'

    # Computed depth from supplied model parameter n
    if version == 1:
        depth = n * 6 + 2
    elif version == 2:
        depth = n * 9 + 2

    model_type = 'resnet%dv%d' % (depth, version)
    
    print("Loading the cifar data")

    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # Input image dimensions.
    input_shape = x_train.shape[1:]
    print("Input image dimensions: ",input_shape)

    # Normalize data.
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255

    # Convert class vectors to binary class matrices.
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    x_train = x_train[:train_nums]
    y_train = y_train[:train_nums]

    model = resnet_v1(input_shape=input_shape, depth=depth, num_classes=num_classes)
    print(model.summary())
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(learning_rate=lr_schedule(0)),
                  metrics=['accuracy'])

    model_name = f'cifar10_{model_type}_{epochs}.h5'
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    filepath = os.path.join(save_dir, model_name)

    # Prepare callbacks for model saving and for learning rate adjustment.
#     checkpoint = ModelCheckpoint(filepath=filepath,
#                                  monitor='val_accuracy',
#                                  verbose=1,
#                                  save_best_only=True,
#                                  mode='auto')

    lr_scheduler = LearningRateScheduler(lr_schedule)
    lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                                   cooldown=0,
                                   patience=5,
                                   min_lr=0.5e-6)

#     callbacks = [checkpoint, lr_reducer, lr_scheduler]
    callbacks = [lr_reducer, lr_scheduler]

    # Run training, with or without data augmentation.
    print("Starting training")
    if not data_augmentation:
        print('Not using data augmentation.')
        model.fit(x_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  validation_data=(x_test, y_test),
                  shuffle=True,
                  callbacks=callbacks)
    else:
        print('Using real-time data augmentation.')
        datagen = ImageDataGenerator(
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True)
        datagen.fit(x_train)
        # Fit the model on the batches generated by datagen.flow().
        history = model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                                      validation_data=(x_test, y_test),
                                      epochs=epochs, verbose=1,
                                      callbacks=callbacks,
                                      steps_per_epoch=x_train.shape[0] // batch_size)

    # Score trained model.
    scores = model.evaluate(x_test, y_test, verbose=1)
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])
    model.save(filepath)