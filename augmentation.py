import os
import random

import numpy as np

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import  matplotlib.pyplot as plt
from trains import Task





if __name__ == '__main__':
    task = Task.init(project_name="Shoe_prints", task_name="periodic_patterns")

    DATADIR = os.getcwd() + '/fixing_data/'
    IMG_SHAPE = (290, 105, 3)


    batch_size = 1
    seed = 1

    train_datagen = ImageDataGenerator(
        featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_whitening=False,
        width_shift_range=0.0,
        height_shift_range=0.0,
        brightness_range=(1,1),
        shear_range=0.0,
        zoom_range=0.5,
        channel_shift_range=0.0,
        fill_mode="nearest",
        cval=0.0,
        horizontal_flip=False,
        vertical_flip=False,
        preprocessing_function=None,
        data_format=None,
        dtype=None,
        validation_split=0.2,
        rescale=1. / 255,
       )

    test_datagen = ImageDataGenerator(rescale=1. / 255)
    train_generator = train_datagen.flow_from_directory(
        DATADIR ,
        target_size=(290, 105),
        batch_size=batch_size,
        class_mode='sparse',
        subset='training',
        shuffle=True,
        save_to_dir=os.getcwd() + '/preview/',
        seed=seed)

    validation_generator = train_datagen.flow_from_directory(
        DATADIR ,
        target_size=(290, 105),
        batch_size=batch_size,
        class_mode='sparse',
        subset='validation',
        seed= seed )
    #
    # test_generator = test_datagen.flow_from_directory(
    #     DATADIR + 'test/',
    #     target_size=(290, 105),
    #     batch_size=batch_size,
    #     class_mode='sparse',
    # )


    # Create the base model from the pre-trained model MobileNet V2
    base_model = tf.keras.applications.ResNet101V2(input_shape=IMG_SHAPE,
                                                   include_top=False,
                                                   weights='imagenet')

    feature_batch = base_model.output
    print(feature_batch.shape)

    # base_model.summary()


    base_model.trainable = False

    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
    feature_batch_average = global_average_layer(feature_batch)
    print(feature_batch_average.shape)

    dropout_layer = tf.keras.layers.Dropout(0.3)
    dropout_batch = dropout_layer(feature_batch)
    print(dropout_batch.shape)

    prediction_layer = tf.keras.layers.Dense(2, activation='softmax')
    prediction_batch = prediction_layer(dropout_batch)
    print(prediction_batch.shape)

    model = tf.keras.Sequential([
        base_model,
        global_average_layer,
        dropout_layer,
        prediction_layer
    ])

    base_learning_rate = 0.001
    model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=base_learning_rate),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    model.summary()

    print("==========model evaluate==========")
    model.evaluate(validation_generator)
    # train_datagen.fit(train_generator)
    # test_datagen.fit(test_generator)
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // batch_size,
        epochs=20,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // batch_size
        )


    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.ylim([min(plt.ylim()), 1])
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Cross Entropy')
    plt.ylim([0, 1.0])
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.show()


    print("==========model evaluate==========")
    model.evaluate(validation_generator)

    # # fine tuning
    # base_model.trainable = True
    # LAYERS_TO_FREEZE = 100
    # for layer in base_model.layers[:LAYERS_TO_FREEZE]:
    #     layer.trainable = False
    #
    # for layer in base_model.layers[LAYERS_TO_FREEZE:]:
    #     layer.trainable = True
    #
    # # new model compile with new trainable layers
    # base_learning_rate = 0.001
    # model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=base_learning_rate),
    #               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    #               metrics=['accuracy'])
    #
    # ft_history = model.fit(
    #     train_generator,
    #     steps_per_epoch=train_generator.samples // batch_size,
    #     epochs=100,
    #     validation_data=validation_generator,
    #     validation_steps=validation_generator.samples // batch_size)
    #
    # acc = ft_history.history['accuracy']
    # val_acc = ft_history.history['val_accuracy']
    #
    # loss = ft_history.history['loss']
    # val_loss = ft_history.history['val_loss']
    #
    # plt.figure(figsize=(8, 8))
    # plt.subplot(2, 1, 1)
    # plt.plot(acc, label='Training Accuracy')
    # plt.plot(val_acc, label='Validation Accuracy')
    # plt.legend(loc='lower right')
    # plt.ylabel('Accuracy')
    # plt.ylim([min(plt.ylim()), 1])
    # plt.title('Training and Validation Accuracy')
    #
    # plt.subplot(2, 1, 2)
    # plt.plot(loss, label='Training Loss')
    # plt.plot(val_loss, label='Validation Loss')
    # plt.legend(loc='upper right')
    # plt.ylabel('Cross Entropy')
    # plt.ylim([0, 1.0])
    # plt.title('Training and Validation Loss')
    # plt.xlabel('epoch')
    # plt.show()
    #
    # from sklearn.metrics import classification_report, confusion_matrix
    #
    # # Confution Matrix and Classification Report
    # Y_pred = model.predict(validation_generator, validation_generator.samples // batch_size)
    # y_pred = np.argmax(Y_pred, axis=1)
    # print('Confusion Matrix')
    # print(confusion_matrix(validation_generator.classes, y_pred))
    # print('Classification Report')
    #
    # # print(classification_report(validation_generator.classes, y_pred, target_names=test_labels))
    # print("==========evaluate after FT==========")
    # # model.evaluate(test_generator)