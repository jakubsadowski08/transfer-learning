import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from keras.layers import Conv2DTranspose
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.preprocessing.image import \
    ImageDataGenerator
from keras.layers.core import Dense, Dropout
from sklearn.metrics import confusion_matrix
import keras
from tensorflow.keras import layers

num_of_test_samples = 392
batch_size = 32
epochs = 1
num_classes = 87

xception_datagen = ImageDataGenerator(
)

xception_test_datagen = ImageDataGenerator()
train_generator_xception = xception_datagen.flow_from_directory(
    "merged/train", target_size=(224, 224), batch_size=32,
    class_mode="categorical", shuffle=True, )

val_generator_xception = xception_datagen.flow_from_directory(
    "merged/val", target_size=(224, 224), batch_size=32,
    class_mode="categorical", shuffle=True, )
test_generator_xception = xception_test_datagen.flow_from_directory(
    "merged/test", batch_size=32,
    class_mode="categorical", shuffle=True, )

xception_base_model = tf.keras.applications.xception.Xception(
    input_shape=(224, 224, 3), include_top=False,
    weights='imagenet', classes=num_classes)

xception_base_model.trainable = False
inputs = keras.Input(shape=(224, 224, 3))
data_augmentation = keras.Sequential(
    [layers.RandomFlip("horizontal"),
     layers.RandomRotation(0.1), layers.RandomHeight(0.1),
     layers.RandomZoom(0.1),
     layers.RandomBrightness(factor=0.2)]
)
x = data_augmentation(inputs)
scale_layer = keras.layers.Rescaling(scale=1 / 127.5,
                                     offset=-1)
x = scale_layer(x)
x = xception_base_model(x, training=False)
x = keras.layers.GlobalAveragePooling2D()(x)
x = Dropout(0.2)(x)
outputs = Dense(num_classes, activation='sigmoid')(x)
xception_model = Model(inputs, outputs)

xception_model.summary()
xception_model.compile(optimizer=SGD(),
                       loss='categorical_crossentropy',
                       metrics=['accuracy'])
savebest = tf.keras.callbacks.ModelCheckpoint(
    'xception_beforefinetuning.h5', save_best_only=True)
history_xception = xception_model.fit(
    train_generator_xception,
    validation_data=val_generator_xception,
    epochs=epochs,
    callbacks=[savebest])
print(xception_model.predict_generator(
    test_generator_xception,
    num_of_test_samples // batch_size + 1))
xception_base_model.trainable = True
xception_model.summary()

xception_model.compile(optimizer=SGD(1e-5),
                       loss='categorical_crossentropy',
                       metrics=['accuracy'])
savebest = tf.keras.callbacks.ModelCheckpoint(
    'xception_afterfinetuning.h5', save_best_only=True)
fine_tuned_history_xception = xception_model.fit(
    train_generator_xception,
    validation_data=val_generator_xception,
    epochs=epochs,
    callbacks=[savebest])

f, ax = plt.subplots(2, 1, figsize=(10, 10))
ax[0].plot(history_xception.history['loss'],
           label='Training Loss')
ax[0].plot(history_xception.history['val_loss'],
           label='Validation Loss')

ax[1].plot(history_xception.history['accuracy'],
           label='Training Accuracy')
ax[1].plot(history_xception.history['val_accuracy'],
           label='Validation Accuracy')

ax[0].legend()
ax[0].set_xlabel("epochs")
ax[1].legend()
ax[1].set_xlabel("epochs")
plt.savefig("before_fine.png")

f, ax = plt.subplots(2, 1, figsize=(10, 10))
ax[0].plot(fine_tuned_history_xception.history['loss'],
           label='Training Loss')
ax[0].plot(
    fine_tuned_history_xception.history['val_loss'],
    label='Validation Loss')

ax[1].plot(
    fine_tuned_history_xception.history['accuracy'],
    label='Training Accuracy')
ax[1].plot(
    fine_tuned_history_xception.history['val_accuracy'],
    label='Validation Accuracy')

ax[0].legend()
ax[0].set_xlabel("epochs")
ax[1].legend()
ax[1].set_xlabel("epochs")
plt.savefig("after_fine.png")

Y_pred_res = xception_model.predict_generator(
    test_generator_xception,
    num_of_test_samples // batch_size + 1)
y_pred_res = np.argmax(Y_pred_res, axis=1)
print('Confusion Matrix')
conf_matrix_res = confusion_matrix(
    test_generator_xception.classes, y_pred_res)
ind = np.unravel_index(
    np.argmax(conf_matrix_res, axis=None),
    conf_matrix_res.shape)
print(ind)
for i in conf_matrix_res:
    print(i)
