## 1. Import Modules and Dataset


# Commented out IPython magic to ensure Python compatibility.
import os

import warnings
warnings.filterwarnings("ignore")

import random
import numpy as np
import pandas as pd
import requests
from io import BytesIO
from PIL import Image, ImageFile, ImageOps, ImageFilter
import cv2


import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import tensorflow as tf
from tensorflow.keras import Model, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense, BatchNormalization, Conv2D, MaxPool2D, AveragePooling2D, Input, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam


from time import time

from tensorflow.keras import backend as K
from keras.utils.vis_utils import plot_model

# %matplotlib inline

from google.colab import drive
drive.mount('/content/drive')

import timeit
start = timeit.default_timer()

gpu_info = !nvidia-smi
gpu_info = '\n'.join(gpu_info)
if gpu_info.find('failed') >= 0:
  print('Not connected to a GPU')
else:
  print(gpu_info)

# Import the dataset

!gdown http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar #dog breed classification dataset
!gdown http://vision.stanford.edu/aditya86/ImageNetDogs/annotation.tar
!pip install patool
import patoolib
patoolib.extract_archive("/content/annotation.tar", outdir= "/content/")
patoolib.extract_archive("/content/images.tar", outdir= "/content/")

"""## 2. Helper Functions"""

def show_dir_images(breed_list, n_to_show):

  plt.figure(figsize=(16,16))
  img_dir = "./Data/{}/".format(breed_list)

  images = os.listdir(img_dir)[:n_to_show]
  for i in range(n_to_show):
      img = mpimg.imread(img_dir + images[i])
      plt.subplot(n_to_show/4+1, 4, i+1)
      plt.imshow(img)
      plt.axis('off')
  # breeds_list = [b.split('-',1)[1] for b in breed_list]

# breeds_list = [b.split('-',1)[1] for b in breed_list]

def plot_history_scores(history):
    with plt.style.context('seaborn-whitegrid'):
        fig = plt.figure(figsize=(15,7))
        # summarize history for accuracy
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], color="g")
        plt.plot(history.history['val_accuracy'],
                 linestyle='--', color="orange")
        plt.title("Model's val_accuracy" , fontsize=18)
        plt.ylabel('val_accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        # summarize history for loss
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], color="g")
        plt.plot(history.history['val_loss'],
                 linestyle='--', color="orange")
        plt.title("Model's val_Loss", fontsize=18)
        plt.ylabel('val_Loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.show()

def benchmark():
  with plt.style.context('seaborn-whitegrid'):
      plt.figure(figsize=(15,7))
      try:
        history_model.history

        plt.plot(history_model.history['accuracy'],
                label='CNN - Mean accuracy: {:.2f}'.format(
                    np.mean(history_model.history['accuracy'])))
      except:
        pass

      try:
        plt.plot(history_nasnet.history['accuracy'],
              label='Nasnet - Mean accuracy: {:.2f}'.format(
                  np.mean(history_nasnet.history['accuracy'])))
      except:
        pass

      try:
        plt.plot(history_xception.history['accuracy'],
                label='Xception - Mean accuracy: {:.2f}'.format(
                np.mean(history_xception.history['accuracy'])))

      except:
        pass

      try:
        plt.plot(fine_tuned_history.history['accuracy'],
                label='Fine-tuned Nasnet - Mean accuracy: {:.2f}'.format(
                    np.mean(fine_tuned_history.history['accuracy'])))
      except:
        pass

      plt.title('Accuracy of differents ConvNet tested over epochs',
                fontsize=18)
      plt.ylabel('Accuracy')
      plt.xlabel('epoch')
      plt.legend(loc='upper left')
      plt.show()

def show_results(df):
  fig = plt.figure(1, figsize=(12,12))
  fig.patch.set_facecolor('#343434')
  plt.suptitle("Predicted VS actual for Nasnet model fine-tuned",
              y=.92, fontsize=22,
              color="white")
  n = 0
  for i in range(8):
    n+=1
    r = int(np.random.randint(0, df.shape[0], 1))
    plt.subplot(3,4,n)
    plt.subplots_adjust(hspace = 0.1, wspace = 0.1)

    image = Image.open('/content/output/test/'+df['Filename'][r])
    plt.imshow(image)
    plt.title('Actual = {}\nPredicted = {}'.format((df['Filename'][r].split('-',1)[1]).split('/',1)[0] , df['Predictions'][r]),
              color="white")
    plt.xticks([]) , plt.yticks([])
  plt.show()

from sklearn.metrics import precision_recall_fscore_support

def classification_report_df(y_test,y_pred,classes):
    clf_rep = precision_recall_fscore_support(y_test, y_pred)
    avgs=[]
    totalsum = np.sum(clf_rep[3])
    for i in range(0,3):
        avgs.append(np.sum(clf_rep[i]*clf_rep[3])/totalsum)

    avgs.append(totalsum)
    mylist = [list(x) for x in clf_rep]
    clf_rep_all = [x + [y] for x,y in zip(mylist,avgs)]
    indices = list(classes) +['avg/total']
    out_dict = {
                 "precision" :clf_rep_all[0]
                ,"recall" : clf_rep_all[1]
                ,"f1-score" : clf_rep_all[2]
                ,"support" : clf_rep_all[3]
                }
    out_df = pd.DataFrame(out_dict, index = indices)
    out_df[["precision","recall","f1-score"]]= out_df[["precision","recall","f1-score"]].apply(lambda x: round(x,2))
    return out_df

# Define a early stopping

stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy',mode='max', verbose=1,patience=2)

"""## 3. Data preprocessing and visualization"""

breed_list = os.listdir("/content/Images/")

num_classes = len(breed_list)
print("{} breeds".format(num_classes))

n_total_images = 0
for breed in breed_list:
    n_total_images += len(os.listdir("/content/Images/{}".format(breed)))
print("{} images".format(n_total_images))

# Commented out IPython magic to ensure Python compatibility.
# %%time
# # Images cropping
# 
# os.mkdir('Data')
# for breed in breed_list:
#     os.mkdir('Data/' + breed)
# print('Created {} folders to store cropped images of the different breeds.'.format(len(os.listdir('Data'))))
# 
#

import xml.etree.ElementTree as ET
# copy from https://www.kaggle.com/gabrielloye/dogs-inception-pytorch-implementation
# reduce the background noise

for breed in os.listdir('Data'):
    for file in os.listdir('/content/Annotation/{}'.format(breed)):
        img = Image.open('/content/Images/{}/{}.jpg'.format(breed, file))
        tree = ET.parse('/content/Annotation/{}/{}'.format(breed, file))
        xmin = int(tree.getroot().findall('object')[0].find('bndbox').find('xmin').text)
        xmax = int(tree.getroot().findall('object')[0].find('bndbox').find('xmax').text)
        ymin = int(tree.getroot().findall('object')[0].find('bndbox').find('ymin').text)
        ymax = int(tree.getroot().findall('object')[0].find('bndbox').find('ymax').text)
        img = img.crop((xmin, ymin, xmax, ymax))
        img = img.convert('RGB')

        # img = ImageOps.equalize(img, mask = None)  # égalisation d'histogrammes
        # img = img.filter(ImageFilter.BoxBlur(1))	 # Image Denoising

        img = img.resize((331, 331))




        img.save('Data/' + breed + '/' + file + '.jpg')
show_dir_images(breed_list[8], 8)

# Data split into Train,Validation and test data
! pip install split-folders[full]

import splitfolders
#https://pypi.org/project/split-folders/

# Split with a ratio.
splitfolders.ratio("./Data", output="output",
    seed=1337, ratio=(.6, .2, .2), group_prefix=None, move=False) # default values

# Image

img_size = 331 #img_size = 299
batch_size = 32
seed = 42

src_path_train = "./output/train/"
src_path_val = "./output/val/"

# Data augmentation

train_datagen = ImageDataGenerator(
        rescale=1 / 255.0,
        zoom_range=0.3,
        width_shift_range=0.05,
        height_shift_range=0.05,
        shear_range=0.3,
        horizontal_flip=True,
        vertical_flip=True,
        rotation_range=40,
        fill_mode="nearest",
)

valid_datagen = ImageDataGenerator(
        rescale=1 / 255.0
)

train_generator = train_datagen.flow_from_directory(
    directory=src_path_train,
    target_size=(img_size, img_size),
    color_mode="rgb",
    batch_size=batch_size,
    class_mode="categorical",

    shuffle=True,
    seed=seed
)


valid_generator = valid_datagen.flow_from_directory(
    directory=src_path_val,
    target_size=(img_size, img_size),
    color_mode="rgb",
    batch_size=batch_size,
    class_mode="categorical",
    shuffle=False,
    seed=seed
)

STEP_SIZE_TRAIN = train_generator.n//train_generator.batch_size
STEP_SIZE_VALID = valid_generator.n//valid_generator.batch_size

"""## 4. Simple Model CNN from scratch as a baseline"""

K.clear_session()

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),input_shape=(331,331,3)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='same'))
model.add(Dropout(0.3))
model.add(Conv2D(64, kernel_size=(3, 3)))
model.add(BatchNormalization(axis=3, scale=False))
model.add(Activation('relu'))
model.add(GlobalAveragePooling2D())
model.add(Dense(480, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(120, activation='softmax'))

model.summary()

model.compile(optimizer="adam",
              loss="categorical_crossentropy",
              metrics=["accuracy"]
              )

history_model = model.fit_generator(train_generator,
                    validation_data = valid_generator,
                    steps_per_epoch = STEP_SIZE_TRAIN,
                    validation_steps = STEP_SIZE_VALID,
                    callbacks=[stop_early],
                    epochs=10)

plot_history_scores(history_model)

"""## 5. Creating, compiling and training the model"""

# Update ImageDataGenerator with preprocess_input

train_datagen = ImageDataGenerator(
        #rescale=1 / 255.0,
        rotation_range=20,
        zoom_range=0.3,
        width_shift_range=0.05,
        height_shift_range=0.05,
        shear_range=0.3,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode="nearest",
        preprocessing_function = tf.keras.applications.nasnet.preprocess_input
)

valid_datagen = ImageDataGenerator(preprocessing_function = tf.keras.applications.nasnet.preprocess_input)


train_generator = train_datagen.flow_from_directory(
    directory=src_path_train,
    target_size=(img_size, img_size),
    color_mode="rgb",
    batch_size=batch_size,
    class_mode="categorical",

    shuffle=True,
    seed=seed
)


valid_generator = valid_datagen.flow_from_directory(
    directory=src_path_val,
    target_size=(img_size, img_size),
    color_mode="rgb",
    batch_size=batch_size,
    class_mode="categorical",

    shuffle=False,
    seed=seed
)

STEP_SIZE_TRAIN = train_generator.n//train_generator.batch_size
STEP_SIZE_VALID = valid_generator.n//valid_generator.batch_size

from tensorflow.keras.applications.nasnet import NASNetLarge

# Creating the NASNetLarge model
base_model = NASNetLarge(input_shape=(img_size, img_size, 3), weights='imagenet', include_top=False)

# freezing all the layers of NASNetLarge model by making them non trainable
for layer in base_model.layers:
    layer.trainable = False

# ADDING OUR NETWORK INCLUDING OUTPUT LAYER ON TOP

# add a global spatial average pooling layer
x = base_model.output
x = BatchNormalization()(x)
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)

# add a fully connected layer 120 neurons
predictions = Dense(120, activation='softmax')(x)

# this is the model we will train
model_nasnet = Model(inputs=base_model.input, outputs=predictions)

#@title
"""
from tensorflow.keras.applications.xception import Xception
# Creating the Xception model
base_model = Xception(input_shape=(img_size, img_size, 3), weights='imagenet', include_top=False)

# freezing all the layers of Xception model by making them non trainable
for layer in base_model.layers:
    layer.trainable = False

# ADDING OUR NETWORK INCLUDING OUTPUT LAYER ON TOP

# add a global spatial average pooling layer
x = base_model.output
x = BatchNormalization()(x)
x = GlobalAveragePooling2D()(x)

# add a fully connected layer 120 neurons
predictions = Dense(120, activation='softmax')(x)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

model.summary()
"""
"""
from tensorflow.keras.applications.resnet_v2 import ResNet152V2
# Creating the ResNet152V2 model
base_model = ResNet152V2(input_shape=(img_size, img_size, 3), weights='imagenet', include_top=False)

# freezing all the layers of ResNet152V2 model by making them non trainable
for layer in base_model.layers:
    layer.trainable = False

# ADDING OUR NETWORK INCLUDING OUTPUT LAYER ON TOP

# add a global spatial average pooling layer
x = base_model.output
x = BatchNormalization()(x)
x = GlobalAveragePooling2D()(x)

# add a fully connected layer 120 neurons
predictions = Dense(120, activation='softmax')(x)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

model.summary()
"""

"""# Creating the InceptionV3 model
from tensorflow.keras.applications import InceptionV3
base_model = InceptionV3(input_shape=(img_size, img_size, 3), weights='imagenet', include_top=False)

# freezing all the layers of InceptionV3 model by making them non trainable
for layer in base_model.layers:
    layer.trainable = False

# ADDING OUR NETWORK INCLUDING OUTPUT LAYER ON TOP

# add a global spatial average pooling layer
x = base_model.output
x = BatchNormalization()(x)
x = GlobalAveragePooling2D()(x)

# add a fully connected layer 120 neurons
predictions = Dense(120, activation='softmax')(x)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

model.summary()"""

model_nasnet.compile(optimizer=Adam(1e-4), loss='categorical_crossentropy',metrics=['accuracy'])

history_nasnet = model_nasnet.fit_generator(train_generator,
                    validation_data = valid_generator,
                    steps_per_epoch = STEP_SIZE_TRAIN,
                    validation_steps = STEP_SIZE_VALID,
                    callbacks=[stop_early],
                    epochs=10)

plot_history_scores(history_nasnet)

benchmark()

Nasnet_mean_accuracy = np.mean(history_nasnet.history['val_accuracy'])
Nasnet_mean_loss = np.mean(history_nasnet.history['val_loss'])
print("-" * 50)
print("Nasnet base model validation Scores :")
print("-" * 50)
print("Mean validation accuracy: {:.2f}"\
      .format(Nasnet_mean_accuracy))
print("Mean validation loss score: {:.2f}"\
      .format(Nasnet_mean_loss))

# Update ImageDataGenerator with preprocess_input

xception_train_datagen = ImageDataGenerator(
        rotation_range=20,
        zoom_range=0.3,
        width_shift_range=0.05,
        height_shift_range=0.05,
        shear_range=0.3,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode="nearest",
        preprocessing_function = tf.keras.applications.xception.preprocess_input
)

xception_valid_datagen = ImageDataGenerator(
        preprocessing_function = tf.keras.applications.xception.preprocess_input
)



xception_train_generator = train_datagen.flow_from_directory(
    directory=src_path_train,
    target_size=(img_size, img_size),
    color_mode="rgb",
    batch_size=batch_size,
    class_mode="categorical",

    shuffle=True,
    seed=seed
)


xception_valid_generator = valid_datagen.flow_from_directory(
    directory=src_path_val,
    target_size=(img_size, img_size),
    color_mode="rgb",
    batch_size=batch_size,
    class_mode="categorical",

    shuffle=False,
    seed=seed
)

STEP_SIZE_TRAIN = xception_train_generator.n//xception_train_generator.batch_size
STEP_SIZE_VALID = xception_valid_generator.n//xception_valid_generator.batch_size

from tensorflow.keras.applications.xception import Xception
# Creating the Xception model
base_model = Xception(input_shape=(img_size, img_size, 3), weights='imagenet', include_top=False)

# freezing all the layers of Xception model by making them non trainable
for layer in base_model.layers:
    layer.trainable = False

# ADDING OUR NETWORK INCLUDING OUTPUT LAYER ON TOP

# add a global spatial average pooling layer
x = base_model.output
x = BatchNormalization()(x)
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)

# add a fully connected layer 120 neurons
predictions = Dense(120, activation='softmax')(x)

# this is the model we will train
model_xception = Model(inputs=base_model.input, outputs=predictions)

model_xception.compile(optimizer=Adam(1e-4), loss='categorical_crossentropy',metrics=['accuracy'])

history_xception= model_xception.fit_generator(xception_train_generator,
                    validation_data = xception_valid_generator,
                    steps_per_epoch = STEP_SIZE_TRAIN,
                    validation_steps = STEP_SIZE_VALID,
                    callbacks=[stop_early],
                    epochs=10)

plot_history_scores(history_xception)

xcept_mean_accuracy = np.mean(history_xception.history['val_accuracy'])
xcept_mean_loss = np.mean(history_xception.history['val_loss'])
print("-" * 50)
print("Xception base model validation Scores :")
print("-" * 50)
print("Mean validation accuracy: {:.2f}"\
      .format(xcept_mean_accuracy))
print("Mean validation loss: {:.2f}"\
      .format(xcept_mean_loss))

benchmark()

"""optimisation"""

img_size = 331 #img_size = 299
batch_size = 32
seed = 42

train_datagen = ImageDataGenerator(
        #rescale=1 / 255.0,
        rotation_range=20,
        zoom_range=0.3,
        width_shift_range=0.05,
        height_shift_range=0.05,
        shear_range=0.3,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode="nearest",
        preprocessing_function = tf.keras.applications.nasnet.preprocess_input
)

valid_datagen = ImageDataGenerator(preprocessing_function = tf.keras.applications.nasnet.preprocess_input)


train_generator = train_datagen.flow_from_directory(
    directory=src_path_train,
    target_size=(img_size, img_size),
    color_mode="rgb",
    batch_size=batch_size,
    class_mode="categorical",

    shuffle=True,
    seed=seed
)


valid_generator = valid_datagen.flow_from_directory(
    directory=src_path_val,
    target_size=(img_size, img_size),
    color_mode="rgb",
    batch_size=batch_size,
    class_mode="categorical",

    shuffle=False,
    seed=seed
)

pip install keras-tuner --upgrade

import keras_tuner as kt
from tensorflow import keras

def model_builder(hp):
    # Load base model
    nasnet_model = NASNetLarge(input_shape=(img_size, img_size, 3), weights='imagenet', include_top=False)

    # freezing all the layers of NASNetLarge model by making them non trainable
    for layer in nasnet_model.layers:
        layer.trainable = False



    base_output = nasnet_model.output

    # Tune dense units
    hp_units = hp.Int('dense_units',
                      min_value=64,
                      max_value=260,
                      step=32,
                      default=128)

    base_output = Dense(units=hp_units,
                        activation='relu')(base_output)
    base_output = Dropout(0.5)(base_output)

    base_output = BatchNormalization()(base_output)
    base_output = GlobalAveragePooling2D()(base_output)
    base_output = Dropout(0.5)(base_output)


    # Output : new classifier
    predictions = Dense(120, activation='softmax')(base_output)

    # Define new model
    my_nasnet_model = Model(inputs=nasnet_model.input,
                       outputs=predictions)

    # Tune learning rate
    hp_learning_rate = hp.Choice(
        name='learning_rate',
        values=[1e-2, 1e-3, 1e-4])

    my_nasnet_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"])

    return my_nasnet_model

from tensorflow.keras.callbacks import EarlyStopping

# Tune the learning rate for the optimizer
# Constuct the tuner of kerastuner
tuner = kt.RandomSearch(
    model_builder,
    objective='val_accuracy',
    max_trials=3)

# Search best params
tuner.search(
    train_generator,
    validation_data=valid_generator,
    steps_per_epoch = STEP_SIZE_TRAIN,
    validation_steps = STEP_SIZE_VALID,
    epochs=10,
    callbacks=[stop_early
               #early_stopping
               ]
             )

# Get the optimal hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

print("-" * 50)
print("Nasnet Hyperparameters optimization :")
print("-" * 50)
print(f"""
Best learning rate : {best_hps.get('learning_rate')}.\n
Best Dense units : {best_hps.get('dense_units')}.""")

hypermodel = tuner.hypermodel.build(best_hps)
hypermodel.fit(
    train_generator,
    validation_data=valid_generator,
    steps_per_epoch = STEP_SIZE_TRAIN,
    validation_steps = STEP_SIZE_VALID,
    epochs=10,
    callbacks=[stop_early]
    )

hypermodel.save('nasnet_hypermodel.h5')
print("Model saved")

def nasnet_fine_tune(nb_layers):
    # Load the pre trained model
    hypermodel_t = tf.keras.models.load_model('./nasnet_hypermodel.h5')

    # re train the last layers
    for i, layer in enumerate(hypermodel_t.layers):
        if i < nb_layers:
            layer.trainable = False
        else:
            layer.trainable = True

    # Compile model
    hypermodel_t.compile(
        optimizer='adam',
        loss="categorical_crossentropy",
        metrics=["accuracy"])

    return hypermodel_t

# Dont train the 1036 first layers
my_tuned_nasnet_model = nasnet_fine_tune(1036) # train only the last 10 layers
fine_tuned_history = my_tuned_nasnet_model.fit(
    train_generator,
    validation_data=valid_generator,
    steps_per_epoch = STEP_SIZE_TRAIN,
    validation_steps = STEP_SIZE_VALID,
    epochs=20,
    callbacks=[stop_early]
    )

len(my_tuned_nasnet_model.layers)

plot_history_scores(fine_tuned_history)

benchmark()

src_path_test = "./output/test/"
test_datagen = ImageDataGenerator(preprocessing_function = tf.keras.applications.nasnet.preprocess_input)
test_generator = test_datagen.flow_from_directory(
    directory=src_path_test,
    target_size=(img_size, img_size),
    color_mode="rgb",
    batch_size=1,
    class_mode='categorical' ,
    shuffle=False,
    seed=seed
)

STEP_SIZE_TEST = test_generator.n//test_generator.batch_size

# Evaluation nasnet fine tuned sur données test

# Model evaluation on test set

Nasnet_eval = fine_tuned_history.model.evaluate(
    test_generator,
    verbose=1)
print("-" * 50)
print("Nasnet model evaluation :")
print("-" * 50)
print('Test Loss: {:.3f}'.format(Nasnet_eval[0]))
print('Test Accuracy: {:.3f}'.format(Nasnet_eval[1]))

# Make predictions

Y_pred = fine_tuned_history.model.predict(test_generator, steps=STEP_SIZE_TEST,verbose=1)
y_pred = np.argmax(Y_pred, axis=1)
y_test = test_generator.classes

# Save the last model
fine_tuned_history.model.save('Nasnet_trained_model.h5')
print("Last model saved")

"""## 6. Making predictions"""

labels = (train_generator.class_indices)
labels = dict((v,k) for k,v in labels.items())

label_predictions = [labels[k].split('-',1)[1] for k in y_pred]

filenames=test_generator.filenames
"""filenames= [k.split('/',1)[1] for k in filenames]"""
results=pd.DataFrame({"Filename":filenames,
                      "Predictions":label_predictions})

show_results(results)

A = range(0,len(filenames))

test_breed = [(k.split('/',1)[0]).split('-',1)[1] for k in filenames]
results['Not_good'] = 0

for k in A:
  if (test_breed [k] != label_predictions [k]):
    results['Not_good'][k] = 1

false_results = results[results['Not_good'] > 0]
false_results.reset_index(drop=True, inplace=True)
show_results(false_results)

breeds_list = [labels[k].split('-',1)[1] for k in labels]

from sklearn.metrics import confusion_matrix
import seaborn as sns

# Confusion Matrix
cf_matrix = confusion_matrix(y_test, y_pred)

fig = plt.figure(figsize=(22,22))
ax = sns.heatmap(cf_matrix, annot=True)
ax.set_xlabel("Predicted labels", color="g")
ax.set_ylabel("True labels", color="orange")
ax.xaxis.set_ticklabels(breeds_list,
                        rotation='vertical')
ax.yaxis.set_ticklabels(breeds_list,
                        rotation='horizontal')
plt.title("Confusion Matrix on Nasnet predicted results\n",
          fontsize=18)
plt.show()

wrong_predict = pd.DataFrame(data=confusion_matrix(y_test, y_pred),index=breeds_list, columns=breeds_list)
for k in breeds_list:
  wrong_predict[k][k] = 0
  for j in breeds_list:
    if (wrong_predict [k][j] < 4):
      wrong_predict [k][j] = 0
wrong_predict = wrong_predict.loc[:, (wrong_predict != 0).any(axis=0)]
wrong_predict = wrong_predict[~(wrong_predict == 0).all(axis=1)]
fig = plt.figure(figsize=(12,12))
ax = sns.heatmap(wrong_predict, annot=True)
ax.set_xlabel("Predicted labels", color="g")
ax.set_ylabel("True labels", color="orange")
ax.xaxis.set_ticklabels(#breeds_list,
                        wrong_predict.columns,
                        rotation='vertical')
ax.yaxis.set_ticklabels(#breeds_list,
                        wrong_predict.index,
                        rotation='horizontal')
plt.title("Falsely predicted results\n",
          fontsize=18)
plt.show()

from sklearn.metrics import classification_report
report = classification_report(
    y_test, y_pred,
    target_names = breeds_list
    )
print(report)

Report = classification_report_df(y_test,y_pred,breeds_list)
Report[Report['precision']<0.90]

stop = timeit.default_timer()
print('Time: ', stop - start)

classes = labels
for k in range(0, len(classes)):
  classes[k] = classes[k].split('-',1)[1]

import pickle
filename = 'classes.pickle'
outfile = open(filename,'wb')
pickle.dump(classes,outfile)
outfile.close()