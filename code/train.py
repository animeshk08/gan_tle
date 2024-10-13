# %%
import os
os.chdir('../')


import sys
sys.path.append("./code")
sys.path.append("./code/utils")
sys.path.append("./code/networks")


# Specify GPU Device
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

# %%
import numpy as np
import tensorflow as tf
from tensorflow.keras import preprocessing
from tensorflow.keras.callbacks import CSVLogger

from settings import *
import utils
from callbacks import *
from figaan import FIGAAN

utils.reset_rand()

# %%
def data_generator(data_dir, fnames, batch_size):
  while True:
    samp = np.random.choice(range(len(fnames)), size = batch_size, replace = False)

    fnames_batch = fnames[samp]
    X = list()
    for fname in fnames_batch:
      img = np.load(os.path.join(data_dir, fname))
      img = ((img - img.min()) / (img.max() - img.min())) * 2 - 1
      X.append(img)
      
    if IMAGE_SIZE == 256:
        X = np.expand_dims(X, -1)
    X = np.array(X)
    
    yield X
    
# define data generators
train_ids = np.array(os.listdir(os.path.join(DATA_DIR, 'train')))
valid_ids = np.array(os.listdir(os.path.join(DATA_DIR, 'val')))
training_generator = data_generator(os.path.join(DATA_DIR, 'train'), train_ids, BATCH_SIZE)
validation_generator = data_generator(os.path.join(DATA_DIR, 'val'), valid_ids, BATCH_SIZE)

# %%
def generate_sample_test_images(path):
    X = []
    dir_list = os.listdir(path)
    for file in dir_list:
        img = np.load(os.path.join(path, file))
        img = ((img - img.min()) / (img.max() - img.min())) * 2 - 1
        X.append(img)
      
    if IMAGE_SIZE == 256:
        X = np.expand_dims(X, -1)
    return np.array(X)    

# %%
samples_z = np.load(os.path.join(LOAD_DIR, "samples_z.npy"))
samples_noise = np.load(os.path.join(LOAD_DIR, "samples_noise.npy"))
# samples_images = np.load(os.path.join(LOAD_DIR, "samples_images.npy"))
samples_images = generate_sample_test_images(os.path.join(DATA_DIR, "test"))

# %%
gan = FIGAAN()
save_found = gan.load_weights(os.path.join(LOAD_DIR, 'models'))
gan.compile()

# %%
history = gan.fit(
    training_generator,
    validation_data = validation_generator,
    batch_size = BATCH_SIZE,
    epochs = NB_EPOCHS, 
    steps_per_epoch = 100, 
    validation_batch_size = BATCH_SIZE,
    validation_steps = 100, 
    shuffle = True,
    callbacks = [
        Updates(),
        SaveSamplesMapping(samples_z, samples_noise),
        SaveSamplesEncoder(samples_images),
        CSVLogger(os.path.join(OUTPUT_DIR, 'results_figaan.csv'), append = True),
        SaveModels()
    ]
)


