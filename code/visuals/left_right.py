
import os
os.chdir('/space/mcdonald-syn01/1/projects/ank028/workspace/figaan_packaged/')


import sys
sys.path.append("./code")
sys.path.append("./code/utils")
sys.path.append("./code/networks")


# Specify GPU Device
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"


import numpy as np
import tensorflow as tf
import pandas as pd

from tensorflow.keras import preprocessing
from tensorflow.keras.callbacks import CSVLogger

from settings import *
import utils
from callbacks import *
from figaan import FIGAAN
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
from os.path import isfile, join
import cv2
import argparse

utils.reset_rand()

def generate_normalized_images(data_list):
    X = []
    for data in data_list:
        img = data
        img = ((img - img.min()) / (img.max() - img.min())) * 2 - 1
        X.append(img)
      
    if IMAGE_SIZE == 256:
        X = np.expand_dims(X, -1)
    return np.array(X)  

def get_latents(data_list):
    latents = []
    for i in tqdm(range(len(data_list))):
        data = gan.augmenter(data_list[i:i+1])
        latent = gan.ma_encoder.predict(data, verbose=0)
        latents.append(latent)

    return np.array(latents)

def get_unseen_valid_df(data_dir):
    left_file = os.path.join(data_dir, "raw_data/processed/side/left_TL.pkl")
    right_file = os.path.join(data_dir, "raw_data/processed/side/right_TL.pkl")

    left_df = pd.read_pickle(left_file)
    right_df = pd.read_pickle(right_file)  

    val_path = os.path.join(data_dir,"val")
    test_path = os.path.join(data_dir,"test")
    val_files = [f for f in os.listdir(val_path) if isfile(join(val_path, f))]
    test_files = [f for f in os.listdir(test_path) if isfile(join(test_path, f))]
    all_files = val_files + test_files
    all_files = [''.join(k.split('.npy')[0]) + '.mat' for k in all_files]

    left_df['file_name'] = left_df['File'].apply(lambda x : ''.join(x.split('_')))
    right_df['file_name'] = right_df['File'].apply(lambda x : ''.join(x.split('_')))

    valid_unseen_left_df = left_df[left_df['file_name'].isin(all_files)]
    valid_unseen_right_df = right_df[right_df['file_name'].isin(all_files)]

    return valid_unseen_left_df, valid_unseen_right_df

def get_generated_images(latents_list):
    generated_images = []
    for i in tqdm(range(len(latents_list))):
        w = latents_list[i]
        img = gan.ma_generator.predict(const_input + ([w] * NB_BLOCKS) + noise, verbose=0)
        generated_images.append(img)

    return np.array(generated_images)

def create_left_right_plot(left_images, right_images):
    fig, ax = plt.subplots(1, 3,figsize=[14, 6])
    for axis in ax.ravel():
        axis.set_axis_off()

    left = cv2.rotate(left_images, cv2.ROTATE_90_COUNTERCLOCKWISE)
    right = cv2.rotate(right_images, cv2.ROTATE_90_COUNTERCLOCKWISE)
    sub = right -left
    ax[0].imshow(left,  cmap="gray")
    ax[1].imshow(right,  cmap="gray")
    im = ax[2].imshow(sub, cmap = 'Spectral',  vmin = -0.25, vmax = 0.25)
    ax[0].title.set_text('Left TLE')
    ax[1].title.set_text('Right TLE')
    ax[2].title.set_text('Right TLE- Left TLE')
    cbar = fig.colorbar(im, ax=ax[2], location="right", shrink=0.6, fraction=0.1)
    cbar.ax.tick_params(labelsize=10)

    fig.patch.set_facecolor('#FFFFFF')
    plt.savefig(os.path.join(save_dir, "left_right_format_2D.png"))
    plt.show()

def create_animation(generated_images_transition):
    fig, ax = plt.subplots(1, 3, figsize=[12, 5])
    for axis in ax.ravel():
        axis.set_axis_off()

    fig.suptitle('Control to TLE variation', fontsize=20)

    # Initialize color bar as None
    cbar = None

    def update(i):
        nonlocal cbar
        left = cv2.rotate(generated_images_transition[0,0,:,:,0], cv2.ROTATE_90_COUNTERCLOCKWISE)
        data = cv2.rotate(generated_images_transition[i,0,:,:,0], cv2.ROTATE_90_COUNTERCLOCKWISE)
        ax[0].imshow(left, animated=True, cmap="gray", aspect=0.9)
        ax[1].imshow(data, animated=True, cmap="gray", aspect=0.9)
        im = ax[2].imshow(data-left, animated=True, cmap='Spectral', vmin=-0.25, vmax=0.25)

        # Add color bar based on the first image
        if i == 0 and cbar is None:
            cbar = fig.colorbar(im, ax=ax[2], location="right", shrink=0.6, fraction=0.1)
            cbar.ax.tick_params(labelsize=10)

            ax[0].set_title('Left TLE', fontsize=10)  # Set top title
            ax[1].set_title('Right TLE', fontsize=10)  # Set top title
            ax[2].set_title('Left TLE- Right TLE', fontsize=10)  # Set top title

    ani = animation.FuncAnimation(fig, update, frames=range(len(generated_images_transition)), interval=50)
    fig.patch.set_facecolor('#FFFFFF')

    plt.close()
    HTML(ani.to_jshtml())
    ani.save(os.path.join(save_dir, "left_right_format_2D.gif"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Simple script with seed and output directory arguments.')
    parser.add_argument('--seed', type=int, required=False, default=1, help='Seed for the random number generator')
    parser.add_argument('--slice', type=int, required=False, default=71, help='Slice number')
    parser.add_argument('--data_dir', type=str, required=False, default="/space/mcdonald-syn01/1/projects/ank028/workspace/figaan_packaged/data_brain", help='Directory to load data from')
    parser.add_argument('--weights_dir', type=str, required=False, default="/space/mcdonald-syn01/1/projects/ank028/workspace/figaan_packaged/output_brain", help='Directory to load data from')
    parser.add_argument('--save_dir', type=str, required=True, help='Directory to save the output file')
    parser.add_argument('--perm', action='store_true', help='Run visualizations with permutations')

    args = parser.parse_args()
    print("Args", args)

    np.random.seed(args.seed)
    data_dir = args.data_dir
    save_dir = args.save_dir
    slice = args.slice
    weights_dir = args.weights_dir
    perm = args.perm

    print("Processing Data")
    valid_unseen_left_df, valid_unseen_right_df = get_unseen_valid_df(data_dir)

    left_list = valid_unseen_left_df['upscaled_image'].to_numpy()
    right_list = valid_unseen_right_df['upscaled_image'].to_numpy()

    left_list_processed = generate_normalized_images(left_list)
    right_list_processed = generate_normalized_images(right_list)

    print("Loading Model")

    gan = FIGAAN()
    save_found = gan.load_weights(os.path.join(weights_dir, 'models'))
    print("Save found: ", save_found)
    gan.compile()

    left_latents = get_latents(left_list_processed)
    right_latents = get_latents(right_list_processed)

    avg_left_latent = np.mean(left_latents, axis=0)
    avg_right_latent = np.mean(right_latents, axis=0)

    batch_size = 1
    const_input = [tf.ones((batch_size, 1))]
    noise = gan.get_noise(batch_size)

    temp_c = np.expand_dims(avg_left_latent, axis = 0)
    temp_p = np.expand_dims(avg_right_latent, axis = 0)
    data = np.concatenate((temp_c, temp_p), axis=0)
    generated_images= get_generated_images(data)
    print("Creating plot for average TLE and average Control")

    create_left_right_plot(generated_images[0,0,:,:,0], generated_images[1,0,:,:,0])

    print("Creating animation. This may take some time.")
    transition_latents = np.linspace(avg_left_latent, avg_right_latent, num=100)
    generated_images_transition = get_generated_images(transition_latents)
    create_animation(generated_images_transition)