
import os
os.chdir('../../')


import sys
sys.path.append("./code")
sys.path.append("./code/utils")
sys.path.append("./code/networks")


# Specify GPU Device
os.environ["CUDA_VISIBLE_DEVICES"]="1"

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
from os import listdir
from os.path import isfile, join
import cv2

from pygam import LinearGAM, s
import imageio

import matplotlib.gridspec as gridspec
import argparse


def get_continous_predicted_images(feature, x_test):
    feature_preds = list()
    lam_all = list()

    # loop through each neuron
    for i in tqdm(range(512)):
        X = feature
        y = x_test[:,i:(i+1)]
        
        #Fit a gam with the default parameters
        gam = LinearGAM(s(0, n_splines = 25)).fit(X, y)
        
        # grid search smoothing parameter
        lams = 10**np.arange(-4, 10, 1.)
        gam.gridsearch(X, y, lam = lams)
        
        # save optimal smoothing parameter
        lam_all.append(gam.lam[0][0])
        
        # create a grid to predict from clinical feature (i.e. age)
        # this predicts the latent feature for ages 18 through 100
        XX = np.arange(min_age, max_age, 1)
        feature_preds.append(gam.predict(X=XX))

    feature_preds = np.transpose(feature_preds, (1, 0))


    # Use the interpolated latent vectors in feature_preds to generate synthetic images
    images = list()
    const = [tf.ones((1, 1))]
    noise = gan.get_noise(1)
    for i in range(len(XX)):
        pred = gan.ma_generator.predict(const + [feature_preds[i:(i+1)]] * NB_BLOCKS + noise, verbose = False)
        images.append(pred[0,:,:,0])

    # this is the final sequence
    return np.array(images), lam_all


def get_latents(data_list):
    latents = []
    for i in tqdm(range(len(data_list))):
        data = gan.augmenter(data_list[i:i+1])
        latent = gan.ma_encoder.predict(data, verbose=0)
        latents.append(latent)

    return np.array(latents)

def generate_normalized_images(data_list):
    X = []
    for data in data_list:
        img = data
        img = ((img - img.min()) / (img.max() - img.min())) * 2 - 1
        X.append(img)
      
    if IMAGE_SIZE == 256:
        X = np.expand_dims(X, -1)
    return np.array(X)  

def plot_continous_grid(images, save_file, title):
    fig, axs = plt.subplots(8, 6, figsize=[50, 50])
    axs = axs.flatten()
    for i in range(len(axs)):
        axs[i].imshow(cv2.rotate(images[i], cv2.ROTATE_90_COUNTERCLOCKWISE), cmap = 'gray'); 
        axs[i].axis("off")
        axs[i].set_title(f"{i+min_age}", fontdict={'fontsize':30})

    fig.patch.set_facecolor('#FFFFFF')  
    fig.suptitle(title, size=50) 
    plt.savefig(save_file)
    plt.show()
    plt.close()

def plot_continous_masked_grid(images, masks, save_file, title):
    fig, axs = plt.subplots(8, 6, figsize=[50, 50])
    axs = axs.flatten()

    for i in range(len(axs)):
        axs[i].imshow(cv2.rotate((images[i]- images[0])*masks[i], cv2.ROTATE_90_COUNTERCLOCKWISE), cmap = 'gray', vmin=0, vmax=1); 
        axs[i].axis("off")
        axs[i].set_title(f"{i+min_age}", fontdict={'fontsize':30})

    fig.patch.set_facecolor('#FFFFFF')   
    fig.suptitle(title, size=50) 
    plt.savefig(save_file)
    plt.close()

def plot_continous_grid_diff(control_images, patient_images, save_file, title):
    fig, axs = plt.subplots(nrows=9, ncols=6, figsize=[60, 60])  # Increase figsize
    axs = axs.flatten()
    for i in range(len(axs)):
        if i < len(patient_images):
            im = axs[i].imshow(cv2.rotate(patient_images[i] - control_images[i], cv2.ROTATE_90_COUNTERCLOCKWISE), cmap='Spectral', vmin=-0.5, vmax=0.5)
            axs[i].axis("off")
            axs[i].set_title(f"{i+min_age}", fontdict={'fontsize': 40})  # Increase fontsize
        else:
            axs[i].axis("off")

    fig.patch.set_facecolor('#FFFFFF')
    cbar = fig.colorbar(im, ax=axs[-6:], pad=0.05, location="bottom")  # Reduce pad for better visibility
    cbar.ax.tick_params(labelsize=30)  # Increase tick label size for better visibility

    fig.suptitle(title, fontsize=60)  # Increase fontsize
    plt.savefig(save_file)
    plt.show()
    plt.close()

def plot_continous_masked_grid_diff(control_images, patient_images,control_mask, patient_mask, save_file, title):
    fig, axs = plt.subplots(nrows=9, ncols=6, figsize=[60, 60])  # Increase figsize
    axs = axs.flatten()
    for i in range(len(axs)):
        if i < len(patient_images):
            p_image = ((patient_images[i] - patient_images[0]) * patient_mask[i])
            c_image = ((control_images[i] - control_images[0]) * control_mask[i])
            im = axs[i].imshow(cv2.rotate(p_image - c_image, cv2.ROTATE_90_COUNTERCLOCKWISE), cmap='Spectral', vmin=-0.5, vmax=0.5)
            axs[i].axis("off")
            axs[i].set_title(f"{i+min_age}", fontdict={'fontsize': 40})  # Increase fontsize
        else:
            axs[i].axis("off")

    fig.patch.set_facecolor('#FFFFFF')
    cbar = fig.colorbar(im, ax=axs[-6:], pad=0.05, location="bottom")  # Reduce pad for better visibility
    cbar.ax.tick_params(labelsize=30)  # Increase tick label size for better visibility

    fig.suptitle(title, fontsize=60)  # Increase fontsize
    plt.savefig(save_file)
    plt.show()
    plt.close()

def create_animation(control_images, patient_images, save_file, title):
    fig, ax = plt.subplots(1, 3, figsize=[14, 6])
    for axis in ax.ravel():
        axis.set_axis_off()

    fig.suptitle(title, fontsize=20)

    # Initialize color bar as None
    cbar = None

    # Create text object for frame number
    frame_text = fig.text(0.5, 0.02, '', ha='center', va='center', fontsize=20)

    def update(i):
        nonlocal cbar
        data = cv2.rotate(patient_images[i], cv2.ROTATE_90_COUNTERCLOCKWISE)
        control = cv2.rotate(control_images[i], cv2.ROTATE_90_COUNTERCLOCKWISE)
        ax[0].imshow(control, animated=True, cmap="gray", vmin=0, vmax=1)
        ax[1].imshow(data, animated=True, cmap="gray", vmin=0, vmax=1)
        im = ax[2].imshow(data - control, animated=True, cmap='Spectral', vmin=-0.5, vmax=0.5)

        # Add color bar based on the first image
        if i == 0 and cbar is None:
            cbar = fig.colorbar(im, ax=ax[2], location="right", shrink=0.6, fraction=0.1)
            cbar.ax.tick_params(labelsize=10)

            ax[0].set_title('Controls', fontsize=10)  # Set top title
            ax[1].set_title('Patients', fontsize=10)  # Set top title
            ax[2].set_title('Patient-Controls', fontsize=10)  # Set top title

        # Update frame number text
        frame_text.set_text(f'Age {i +18}')

    ani = animation.FuncAnimation(fig, update, frames=range(len(patient_images)), interval=150)
    fig.patch.set_facecolor('#FFFFFF')

    plt.close()
    HTML(ani.to_jshtml())
    ani.save(save_file)

def create_animation_masked(control_images, patient_images, control_mask, patient_mask, save_file, title):
    fig, ax = plt.subplots(1, 3, figsize=[14, 6])
    for axis in ax.ravel():
        axis.set_axis_off()

    fig.suptitle(title, fontsize=20)

    # Initialize color bar as None
    cbar = None

    # Create text object for frame number
    frame_text = fig.text(0.5, 0.02, '', ha='center', va='center', fontsize=20)

    def update(i):
        nonlocal cbar
        data = cv2.rotate((patient_images[i] - patient_images[0]) * patient_mask[i], cv2.ROTATE_90_COUNTERCLOCKWISE)
        control = cv2.rotate((control_images[i] - control_images[0]) * control_mask[i], cv2.ROTATE_90_COUNTERCLOCKWISE)
        ax[0].imshow(control, animated=True, cmap = 'gray', vmin=0, vmax=1)
        ax[1].imshow(data, animated=True, cmap = 'gray', vmin=0, vmax=1)
        im = ax[2].imshow(data - control, animated=True, cmap='Spectral', vmin=-0.5, vmax=0.5)

        # Add color bar based on the first image
        if i == 0 and cbar is None:
            cbar = fig.colorbar(im, ax=ax[2], location="right", shrink=0.6, fraction=0.1)
            cbar.ax.tick_params(labelsize=10)

            ax[0].set_title('Controls', fontsize=10)  # Set top title
            ax[1].set_title('Patients', fontsize=10)  # Set top title
            ax[2].set_title('Patient-Controls', fontsize=10)  # Set top title

        # Update frame number text
        frame_text.set_text(f'Age {i +min_age}')

    ani = animation.FuncAnimation(fig, update, frames=range(len(patient_images)), interval=150)
    fig.patch.set_facecolor('#FFFFFF')

    plt.close()
    HTML(ani.to_jshtml())
    ani.save(save_file)


def get_unseen_valid_df(data_dir, metadata_path):
    controls_file = os.path.join(data_dir, "raw_data/processed/unique/control.pkl")
    patients_file = os.path.join(data_dir, "raw_data/processed/unique/patients.pkl")

    controls_df = pd.read_pickle(controls_file)
    patients_df = pd.read_pickle(patients_file)  

    val_path = os.path.join(data_dir,"val")
    test_path = os.path.join(data_dir,"test")
    val_files = [f for f in os.listdir(val_path) if isfile(join(val_path, f))]
    test_files = [f for f in os.listdir(test_path) if isfile(join(test_path, f))]
    all_files = val_files + test_files
    all_files = [''.join(k.split('.npy')[0]) + '.mat' for k in all_files]

    patients_meta_df = pd.read_excel(metadata_path, header=1, sheet_name='Patients') 
    controls_meta_df = pd.read_excel(metadata_path, header=0, sheet_name='Controls')

    patients_meta_df['file_name'] = patients_meta_df['ID'].apply(lambda x : ''.join(x.split('_')) + ".mat")
    controls_meta_df['file_name'] = controls_meta_df['NewID'].apply(lambda x : ''.join(x.split('_')) + ".mat")

    controls_df['file_name'] = controls_df['File'].apply(lambda x : ''.join(x.split('_')))
    patients_df['file_name'] = patients_df['File'].apply(lambda x : ''.join(x.split('_')))

    controls_df_meta_enhanced = pd.merge(controls_df, controls_meta_df, left_on='file_name', right_on='file_name', how='left')
    patients_df_meta_enhanced = pd.merge(patients_df, patients_meta_df, left_on='file_name', right_on='file_name', how='left')

    unseen_control_df = controls_df_meta_enhanced[controls_df_meta_enhanced['file_name'].isin(all_files)]
    unseen_patients_df = patients_df_meta_enhanced[patients_df_meta_enhanced['file_name'].isin(all_files)]

    valid_age_unseen_control_df = unseen_control_df[unseen_control_df['Age'].notna()]
    valid_age_unseen_patients_df = unseen_patients_df[unseen_patients_df['Age'].notna()]

    return valid_age_unseen_control_df, valid_age_unseen_patients_df

def normalize(img):
    return ((img - img.min()) / (img.max() - img.min()))

def get_permutation_masks(feature, x_test, lam_all, images):
    # # to perform permutations
    nb_samp = 100
    const = [tf.ones((1, 1))]
    noise = gan.get_noise(1)
    pmaps = [np.zeros(np.array(images).shape[1:])] * nb_samp
    for perm in tqdm(range(nb_samp)):
        # shuffle labels for a single permutation interation
        index = np.random.choice(range(feature.shape[0]), feature.shape[0], replace = False)
        feature = feature[index]
    
        
        # fit the gam using the shuffled labels
        feature_preds = list()
        for i in range(512):
            X = feature[:]
            y = x_test[:,i:(i+1)]
            gam = LinearGAM(s(0, n_splines = 25),  lam = lam_all[i]).fit(X, y)
            XX = np.arange(min_age, max_age, 1)
            feature_preds.append(gam.predict(X=XX))
        
        feature_preds = np.transpose(feature_preds, (1, 0))
        
        perm_images = list()
        for i in range(len(XX)):
            pred = gan.ma_generator.predict(const + [feature_preds[i:(i+1)]] * NB_BLOCKS + noise, verbose = False)
            perm_images.append(pred[0,:,:,0])
        
        perm_images = np.array(perm_images)
        
        perm_diffs = []
        diffs = []
        for i in range(len(perm_diffs)):
            perm_diffs.append(perm_images[i] - perm_images[0])
            diffs.append(images[i] - images[0])
            pmaps[i] = pmaps[i] + (np.abs(perm_diffs) > np.abs(diffs)) + 0.

    masks = []
    for i in range(len(pmaps)):
        mask = ((pmaps[i]/nb_samp) < 0.05) + 0.0
        masks.append(mask)
    return masks



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Simple script with seed and output directory arguments.')
    parser.add_argument('--seed', type=int, required=False, default=1, help='Seed for the random number generator')
    parser.add_argument('--slice', type=int, required=False, default=71, help='Slice number')
    parser.add_argument('--data_dir', type=str, required=False, default="/space/mcdonald-syn01/1/projects/ank028/workspace/figaan_packaged/data_brain", help='Directory to load data from')
    parser.add_argument('--weights_dir', type=str, required=False, default="/space/mcdonald-syn01/1/projects/ank028/workspace/figaan_packaged/output_brain", help='Directory to load data from')
    parser.add_argument('--metadata_path', type=str, required=False, default="/space/mcdonald-syn01/1/projects/ank028/workspace/figaan_packaged/data/forMachineLearningGroup_September2023_Demographic_ClinicalInfo.xlsx", help='Path to patient metadata excel file')
    parser.add_argument('--save_dir', type=str, required=True, help='Absolute path to directory to save the output file')
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
    valid_age_unseen_control_df, valid_age_unseen_patients_df = get_unseen_valid_df(data_dir, args.metadata_path)

    controls_list = valid_age_unseen_control_df['upscaled_image'].to_numpy()
    patients_list = valid_age_unseen_patients_df['upscaled_image'].to_numpy()
    controls_list_processed = generate_normalized_images(controls_list)
    patients_list_processed = generate_normalized_images(patients_list)

    print("Loading Model")

    gan = FIGAAN()
    save_found = gan.load_weights(os.path.join(weights_dir, 'models'))
    print("Save found: ", save_found)
    gan.compile()

    print("Generating Latents")
    control_latents = get_latents(controls_list_processed)
    patient_latents = get_latents(patients_list_processed)

    valid_age_unseen_control_df['latent'] = control_latents[:,0,:].tolist()
    valid_age_unseen_patients_df['latent'] = patient_latents[:,0,:].tolist()

    print(valid_age_unseen_patients_df['Age'].max())
    print(valid_age_unseen_patients_df['Age'].min())
    print(valid_age_unseen_control_df['Age'].max())
    print(valid_age_unseen_control_df['Age'].min())

    max_age = min(valid_age_unseen_control_df['Age'].max(), valid_age_unseen_patients_df['Age'].max())
    min_age = max(valid_age_unseen_control_df['Age'].min(), valid_age_unseen_patients_df['Age'].min())

    print("Creating plots for Patients")
    # Patients no perm
    feature = np.array(valid_age_unseen_patients_df['Age'].to_list())# this would be age or some other continuous feature
    x_test  = np.array(valid_age_unseen_patients_df['latent'].to_list())# this is the latent vector for each image
    patient_images, lam_patients = get_continous_predicted_images(feature, x_test)
    plot_continous_grid(patient_images, os.path.join(save_dir, "no_perm_age_patients.png",), "Continous Variation of Age in Patient (No Permutations)")

    print("Creating plots for Controls")
    # Controls no perm
    feature = np.array(valid_age_unseen_control_df['Age'].to_list())# this would be age or some other continuous feature
    x_test  = np.array(valid_age_unseen_control_df['latent'].to_list())# this is the latent vector for each image
    control_images, lam_controls = get_continous_predicted_images(feature, x_test)
    plot_continous_grid(control_images, os.path.join(save_dir, "no_perm_age_controls.png"), "Continous Variation of Age in Control (No Permutations)")

    print("Creating plots for Diff. This may take some time...")
    plot_continous_grid_diff(control_images, patient_images, os.path.join(save_dir, "no_perm_age_control_tle_diff.png"), 'Patient - Control at each age level (No Permutation)')
    create_animation(control_images, patient_images, os.path.join(save_dir, "no_perm_continous_age.gif"), 'Continous Variation of Age for Controls and Patients(No Permutation)')

    ############################permutations######################################
    if args.perm:
        print("Creating plots for patient with permutations")
        feature = np.array(valid_age_unseen_patients_df['Age'].to_list())# this would be age or some other continuous feature
        x_test  = np.array(valid_age_unseen_patients_df['latent'].to_list())# this is the latent vector for each image
        patient_masks = get_permutation_masks(feature, x_test, lam_patients, patient_images)
        plot_continous_masked_grid(patient_images, patient_masks, os.path.join(save_dir,"perm_age_patients.png"),"Continous Variation of Age in Patient (With Permutations)")

        print("Creating plots for control with permutations")
        # permutations
        feature = np.array(valid_age_unseen_control_df['Age'].to_list())# this would be age or some other continuous feature
        x_test  = np.array(valid_age_unseen_control_df['latent'].to_list())# this is the latent vector for each image
        control_masks = get_permutation_masks(feature, x_test, lam_controls, control_images)
        plot_continous_masked_grid(control_images, control_masks, os.path.join(save_dir,"perm_age_controls.png"),"Continous Variation of Age in Control (With Permutations)")

        print("Creating plots for diff with permutations. This may take some time...")
        plot_continous_masked_grid_diff(control_images, patient_images, control_masks, patient_masks, os.path.join(save_dir, "perm_age_control_tle_diff.png"), 'Patient - Control at each age level (With Permutation)')
        create_animation_masked(control_images, patient_images, control_masks, patient_masks, os.path.join(save_dir, "perm_continous_age.gif"), 'Continous Variation of Age for Controls and Patients(With Permutation)')