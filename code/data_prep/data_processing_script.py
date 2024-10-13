
import mat73
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage import rotate
import os
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split
import math
import argparse
from tqdm import tqdm

def check_and_create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def write_numpy_files(dir_path, data):
    count = 0
    for i in range(len(data)):
        file_name = data[i][0].split('.')[0] + ".npy"
        np.save(os.path.join(dir_path, file_name), data[i][2])# 2 has the upscaled image
        count+=1 
    print("Written {} file to {}".format(count, dir_path))

def write_numpy_data(data_list, save_dir, val_size = 0.1, test_size=4):
    np.random.shuffle(data_list)
    
    val_path = os.path.join(save_dir,"val/")
    train_path = os.path.join(save_dir,"train/")
    test_path = os.path.join(save_dir,"test/")
    
    check_and_create_dir(train_path)
    check_and_create_dir(val_path)
    check_and_create_dir(test_path)

    test_list = data_list[:test_size]
    train_data = data_list[test_size:]

    train_list, val_list, _, _ = train_test_split(train_data, np.zeros(len(train_data)), test_size=val_size, shuffle=False)

    print("Train", len(train_list), "Val", len(val_list), "Test", len(test_list))

    write_numpy_files(train_path, train_list)
    write_numpy_files(val_path, val_list)
    write_numpy_files(test_path, test_list)

    
def read_data(path, primaryIndex = "session", secondaryIndex = "pre"):
    data = {}
    dir_list = os.listdir(path)
    for file in tqdm(dir_list):
        data_info = mat73.loadmat(os.path.join(path, file))
        try: 
            image = data_info[primaryIndex]["vbm_gm"]["dat"]
        except Exception as e:
            image = data_info[secondaryIndex]["vbm_gm"]["dat"]
        data[file] = image
    return data

def convert_to_df(data_dict):
    df = pd.DataFrame([(k,v) for k, v in data_dict.items()], columns=['File', 'image'])
    return df

def upscale_and_update(df, slice=71):
    new_df = df.copy()
    images = new_df['image']
    images_list = images.tolist() 
    upscaled_list = []  
    for img in images_list:
        upscaled_list.append(cv2.resize(img[:,slice,:], (256, 256), interpolation = cv2.INTER_LINEAR))
    new_df['upscaled_image'] = upscaled_list
    return new_df

def process_control_patients(controls_dir, slice, patients_dir):
    #Read data and write to DF
    print("Working on controls")
    control_data = read_data(controls_dir, primaryIndex = "session", secondaryIndex = "pre")
    control_data_df = convert_to_df(control_data)
    print("Working on TLEs")
    patients_data = read_data(patients_dir, primaryIndex = "pre", secondaryIndex = "session")
    patients_data_df = convert_to_df(patients_data)

    # Upscale images and update the DF
    controls_df_upscaled = upscale_and_update(control_data_df, slice=slice)
    patients_df_upscaled = upscale_and_update(patients_data_df, slice=slice)

    # Final DF structure:
    # 'File' - Name of the file
    # 'image' - a 3 dimension NP array representation of the image
    # 'upscaled_image' - a 2 dimension (256,256) image of the specified slice 
    return controls_df_upscaled, patients_df_upscaled

def create_left_right_tle_splits(patients_df, save_dir, metadata_path):
    patients_df['file_name'] = patients_df['File'].apply(lambda x : ''.join(x.split('_')))
    patients_meta_df = pd.read_excel(metadata_path, header=1, sheet_name='Patients')
    patients_meta_df['file_name'] = patients_meta_df['ID'].apply(lambda x : ''.join(x.split('_')) + ".mat")
    left_tle = patients_meta_df.loc[patients_meta_df['Side of Epilepsy'] == 'L']
    right_tle = patients_meta_df.loc[patients_meta_df['Side of Epilepsy'] == 'R']
    print("Number of left TLE in metadata", len(left_tle))
    print("Number of right TLE in metdata", len(right_tle))
    left_tle_patient_ids = left_tle['file_name'].to_list()
    right_tle_patient_ids = right_tle['file_name'].to_list()
    left_patients_df = patients_df[patients_df['file_name'].isin(left_tle_patient_ids)]
    right_patients_df = patients_df[patients_df['file_name'].isin(right_tle_patient_ids)]

    print("Number of LTLE in dataset", len(left_patients_df))
    print("Number of RTLE in dataset",len(right_patients_df))
    check_and_create_dir(os.path.join(save_dir, "rawdata"))
    left_patients_df.to_pickle(os.path.join(save_dir, "rawdata/left_TL.pkl"))
    right_patients_df.to_pickle(os.path.join(save_dir, "rawdata/right_TL.pkl"))
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Simple script with seed and output directory arguments.')
    parser.add_argument('--seed', type=int, required=False, default=1, help='Seed for the random number generator')
    parser.add_argument('--slice', type=int, required=False, default=71, help='Slice number')
    parser.add_argument('--controls_dir', type=str, required=False, default="/space/mcdonald-syn01/1/projects/ekaestner/Diffusion_T1_forErik_12.15.23/Processed-Controls/", help='Directory to load Controls data from')
    parser.add_argument('--patients_dir', type=str, required=False, default="/space/mcdonald-syn01/1/projects/ekaestner/Diffusion_T1_forErik_12.15.23/Processed-Patients/", help='Directory to load Patients data from')
    parser.add_argument('--save_dir', type=str, required=True, help='Directory to save the processed data')
    parser.add_argument('--metadata_path', type=str, required=False, default="/space/mcdonald-syn01/1/projects/ank028/workspace/figaan_packaged/data/forMachineLearningGroup_September2023_Demographic_ClinicalInfo.xlsx", help='Path to patient metadata excel file')
    parser.add_argument('--n_test', type=int, required=False, default=4, help='Number of samples in test from each of control and patient')

    args = parser.parse_args()
    print("Args", args)
    np.random.seed(args.seed)

    print("Processing MAT files to dataframe")
    controls_df_upscaled, patients_df_upscaled = process_control_patients(args.controls_dir, args.slice, args.patients_dir)
    save_dir = args.save_dir
    check_and_create_dir(save_dir)
    check_and_create_dir(os.path.join(save_dir, "rawdata"))

    # Write dataframe -
    print(f"Found Controls: {len(controls_df_upscaled)} TLE: {len(patients_df_upscaled)}")
    print("Writing control and patients dataframe")
    controls_df_upscaled.to_pickle(os.path.join(save_dir, "rawdata/control.pkl"))
    patients_df_upscaled.to_pickle(os.path.join(save_dir, "rawdata/patients.pkl")) 

    # Write train/test/valid
    val_size = 0.1
    test_size = args.n_test

    controls_list = controls_df_upscaled.to_numpy()
    patients_list = patients_df_upscaled.to_numpy()
    print("Creating splits for controls")
    write_numpy_data(controls_list, save_dir, val_size, test_size)
    print("Creating splits for TLEs")
    write_numpy_data(patients_list, save_dir, val_size, test_size)

    print("Creating Left and Right TLE splits")
    create_left_right_tle_splits(patients_df_upscaled, save_dir, args.metadata_path)

    print("Done!")
