
# Directory structure

- /networks has all the code related to the model

- /utils has utility functions to save the model and to generate images and metrics

- /env has the conda environment config

- train.py it the code to train the model
- setting.py has add the parameters needed to train the model

# Installation

`conda env create -f environment.yml`

This will create an environment `mri-gan`

# Running data preparation.

- Navigate to code/data_prep directory
- Run `python data_processing_script.py --save_dir <path>`

You will have the following files/directories created at `path`:
- `train` - directory containing the training samples. Each is a (256,256) image of a single slice saved as a numpy array
- `test` - directory containing the test samples. Each is a (256,256) image of a single slice saved as a numpy array
- `val` - directory containing the validation samples. Each is a (256,256) image of a single slice saved as a numpy array
- `rawdata/control.pkl` - All Control records as a pandas dataframe
- `rawdata/patients.pkl` - All TLE records as a pandas dataframe
- `rawdata/left_TL.pkl` - All left TLE records as a pandas dataframe
- `rawdata/right_TL.pkl` - All right TLE records as a pandas dataframe

> Note all the pandas dataframe have format as:

    # 'File' - Name of the file

    # 'image' - a 3 dimension NP array representation of the image

    # 'upscaled_image' - a 2 dimension (256,256) image of the specified slice

# Running the training code.

- Extract sample_output.zip
- Verify all the settings in settings.py
- Verify the GPU number in train.py
- run `python train.py`

Note `train.py` automatically load the latest checkpoint from sample_output/models, if you do not want to do that empty this directory or comment to code to load weights `save_found = gan.load_weights(os.path.join(LOAD_DIR, 'models'))`

# Running the visualizations.

- Navigate to /visuals

- For Age related continous plots
`python continous_data_visual.py --save_dir <absolute dir path where plots will be save> --perm`

- For Control vs TLE plots
`python control_TLE.py --save_dir <absolute dir path where plots will be save>`

- For Left vs Right plots
`python left_right.py --save_dir <absolute dir path where plots will be save>`