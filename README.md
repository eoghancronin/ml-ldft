# Machine-learning a semi-local Hartree-exchange-correlation functional for the Hubbard model

This repository contains data, trained weights, and scripts related to the machine-learning of the Hartree-exchange-correlation functional for the one-dimensional Hubbard model as outlined in our paper [doi here].

## Contents

1. **Data**
   - `exact_data`: Data computed via DMRG and reverse-engineering.
   - `test_a_data`: Hxc energies and potentials computed by inputting the exact densities to the trained model.
   - `test_b_data`: Converged Kohn-Sham results.

2. **Weights**
   - `trained_weights`: Saved in a .h5 format.

3. **Scripts**
   - `training_example.py`: Python script demonstrating the training of the model.
   - `kohn-sham_example.py`: Demonstrates the application of a trained model to solve the Kohn-Sham equations for a set of disorder configurations.
   - `tools.py`: Contains several functions useful for the other scripts.
   - `data_generation_example.py`: Contains the key steps in generating the exact data. Note that this script requires an installation of TenPy.

4. **Jupyter Notebook**
   - `plotting.ipynb`: Jupyter notebook reproducing figures 2,3 and 4 from the manuscript.

## Usage

To use this repository, follow these steps:

1. Clone the repository to your local machine:
git clone https://github.com/eoghancronin/ml-ldft.git

2. Navigate to the repository directory:
cd ml-ldft

3. Explore the folders and files containing data, trained weights, scripts, and notebooks.

4. Run the `training_example.py` to train the model.

5. Open and execute the `plotting.ipynb` to reproduce the plots discussed in the manuscript.

## Requirements

The script and notebook are confirmed to be working with the following:

- Python (version 3.8.12)
- NumPy (version 1.19.5)
- TensorFlow (version 2.13.0)
- Jupyter Notebook (version 6.4.11)

## Citation

If you find this work helpful for your research, please consider citing:

[doi here]
