# Machine-learning a semi-local exchange-correlation functional for the Hubbard model

## Description
This repository contains data, trained weights, and scripts related to the machine-learning of the exchange-correlation functional for the one-dimensional Hubbard model as outlined in our paper https://arxiv.org/abs/2501.16893.

## Contents

1. **Data**
   - `data_json`: Folder containing all of the data in a .json format
   - `exact_data`: Data computed via DMRG and reverse-engineering.
   - `test_a_data`: Exchange-correlation energies and potentials computed by inputting the exact densities to the trained model.
   - `test_b_data`: Converged Kohn-Sham results.

3. **Weights**
   - `trained_weights`: Saved in a .h5 format.

4. **Scripts**
   - `tools.py`: Contains several functions that are useful for other scripts.

5. **Jupyter Notebook**
   - `plotting.ipynb`: Reproduces figures 2,3 and 4 from the manuscript (Bethe-Ansatz LDA results yet to be added).
   - `training_example.ipynb`: Demonstrates the training of the model.
   - `kohn-sham_example.ipynb`: Demonstrates the application of a trained model to solve the Kohn-Sham equations for a set of disorder configurations.
   - `data_generation_example.ipynb`: Contains the key steps in generating the exact data.
## Installation Instructions

### Prerequisites
- [Anaconda](https://www.anaconda.com/products/distribution) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html)

### Steps

1. **Clone the repository:**
    ```bash
    git clone https://github.com/eoghancronin/ml-ldft.git
    cd ml-ldft
    ```

2. **Create a new conda environment:**
    ```bash
    conda create --name ml-ldft-env python=3.8.12
    conda activate ml-ldft-env
    ```

3. **Install the dependencies:**
    ```bash
    pip install tensorflow==2.13.0 physics-tenpy==0.10.0 matplotlib==3.5.1 scipy==1.10.1
    ```

4. **Install Jupyter Notebook:**
    ```bash
    conda install jupyter
    ```
5. **Launch Jupyter Notebook:**
    ```bash
    jupyter-notebook
    ```

## Citation

If you find this work helpful for your research, please consider citing:

https://arxiv.org/abs/2501.16893
