import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import sys
import yaml

from SRC.extract_data import extract_data, create_combined_regression_array
from SRC.filtre_convection import create_convection_filter

def load_input(file_path):
    with open(file_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def main(input_file = "inputs/inputs.yaml",paths_file = "inputs/paths.yaml"):
    """
    La fonction principale qui excécute la régression linéaire pour estimer la vitesse verticale
    """

    # Lecture des paramètres depuis le fichier YAML
    print(f"Loading input configuration from file : {input_file}")
    config = load_input(input_file)
    print(f"Loading paths configuration from file : {paths_file}")
    paths = load_input(paths_file)

    data_path = paths["data_file"]

    # Extraction des données
    print(f"Extraction of data from file : {data_path}")
    data = extract_data()
    print(data['time'])
    print("Data extracted successfully.")
    
    # Filtrage des données pour éviter les zones de convetion    
    print("Creating a filter ...")
    filter = create_convection_filter()
    print("Filter created successfully.")
    
    print("Filter the data and create regression arrays")
    
    x_data, y_data = create_combined_regression_array (data, filter)
    print("Regression arrays created.") 
    print(f"x_data shape: {x_data.shape}")
    print(f"y_data shape: {y_data.shape}") 

    

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Parser for simulation input file.")
    
    # Définition des arguments du parser
    parser.add_argument("--input", type=str, required=False, default="inputs/inputs.yaml", \
        help="Nom ou chemin du fichier de configuration YAML")
    parser.add_argument("--paths", type=str, required=False, default="inputs/paths.yaml", \
        help="Nom ou chemin du fichier de configuration YAML")
    
    # Lecture des arguments
    args = parser.parse_args()
    input_file = args.input
    paths_file = args.paths
    
    # Appel de la fonction principale
    main(input_file, paths_file)