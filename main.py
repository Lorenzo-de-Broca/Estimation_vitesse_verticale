import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import sys
import yaml

from SRC.extract_data import extract_data

def load_input(file_path):
    with open(file_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def main(input_file = "input.yaml"):
    """
    La fonction principale qui excécute la simulation Monte Carlo
    """

    # Lecture des paramètres depuis le fichier YAML
    print(f"Loading configuration from file : {input_file}")
    config = load_input(input_file)

    data_path = config["data_path"]

    data = extract_data(data_path)
    
    print(data['time'])
    

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Parser for simulation input file.")
    
    # Définition des arguments du parser
    parser.add_argument("--file", type=str, required=False, default="input.yaml", \
        help="Nom ou chemin du fichier de configuration YAML")
    
    # Lecture des arguments
    args = parser.parse_args()
    input_file = args.file

    main(input_file)