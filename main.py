import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import sys
from sklearn.ensemble import RandomForestRegressor
import yaml
from sklearn.linear_model import LinearRegression

from SRC.extract_data import extract_data, create_reg_array3, create_combined_regression_array, create_combined_regression_array_delta_t, create_train_test_matrix, create_PCA
from SRC.filtre_convection import create_convection_filter
from SRC.regression import multi_lin_reg, test_model, random_forest_reg
from SRC.plots import plot_real_velocity_map, plot_reconstructed_velocity_map, plot_test_model, plot_residuals, plot_velocity_comparison


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
    
    training_ratio = config["training_ratio"]
    
    compute_random_forest = config["compute_random_forest"]
    
    compute_PCA = config["compute_PCA"]
        
    compute_delta_T = config["compute_delta_T"]
    
    if compute_random_forest :
        title = "Random_Forest"
        compute_PCA = False
    elif compute_delta_T :
        title = "Delta T"
    else :
        title = "Absolute T"
    
    if compute_PCA :
        pca_components = config["PCA_components"]
    elif compute_random_forest :
        pca_components = 0
    else :
        pca_components = 0
    
    if compute_PCA :
        title += f"_PCA_{pca_components}"
    
    print(f"Training ratio set to : {training_ratio}")
    
    # Extraction des données
    print(f"Extraction of data from file : {data_path}")
    data = extract_data()
    print("Data time : ")
    print(data['time'])
    print("Data extracted successfully.")
    
    # Filtrage des données pour éviter les zones de convetion    
    print("Creating a filter ...")
    filter = create_convection_filter()
    train_matrix, test_matrix = create_train_test_matrix (train_ratio=training_ratio)
    print("Filter created successfully.")
    
    print("Filtering the data and creating regression arrays ...")
    
    if compute_delta_T :
        print("Using Delta T for regression.")
        x_data, y_data = create_combined_regression_array_delta_t (data, filter=np.ones((88,500,500)), train_matrix=np.ones((500,500)))
        x_train, y_train = create_combined_regression_array_delta_t (data, filter, train_matrix)
        #x_test, y_test = create_combined_regression_array_delta_t (data, filter, test_matrix)
    
    else :
        print("Using absolute T for regression.")
        x_data, y_data = create_combined_regression_array (data, filter=np.ones((88,500,500)), train_matrix=np.ones((500,500)))
        x_train, y_train = create_combined_regression_array (data, filter, train_matrix)
        #x_test, y_test = create_combined_regression_array (data, filter, test_matrix)
    
    print("Regression arrays created.") 
    print(f"x_train shape: {x_train.shape}")
    print(f"y_train shape: {y_train.shape}")

    if compute_random_forest :
        print("Training a Random Forest Regressor ...")
        model = random_forest_reg (x_train, y_train)
        
    else :
        print("Training a Multiple Linear Regression model ...")
        if compute_PCA :
            print("Computing PCA ...")
            X_pca = create_PCA (x_train, y_train, pca_components)
            X_all_data_pca = create_PCA (x_data, y_data, pca_components)
            print("PCA computed.") 
            model = multi_lin_reg (X_pca, y_train)    
        else:
            model = multi_lin_reg (x_train, y_train)
    
    print("Model trained successfully.")
    
    print("Testing the model ...")
    print("x_train shape for testing:", x_train.shape)
    print("y_train shape for testing:", y_train.shape)
    
    if compute_random_forest :
        y_pred, rmse, residuals, r2 = test_model (model, x_train, y_train)
        y_all_data_pred, rmse_all, residuals_all, r2_all = test_model (model, x_data, y_data)
    
    else :
        if compute_PCA :
            y_pred, rmse, residuals, r2 = test_model (model, X_pca, y_train)
            y_all_data_pred, rmse_all, residuals_all, r2_all = test_model (model, X_all_data_pca, y_data)
        
        else :
            y_pred, rmse, residuals, r2 = test_model (model, x_train, y_train)
            y_all_data_pred, rmse_all, residuals_all, r2_all = test_model (model, x_data, y_data)
        
    print(f"RMSE: {rmse}")
    print(f"R²: {r2}")

    print("Plotting results ...")
    plot_test_model (y_train, y_pred, title)
    
    print("comparison plotted")
    plot_residuals (y_pred, residuals, title)
    print("residuals plotted")

    if compute_delta_T :
        plot_velocity_comparison(x_data, y_data.reshape(87,500,500), y_all_data_pred.reshape(87,500,500), filter, title=title, t=0)
    else :
        plot_velocity_comparison(x_data, y_data.reshape(88,500,500), y_all_data_pred.reshape(88,500,500), filter, title=title, t=0)


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