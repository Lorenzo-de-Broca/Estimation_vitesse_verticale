import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import sys
from sklearn.ensemble import RandomForestRegressor
import yaml
from sklearn.linear_model import LinearRegression
from tensorflow import keras
import tensorflow as tf
import math
from tensorflow.keras.callbacks import EarlyStopping

from SRC.extract_data import extract_data, create_reg_array3, create_combined_regression_array, create_combined_regression_array_delta_t, create_train_test_matrix, create_PCA, create_data_neuronal
from SRC.filtre_convection import create_convection_filter
from SRC.regression import multi_lin_reg, test_model, random_forest_reg
from SRC.plots import plot_real_velocity_map, plot_reconstructed_velocity_map, plot_test_model, plot_residuals, plot_velocity_comparison, plot_difference_velocity_map, plot_losses


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
    
    n_trees = config.get("n_trees", 10)  # Default to 10 if not specified
    
    if compute_random_forest :
        title = f"Random_Forest ({n_trees} trees) ({training_ratio*100} % train)"
        output_dir = f"figures/Random_Forest_{n_trees}_trees_{training_ratio*100}_percent_train"
        compute_PCA = False
        model_name = "Random Forest"
    elif compute_delta_T :
        title = f"Delta_T ({training_ratio*100} % train)"
        output_dir = f"figures/Delta_T_{training_ratio*100}_percent_train"
        model_name = "Multiple Linear Regression"
    else :
        title = f"Absolute_T ({training_ratio*100} % train)"
        output_dir = f"figures/Absolute_T_{training_ratio*100}_percent_train"
        model_name = "Multiple Linear Regression"
    
    if compute_PCA :
        pca_components = config["PCA_components"]
    elif compute_random_forest :
        pca_components = 0
    else :
        pca_components = 0
    
    if compute_PCA :
        title += f"_PCA_{pca_components}"
        output_dir += f"_PCA_{pca_components}"
    
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
        x_test, y_test = create_combined_regression_array_delta_t (data, filter, test_matrix)
    
    else :
        print("Using absolute T for regression.")
        x_data, y_data = create_combined_regression_array (data, filter=np.ones((88,500,500)), train_matrix=np.ones((500,500)))
        x_train, y_train = create_combined_regression_array (data, filter, train_matrix)
        x_test, y_test = create_combined_regression_array (data, filter, test_matrix)
    
    print("Regression arrays created.") 
    print(f"x_train shape: {x_train.shape}")
    print(f"y_train shape: {y_train.shape}")

    if compute_random_forest :
        print("Training a Random Forest Regressor ...")
        model = random_forest_reg (x_train, y_train, n_estimators=n_trees)
        
    else :
        print("Training a Multiple Linear Regression model ...")
        if compute_PCA :
            print("Computing PCA ...")
            X_pca = create_PCA (x_train, y_train, pca_components)
            X_test_pca = create_PCA (x_test, y_test, pca_components)
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
        y_train_pred, rmse_train, residuals_train, r2_train = test_model (model, x_train, y_train, model_name=model_name)
        y_test_pred, rmse_test, residuals_test, r2_test = test_model (model, x_test, y_test, model_name=model_name)
        y_all_data_pred, rmse_all, residuals_all, r2_all = test_model (model, x_data, y_data, model_name=model_name)

    else :
        if compute_PCA :
            y_train_pred, rmse_train, residuals_train, r2_train = test_model (model, X_pca, y_train, model_name=model_name)
            y_test_pred, rmse_test, residuals_test, r2_test = test_model (model, X_test_pca, y_test, model_name=model_name)
            y_all_data_pred, rmse_all, residuals_all, r2_all = test_model (model, X_all_data_pca, y_data, model_name=model_name)

        else :
            y_train_pred, rmse_train, residuals_train, r2_train = test_model (model, x_train, y_train, model_name=model_name)
            y_test_pred, rmse_test, residuals_test, r2_test = test_model (model, x_test, y_test, model_name=model_name)
            y_all_data_pred, rmse_all, residuals_all, r2_all = test_model (model, x_data, y_data, model_name=model_name)

    print("Model tested successfully.")
    print(f"RMSE on training set: {rmse_train}")
    print(f"R² on training set: {r2_train}")
    
    print(f"RMSE on test set: {rmse_test}")
    print(f"R² on test set: {r2_test}")
    
    print(f"RMSE on all data: {rmse_all}")
    print(f"R² on all data: {r2_all}")

    print("Plotting results ...")
    
    # Créer le dossier pour les résultats avec chemin absolu
    
    print(f"Creating output directory at: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    
    plot_test_model (y_train, y_train_pred, title, R2=r2_train, rmse=rmse_train, data_set='train', output_dir=output_dir)
    plot_test_model (y_test, y_test_pred, title, R2=r2_test, rmse=rmse_test, data_set='test', output_dir=output_dir)
    print("comparisons plotted")
    
    plot_residuals (y_train_pred, residuals_train, title, data_set='train', output_dir=output_dir)
    plot_residuals (y_test_pred, residuals_test, title, data_set='test', output_dir=output_dir)
    print("residuals plotted")

    if compute_delta_T :
        plot_velocity_comparison(x_data, y_data.reshape(87,500,500), y_all_data_pred.reshape(87,500,500), filter, title=title, t=0, output_dir=output_dir)
        plot_difference_velocity_map(y_data.reshape(87,500,500), y_all_data_pred.reshape(87,500,500), filter, title, output_dir=output_dir)
    else :
        plot_velocity_comparison(x_data, y_data.reshape(88,500,500), y_all_data_pred.reshape(88,500,500), filter, title=title, t=0, output_dir=output_dir)
        plot_difference_velocity_map(y_data.reshape(88,500,500), y_all_data_pred.reshape(88,500,500), filter, title, output_dir=output_dir)
    

def neuronal_network(input_file = "inputs/inputs.yaml",paths_file = "inputs/paths.yaml"):
    """
    La fonction qui excécute la régression par réseau de neuronnes pour estimer la vitesse verticale
    """
    
    # Lecture des paramètres depuis le fichier YAML
    print(f"Loading input configuration from file : {input_file}")
    config = load_input(input_file)
    print(f"Loading paths configuration from file : {paths_file}")
    paths = load_input(paths_file)

    data_path = paths["data_file"]
    
    pop_size = config["pop_size"]
    train_ratio = config["train_ratio"]
    
    print(f"Population used for Neural Network training : {pop_size}")
    print(f"Training ratio set to : {train_ratio}")
    
    # Extraction des données
    print(f"Extraction of data from file : {data_path}")
    data = extract_data()
    print("Data extracted successfully.")

    print("Creating the convection filter ...")
    filter = create_convection_filter()
    print("Filter created successfully.")

    print("Creating regression arrays for Neuronal Network ...")
    x_data_train, W_filtered_train, x_data_test, W_filtered_test, x_data_all, W_filtered_all = create_data_neuronal(data, filter, train_ratio, pop_size)
    print("Regression arrays created.") 
    print(f"x_data_train shape: {x_data_train.shape}")
    print(f"W_filtered_train shape: {W_filtered_train.shape}")  

    learning_rate = config["learning_rate"]
    print("Building the Neural Network model ...")
    model = keras.Sequential()
    model.add(keras.layers.Dense(units = 128, activation = 'relu', input_shape=x_data_train.shape[1:]))
    model.add(keras.layers.Dense(units = 128, activation = 'relu'))
    model.add(keras.layers.Dense(units = 128, activation = 'relu'))
    model.add(keras.layers.Dense(units = 128, activation = 'relu'))
    model.add(keras.layers.Dense(units = 1, activation = 'linear'))
    model.compile(loss='mae', optimizer=tf.optimizers.Adam(learning_rate=learning_rate))
    print(model.summary())
    print("learning rate = ", learning_rate)
    print("Model compiled successfully.")

    print("Setting up early stopping ...")
    early_stop = EarlyStopping(
    monitor=config["monitor_metric"],                   # Monitor validation loss
    patience=config["nn_early_stopping_patience"],      # Stop if no improvement for 10 epochs
    restore_best_weights=True,                          # Restore weights from best epoch
    verbose=1                                           # Print when stopping
    )
    epochs=config["nn_epochs"]
    print(f"Early stopping monitor: {config['monitor_metric']}, patience: {config['nn_early_stopping_patience']}, epochs: {epochs}")
    
    print("Fitting the model ...")
    history = model.fit(x_data_train, W_filtered_train, epochs=epochs, verbose=1, validation_data=(x_data_test, W_filtered_test), callbacks=[early_stop])
    print("Model trained successfully.")

    print("Predicting with the model ...")
    W_predicted_test = model.predict(x_data_test)
    W_predicted_train = model.predict(x_data_train)
    W_predicted_all = model.predict(x_data_all)
    print("Prediction completed.")
    score = model.evaluate(x_data_test, W_filtered_test, verbose=1)
    print(f"Model evaluation score on test set: {score}")

    y_train_pred, rmse_train, residuals_train, r2_train = test_model (model, x_data_train, W_filtered_train, model_name="Neuronal Network")
    y_test_pred, rmse_test, residuals_test, r2_test = test_model (model, x_data_test, W_filtered_test, model_name="Neuronal Network")
    # y_all_data_pred, rmse_all, residuals_all, r2_all = test_model (model, x_data_pred, W_filtered_pred[:,0], model_name="Neuronal Network")

    print("Model tested successfully.")
    print(f"RMSE on training set: {rmse_train}")
    print(f"R² on training set: {r2_train}")
    print(f"RMSE on test set: {rmse_test}")
    print(f"R² on test set: {r2_test}")

    print("Plotting results ...")
    output_dir = f"figures/Neuronal_Network_{pop_size}_population_lr_{learning_rate}"
    os.makedirs(output_dir, exist_ok=True)

    plot_losses(history, score=score, output_dir=output_dir)

    plot_test_model(W_filtered_train, W_predicted_train[:,0], title=f"Neuronal Network (training {100*train_ratio:.0f}% on {pop_size} population)", R2=r2_train, rmse=rmse_train, data_set='train', output_dir=output_dir)
    plot_test_model(W_filtered_test, W_predicted_test[:,0], title=f"Neuronal Network (training {100*train_ratio:.0f}% on {pop_size} population)", R2=r2_test, rmse=rmse_test, data_set='test', output_dir=output_dir)
    print("comparisons plotted")

    title = f"Neuronal_Network (training {100*train_ratio:.0f}% on {pop_size} population)"
    plot_residuals (y_train_pred, residuals_train, title, data_set='train', output_dir=output_dir)
    plot_residuals (y_test_pred, residuals_test, title, data_set='test', output_dir=output_dir)
    print("residuals plotted")

    compute_delta_T = True
    if compute_delta_T :
        plot_velocity_comparison(x_data_all, W_filtered_all.reshape((500,500)), W_predicted_all.reshape((500,500)), filter, title=title, t=100, output_dir=output_dir)
        plot_difference_velocity_map(W_filtered_all.reshape((500,500)), W_predicted_all.reshape((500,500)), filter, title, t=100, output_dir=output_dir)
    else :
        plot_velocity_comparison(x_data_all, W_filtered_all, W_predicted_all, filter, title=title, t=100, output_dir=output_dir)
        plot_difference_velocity_map(W_filtered_all.reshape((500,500)), W_predicted_all.reshape((500,500)), filter, title, t=100, output_dir=output_dir)

# x_data, y_data, y_data_pred, filter, title, t, output_dir = x_data_all, W_filtered_all.reshape((500,500)), W_predicted_all.reshape((500,500)), filter, title, 100, output_dir

# model, X_test, y_test, model_name = model, x_data_train, W_filtered_train, "Neuronal Network"

# y_pred, residuals, title, data_set, output_dir = y_train_pred, residuals_train, title, 'train', output_dir
# y_pred, residuals, title, data_set, output_dir = x_data, y_data.reshape(87,500,500), y_all_data_pred.reshape(87,500,500), filter, title=title, t=0, output_dir=output_dir

# y_data,y_pred,title,R2,rmse,data_set,output_dir = W_filtered_train, W_predicted_train.data, f"Neuronal Network ({pop_size} population) - Train", r2_train, rmse_train, 'train', output_dir

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