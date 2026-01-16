import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import os
import sys 

alpha = 0.01

def plot_filter (x_data, filter, train_matrix) :
    t=0
    plt.figure()
    norm1 = mpl.colors.Normalize(vmin=-0.3, vmax=0.1)
    cmapb = mpl.colors.ListedColormap(['None','black','None'])
    cmapw = mpl.colors.ListedColormap(['None','white','None'])
    bounds=[-1,-0.1,0.1,1]
    norm = mpl.colors.BoundaryNorm(bounds, cmapb.N)
    plt.imshow(x_data[t,:,:], origin='lower', cmap='viridis', norm=norm1)
    plt.colorbar()
    plt.imshow(train_matrix, origin='lower', cmap=cmapb, norm=norm)
    plt.imshow(filter[t,:,:]*filter[t+1,:,:], origin='lower', cmap=cmapw, norm=norm)
    plt.title(f'Training data points overlayed on Δaos_1830BT at t={t}')
    print("begin fit")



def plot_test_model (y_data,y_pred,pca_components) :
    """Plot the true vs predicted values for the regression model
    Args:
        y_data (np.array): 1D array (n_samples,) of true output data
        y_pred (np.array): 1D array (n_samples,) of predicted output data
    """
    
    plt.figure()
    n = 100  # garder 1 point sur 100
    indices = np.arange(len(y_data))[::n]
    plt.scatter(y_data[indices], y_pred[indices], color = 'darkblue', alpha=alpha)
    plt.plot([y_data.min(), y_data.max()], [y_data.min(), y_data.max()], 'k--')
    plt.xlabel("Vitesse verticale vraie [m/s]")
    plt.ylabel("Vitesse verticale prédite [m/s]")
    plt.title(f"Régression multinéaire (PCA - {int(pca_components)}) - Entraînement")

    plt.savefig(f'figures/Data_Vs_Prediction_PCA_components_{int(pca_components)}.png', dpi=300, bbox_inches='tight')

    #plt.show()
    plt.close()

def plot_residuals (y_pred, residuals, pca_components) :
    """Plot the residuals of the regression model
    Args:
        residuals (np.array): 1D array (n_samples,) of residuals
    """
    plt.figure()
    plt.hist(residuals, bins=1000, alpha=0.5)
    plt.xlabel("Résidus [m/s]")
    plt.ylabel("Fréquence")
    plt.title(f"Histogramme des résidus (PCA - {int(pca_components)})")
    plt.savefig(f'figures/Residual_histograms_PCA_components_{int(pca_components)}.png', dpi=300, bbox_inches='tight')

    #plt.show()
    plt.close()
    
    plt.figure()
    n = 100  # garder 1 point sur 100
    indices = np.arange(len(y_pred))[::n]
    plt.scatter(y_pred[indices], residuals[indices], color = 'darkblue', alpha=alpha)
    plt.axhline(0, color='k', linestyle='--')
    plt.xlabel("Valeur prédite [m/s]")
    plt.ylabel("Résidu [m/s]")
    plt.title(f"Analyse des résidus (PCA - {int(pca_components)})")
    plt.savefig(f'figures/Residual_analysis_PCA_components_{int(pca_components)}.png', dpi=300, bbox_inches='tight')

    #plt.show()
    plt.close()
    
def plot_reconstructed_velocity_map (x_data, model, filter):
    """Plot the reconstructed map of vertical velocity
    """
    t=0
    coeffs = model.coef_
    print("model coefficients:", coeffs)
    print("model coefficient shape:", coeffs.shape)
    plt.figure()
    norm1 = mpl.colors.Normalize(vmin=-1, vmax=1)
    cmapb = mpl.colors.ListedColormap(['None','black','None'])
    cmapw = mpl.colors.ListedColormap(['None','white','None'])
    bounds=[-1,-0.1,0.1,1]
    norm = mpl.colors.BoundaryNorm(bounds, cmapb.N)
    print("x_data shape:", x_data.shape)
    y_data_pred = model.predict(x_data).reshape(88,500,500)
    print("y_data_pred shape:", y_data_pred.shape)
    plt.imshow(y_data_pred[t,:,:], origin='lower', cmap='viridis',norm=norm1)
    plt.colorbar()
    plt.imshow(filter[t,:,:], origin='lower', cmap=cmapw, norm=norm)
    plt.title(f'Predicted velocity at t={t}')
    plt.savefig(f'figures/Predicted_velocity_at_t_{t}.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_real_velocity_map (y_data, filter):
    """Plot the reconstructed map of vertical velocity
    """
    t=0
    
    plt.figure()
    norm1 = mpl.colors.Normalize(vmin=-1, vmax=1)
    cmapb = mpl.colors.ListedColormap(['None','black','None'])
    cmapw = mpl.colors.ListedColormap(['None','white','None'])
    bounds=[-1,-0.1,0.1,1]
    norm = mpl.colors.BoundaryNorm(bounds, cmapb.N)
    print("y_data shape:", y_data.shape)
    plt.imshow(y_data[t,:,:], origin='lower', cmap='viridis', norm=norm1)
    plt.colorbar()
    plt.imshow(filter[t,:,:], origin='lower', cmap=cmapw, norm=norm)
    plt.title(f'Measured velocity at t={t}')
    plt.savefig(f'figures/Measured_velocity_at_t_{t}.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_velocity_comparison(x_data, y_data, y_data_pred, filter, t=0):
    """Plot measured and predicted velocity maps side by side with a shared colorbar
    
    Args:
        y_data (np.array): 3D array of true vertical velocity data
        x_data (np.array): Input features for prediction
        model: Trained regression model
        filter (np.array): 3D array of filter mask
        t (int): Time index to plot (default: 0)
    """
    # Get predictions
    #y_data_pred = model.predict(x_data).reshape(88, 500, 500)
    
    # Create figure with 2 subplots side by side
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Define normalization and colormaps
    norm1 = mpl.colors.Normalize(vmin=-0.1, vmax=0.1)
    cmapw = mpl.colors.ListedColormap(['None', 'white', 'None'])
    bounds = [-1, -0.1, 0.1, 1]
    norm_mask = mpl.colors.BoundaryNorm(bounds, cmapw.N)
    
    # Left plot: Measured velocity
    im1 = axes[0].imshow(y_data[t, :, :], origin='lower', cmap='viridis', norm=norm1)
    axes[0].imshow(filter[t, :, :], origin='lower', cmap=cmapw, norm=norm_mask)
    axes[0].set_title(f'Measured velocity at t={t}', fontsize=12)
    axes[0].set_xlabel('X')
    axes[0].set_ylabel('Y')
    
    # Right plot: Predicted velocity
    im2 = axes[1].imshow(y_data_pred[t, :, :], origin='lower', cmap='viridis', norm=norm1)
    axes[1].imshow(filter[t, :, :], origin='lower', cmap=cmapw, norm=norm_mask)
    axes[1].set_title(f'Predicted velocity at t={t}', fontsize=12)
    axes[1].set_xlabel('X')
    axes[1].set_ylabel('Y')
    
    # Add a single colorbar for both subplots
    cbar = fig.colorbar(im2, ax=axes, orientation='vertical', pad=0.1, shrink=0.8)
    cbar.set_label('Vertical velocity [m/s]', fontsize=11)
    
    #plt.tight_layout()
    plt.savefig(f'figures/Velocity_comparison_at_t_{t}.png', dpi=300, bbox_inches='tight')
    plt.show()
