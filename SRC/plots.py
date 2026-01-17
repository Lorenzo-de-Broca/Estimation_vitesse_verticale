import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import os
import sys 

alpha = 0.1
eps = -1e-4

def create_c_maps():
    """Create custom colormaps for plotting
    Returns:
        cmapb, cmapw: colormaps for black and white masks
    """
    cmapb = mpl.colors.ListedColormap(['None','black','None'])
    cmapw = mpl.colors.ListedColormap(['None','white','None'])
    bounds=[-1,-0.1,0.1,1]
    norm = mpl.colors.BoundaryNorm(bounds, cmapb.N)
    
    max = 0.1
    norm_plot = mpl.colors.Normalize(vmin=-max, vmax=max)

    cmapb_pos = mpl.colors.ListedColormap(['black','None'])
    norm_pos = mpl.colors.BoundaryNorm([-1,eps,1], cmapb_pos.N)
    norm_strict_pos = mpl.colors.BoundaryNorm([-1,0,1], cmapb_pos.N)
    
    return cmapb, cmapw, norm, norm_plot, norm_pos, norm_strict_pos, cmapb_pos


def plot_filter (x_data, filter, train_matrix, t=0) :

    cmapb, cmapw, norm, norm_plot, norm_pos, norm_strict_pos, cmapb_pos = create_c_maps()
    
    plt.figure()
  
    plt.imshow(x_data[t,:,:], origin='lower', cmap='viridis', norm=norm_plot)
    plt.colorbar()
    plt.imshow(train_matrix, origin='lower', cmap=cmapb, norm=norm)
    plt.imshow(filter[t,:,:]*filter[t+1,:,:], origin='lower', cmap=cmapw, norm=norm)
    plt.title(f'Training data points overlayed on Δaos_1830BT at t={t}')
    print("begin fit")



def plot_test_model (y_data,y_pred,title,R2,rmse,data_set='train') :
    """Plot the true vs predicted values for the regression model
    Args:
        y_data (np.array): 1D array (n_samples,) of true output data
        y_pred (np.array): 1D array (n_samples,) of predicted output data
    """
    
    plt.figure()
    n = 100  # garder 1 point sur 100
    indices = np.arange(len(y_data))[::n]
    y_min, y_max = y_data[indices].min(), y_data[indices].max()
    plt.scatter(y_data[indices], y_pred[indices], color = 'darkblue', alpha=alpha, label=f'R2 = {R2:.2f}, RMSE = {rmse:.2e}')
    plt.plot([y_min, y_max], [y_min, y_max], 'k--')
    plt.plot([y_min, y_max], [0, 0], 'r--')
    plt.plot([0, 0], [y_min, y_max], 'r--')
    plt.xlabel("Vitesse verticale vraie [m/s]")
    plt.ylabel("Vitesse verticale prédite [m/s]")
    plt.title(f"Régression multinéaire - {title} on {data_set} set")
    plt.legend()
    
    plt.savefig(f'figures/Data_Vs_Prediction_{title}_{data_set}.png', dpi=300, bbox_inches='tight')
    #plt.show()
    plt.close()


def plot_residuals (y_pred, residuals, title, data_set='train') :
    """Plot the residuals of the regression model
    Args:
        residuals (np.array): 1D array (n_samples,) of residuals
    """
    plt.figure()
    plt.hist(residuals, bins=1000, alpha=0.5)
    plt.xlabel("Résidus [m/s]")
    plt.ylabel("Fréquence")
    plt.title(f"Histogramme des résidus - {title} on {data_set} set")
    plt.savefig(f'figures/Residual_histograms_{title}_{data_set}.png', dpi=300, bbox_inches='tight')

    #plt.show()
    plt.close()
    
    plt.figure()
    n = 100  # garder 1 point sur 100
    indices = np.arange(len(y_pred))[::n]
    plt.scatter(y_pred[indices], residuals[indices], color = 'darkblue', alpha=alpha)
    plt.axhline(0, color='k', linestyle='--')
    plt.xlabel("Valeur prédite [m/s]")
    plt.ylabel("Résidu [m/s]")
    plt.title(f"Analyse des résidus - {title} on {data_set} set")
    plt.savefig(f'figures/Residual_analysis_{title}_{data_set}.png', dpi=300, bbox_inches='tight')

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
    
    #plt.show()
    plt.close()
    
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
    
    #plt.show()
    plt.close()

def plot_velocity_comparison(x_data, y_data, y_data_pred, filter, title, t=0):
    """Plot measured and predicted velocity maps side by side with a shared colorbar
    
    Args:
        y_data (np.array): 3D array of true vertical velocity data
        x_data (np.array): Input features for prediction
        model: Trained regression model
        filter (np.array): 3D array of filter mask
        t (int): Time index to plot (default: 0)
    """
    cmapb, cmapw, norm, norm_plot, norm_pos, norm_strict_pos, cmapb_pos = create_c_maps()
    
    # Create figure with 2 subplots side by side
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left plot: Measured velocity
    im1 = axes[0].imshow(y_data[t, :, :], origin='lower', cmap='viridis', norm=norm_plot)
    axes[0].imshow(filter[t, :, :], origin='lower', cmap=cmapw, norm=norm)
    axes[0].set_title(f'Measured velocity at t={t}', fontsize=12)
    axes[0].set_xlabel('X')
    axes[0].set_ylabel('Y')
    
    # Right plot: Predicted velocity
    im2 = axes[1].imshow(y_data_pred[t, :, :], origin='lower', cmap='viridis', norm=norm_plot)
    axes[1].imshow(filter[t, :, :], origin='lower', cmap=cmapw, norm=norm)
    axes[1].set_title(f'Predicted velocity at t={t}', fontsize=12)
    axes[1].set_xlabel('X')
    axes[1].set_ylabel('Y')
    
    # Add a single colorbar for both subplots
    cbar = fig.colorbar(im2, ax=axes, orientation='vertical', pad=0.03, shrink=0.8)
    cbar.set_label('Vertical velocity [m/s]', fontsize=11)
    
    #plt.tight_layout()

    plt.suptitle(f'Velocity compararison - {title}', fontsize=14)
    #plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    plt.savefig(f'figures/Velocity_comparison_at_t_{t}_{title}.png', dpi=300, bbox_inches='tight')
    
    #plt.show()
    plt.close()

def plot_difference_velocity_map(w_data, w_pred_tot, filter, title, t=0):
    
    cmapb, cmapw, norm, norm_plot, norm_pos, norm_strict_pos, cmapb_pos = create_c_maps()

    # Create mask for negative values of the product
    product = w_data[t,:,:]*w_pred_tot[t,:,:]
    strict_negative_mask = np.zeros_like(product)
    strict_negative_mask[product < 0] = 1
    negative_mask = np.zeros_like(product)
    negative_mask[product < eps] = 1
    
    ratio_negative = negative_mask.sum()/(negative_mask.shape[0]*negative_mask.shape[1])
    print(f"{ratio_negative*100:.2f} % of points with opposite signs between true and predicted velocities at t={t}")
    ratio_strict_negative = strict_negative_mask.sum()/(strict_negative_mask.shape[0]*strict_negative_mask.shape[1])
    print(f"{ratio_strict_negative*100:.2f} % of points with strictly opposite signs between true and predicted velocities at t={t}")

    # Create figure with 2 subplots side by side
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Left plot: negative_mask with norm_pos
    axes[0].imshow(w_data[t,:,:], origin='lower', cmap='Spectral', norm=norm_plot, alpha=0.6)
    axes[0].imshow(product, origin='lower', cmap=cmapb_pos, norm=norm_pos)
    axes[0].imshow(filter[t,:,:]*filter[t+1,:,:], origin='lower', cmap=cmapw, norm=norm)
    axes[0].set_title(f'Product < 1e-2 at t={t} (ratio={ratio_negative*100:.2f}%)', fontsize=12)
    axes[0].set_xlabel('X')
    axes[0].set_ylabel('Y')
    
    # Right plot: strict_negative_mask with norm_strict_pos
    im = axes[1].imshow(w_data[t,:,:], origin='lower', cmap='Spectral', norm=norm_plot, alpha=0.6)
    axes[1].imshow(product, origin='lower', cmap=cmapb_pos, norm=norm_strict_pos)
    axes[1].imshow(filter[t,:,:]*filter[t+1,:,:], origin='lower', cmap=cmapw, norm=norm)
    axes[1].set_title(f'Product < 0 (strictly negative) at t={t} (ratio={ratio_strict_negative*100:.2f}%)', fontsize=12)
    axes[1].set_xlabel('X')
    axes[1].set_ylabel('Y')
    
    # Add a single colorbar
    cbar = fig.colorbar(im, ax=axes, orientation='vertical', pad=0.03, shrink=0.8)
    cbar.set_label('W velocity [m/s]', fontsize=11)

    plt.suptitle(f'Difference velocity - {title}', fontsize=14)
    #plt.subplots_adjust(left=0.07, right=0.9, top=0.8, bottom=0.1, wspace=0.3)
    plt.savefig(f'figures/Difference_velocity_map_t_{t}_{title}.png', dpi=300, bbox_inches='tight')
    
    #plt.show()
    plt.close()
    
    