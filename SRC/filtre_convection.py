import numpy as np
import matplotlib.pyplot as plt
from SRC.extract_data import extract_data
import matplotlib as mpl

def create_convection_filter():
    """creates a filter to avoid convection zones. Mask where TB (183+/-1) is higher than TB (183+/-7).

    Returns:
        filter (np.array): 3D array (time, lat, lon) with 1 where no convection and 0 where convection.
    """
    frame = extract_data()
    p1 = np.array(frame['aos_1830BT'][:,:,:])
    p7 = np.array(frame['aos_1837BT'][:,:,:])

    filter = np.ones(np.shape(p1))
    filter[p1>p7] = 0

    filter = filter*(np.ones(np.shape(p1))-np.ma.getmask(frame['W_at_BT']))

    return filter


# %% figure test
def plot_filter():
    """plots the convection filter at different time steps.
    """
    filter = create_convection_filter()

    fig, ax = plt.subplots(3,3, layout='constrained')
    ax[0,0].imshow(filter[0,:,:], origin='lower', cmap='Greys',  interpolation='nearest')
    ax[0,0].set_title('t=0')
    ax[0,1].imshow(filter[10,:,:], origin='lower', cmap='Greys',  interpolation='nearest')
    ax[0,1].set_title('t=10')
    ax[0,2].imshow(filter[20,:,:], origin='lower', cmap='Greys',  interpolation='nearest')
    ax[0,2].set_title('t=20')

    ax[1,0].imshow(filter[30,:,:], origin='lower', cmap='Greys',  interpolation='nearest')
    ax[1,0].set_title('t=30')
    ax[1,1].imshow(filter[40,:,:], origin='lower', cmap='Greys',  interpolation='nearest')
    ax[1,1].set_title('t=40')
    ax[1,2].imshow(filter[50,:,:], origin='lower', cmap='Greys',  interpolation='nearest')
    ax[1,2].set_title('t=50')

    ax[2,0].imshow(filter[60,:,:], origin='lower', cmap='Greys',  interpolation='nearest')
    ax[2,0].set_title('t=60')
    ax[2,1].imshow(filter[70,:,:], origin='lower', cmap='Greys',  interpolation='nearest')
    ax[2,1].set_title('t=70')
    ax[2,2].imshow(filter[87,:,:], origin='lower', cmap='Greys',  interpolation='nearest')
    ax[2,2].set_title('t=87')

    fig.suptitle(r'Filter convection 183TBT-183$\pm 7$ BT')


def plot_filtered_data():
    """plots the filtered ΔTB and W_at_BT data.
    """
    filter = create_convection_filter()
    frame = extract_data()
 
    plt.figure(figsize=(12,12))
    cmapb = mpl.colors.ListedColormap(['None','black','None'])
    cmapw = mpl.colors.ListedColormap(['None','white','None'])
    bounds=[-1,-0.1,0.1,1]
    norm = mpl.colors.BoundaryNorm(bounds, cmapb.N)

    t=0

    # norm1 = mpl.colors.Normalize(vmin=max(frame['W_at_BT'][t,:,:].min(), w_pred_tot[:,:].min()), 
    #                              vmax=min(frame['W_at_BT'][t,:,:].max(), w_pred_tot[:,:].max()))

    norm1 = mpl.colors.Normalize(vmin=-0.2, vmax=0.2)

    # plt.imshow(frame['W_at_BT'][t,:,:], origin='lower', cmap='Spectral', norm=norm1)
    plt.imshow(filter[t,:,:]*filter[t+1,:,:], origin='lower', cmap=cmapb, norm=norm)
    plt.title(f'W_at_BT at t={t}')

    # plt.savefig('Heatmaps_pred_W, monitor=val_loss,patience=70, 5 couches, learning_rate=1e-5.pdf', dpi=300)
    cbar = plt.colorbar(plt.imshow(frame['W_at_BT'][t,:,:], origin='lower', cmap='Spectral', norm=norm1), shrink=0.6)
    cbar.set_label('W_at_BT (mm/hr)')
    plt.show()