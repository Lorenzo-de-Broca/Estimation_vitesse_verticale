import numpy as np
import matplotlib.pyplot as plt
from SRC.extract_data import extract_data

def create_convection_filter():
    """creates a filter to avoid convection zones. Mask where TB (183+/-1) is higher than TB (183+/-7).

    Returns:
        filter (np.array): 3D array (time, lat, lon) with 1 where no convection and 0 where convection.
    """
    frame = extract_data()
    p1 = np.array(frame['aos_1830BT'][:,:,:])
    p7 = np.array(frame['aos_1837BT'][:,:,:])

    filter = np.ones(np.shape(p1))
    filter[p1>p7-20] = 0

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


