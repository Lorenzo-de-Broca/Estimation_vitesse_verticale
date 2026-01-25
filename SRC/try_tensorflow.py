#%%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from SRC.extract_data import extract_data, create_reg_array1, create_reg_array2, create_reg_array3, create_train_test_matrix
from SRC.filtre_convection import create_convection_filter
from sklearn.linear_model import LogisticRegression

#%% import training, test data and filter
frame = extract_data()
filter = create_convection_filter()

train_frac = 0.6
train_mat, test_mat = create_train_test_matrix(train_frac)

DTB_1830_train, W_filtered_1830_train = create_reg_array2('1830', frame, filter, train_mat)
DTB_1833_train, W_filtered_1833_train = create_reg_array2('1833', frame, filter, train_mat)
DTB_1835_train, W_filtered_1835_train = create_reg_array2('1835', frame, filter, train_mat)
DTB_1837_train, W_filtered_1837_train = create_reg_array2('1837', frame, filter, train_mat)
DTB_183T_train, W_filtered_183T_train = create_reg_array2('183T', frame, filter, train_mat)
DTB_3250_train, W_filtered_3250_train = create_reg_array2('3250', frame, filter, train_mat)
DTB_3253_train, W_filtered_3253_train = create_reg_array2('3253', frame, filter, train_mat)
DTB_3255_train, W_filtered_3255_train = create_reg_array2('3255', frame, filter, train_mat)
DTB_3257_train, W_filtered_3257_train = create_reg_array2('3257', frame, filter, train_mat)
DTB_325T_train, W_filtered_325T_train = create_reg_array2('325T', frame, filter, train_mat)

DTB_1830_test, W_filtered_1830_test = create_reg_array2('1830', frame, filter, test_mat)
DTB_1833_test, W_filtered_1833_test = create_reg_array2('1833', frame, filter, test_mat)
DTB_1835_test, W_filtered_1835_test = create_reg_array2('1835', frame, filter, test_mat)
DTB_1837_test, W_filtered_1837_test = create_reg_array2('1837', frame, filter, test_mat)
DTB_183T_test, W_filtered_183T_test = create_reg_array2('183T', frame, filter, test_mat)
DTB_3250_test, W_filtered_3250_test = create_reg_array2('3250', frame, filter, test_mat)
DTB_3253_test, W_filtered_3253_test = create_reg_array2('3253', frame, filter, test_mat)
DTB_3255_test, W_filtered_3255_test = create_reg_array2('3255', frame, filter, test_mat)
DTB_3257_test, W_filtered_3257_test = create_reg_array2('3257', frame, filter, test_mat)
DTB_325T_test, W_filtered_325T_test = create_reg_array2('325T', frame, filter, test_mat)


DTB_1830 = np.array([(frame['aos_1830BT'][t+1,:,:]-frame['aos_1830BT'][t,:,:])/30 for t in range(87)])
DTB_1833 = np.array([(frame['aos_1833BT'][t+1,:,:]-frame['aos_1833BT'][t,:,:])/30 for t in range(87)])
DTB_1835 = np.array([(frame['aos_1835BT'][t+1,:,:]-frame['aos_1835BT'][t,:,:])/30 for t in range(87)])
DTB_1837 = np.array([(frame['aos_1837BT'][t+1,:,:]-frame['aos_1837BT'][t,:,:])/30 for t in range(87)])
DTB_183T = np.array([(frame['aos_183TBT'][t+1,:,:]-frame['aos_183TBT'][t,:,:])/30 for t in range(87)])
DTB_3250 = np.array([(frame['aos_3250BT'][t+1,:,:]-frame['aos_3250BT'][t,:,:])/30 for t in range(87)])
DTB_3253 = np.array([(frame['aos_3253BT'][t+1,:,:]-frame['aos_3253BT'][t,:,:])/30 for t in range(87)])
DTB_3255 = np.array([(frame['aos_3255BT'][t+1,:,:]-frame['aos_3255BT'][t,:,:])/30 for t in range(87)])
DTB_3257 = np.array([(frame['aos_3257BT'][t+1,:,:]-frame['aos_3257BT'][t,:,:])/30 for t in range(87)])
DTB_325T = np.array([(frame['aos_325TBT'][t+1,:,:]-frame['aos_325TBT'][t,:,:])/30 for t in range(87)])

#%% sélection d'un sous-échantillon pour accélérer les tests
np.random.seed(42)
size = 10000
indices_train = np.random.choice(len(DTB_1830_train), size=size, replace=False)
DTB_1830_train, DTB_1833_train, DTB_1835_train, DTB_1837_train, DTB_183T_train = DTB_1830_train[indices_train], DTB_1833_train[indices_train], DTB_1835_train[indices_train], DTB_1837_train[indices_train], DTB_183T_train[indices_train]
DTB_3250_train, DTB_3253_train, DTB_3255_train, DTB_3257_train, DTB_325T_train = DTB_3250_train[indices_train], DTB_3253_train[indices_train], DTB_3255_train[indices_train], DTB_3257_train[indices_train], DTB_325T_train[indices_train]
W_filtered_1830_train = W_filtered_1830_train[indices_train]

indices_test = np.random.choice(len(DTB_1830_test), size=int((1-train_frac)/train_frac*size), replace=False)
DTB_1830_test, DTB_1833_test, DTB_1835_test, DTB_1837_test, DTB_183T_test = DTB_1830_test[indices_test], DTB_1833_test[indices_test], DTB_1835_test[indices_test], DTB_1837_test[indices_test], DTB_183T_test[indices_test]
DTB_3250_test, DTB_3253_test, DTB_3255_test, DTB_3257_test, DTB_325T_test = DTB_3250_test[indices_test], DTB_3253_test[indices_test], DTB_3255_test[indices_test], DTB_3257_test[indices_test], DTB_325T_test[indices_test]
W_filtered_1830_test = W_filtered_1830_test[indices_test]


#%% tensorflow example
from tensorflow import keras
import tensorflow as tf
import math
from tensorflow.keras.callbacks import EarlyStopping

x_data_train=np.array([DTB_1830_train, DTB_1833_train, DTB_1835_train, DTB_1837_train, DTB_183T_train, DTB_3250_train, DTB_3253_train, DTB_3255_train, DTB_3257_train, DTB_325T_train]).T
x_data_test=np.array([DTB_1830_test, DTB_1833_test, DTB_1835_test, DTB_1837_test, DTB_183T_test, DTB_3250_test, DTB_3253_test, DTB_3255_test, DTB_3257_test, DTB_325T_test]).T

model = keras.Sequential()
model.add(keras.layers.Dense(units = 128, activation = 'relu', input_shape=x_data_train.shape[1:]))
model.add(keras.layers.Dense(units = 128, activation = 'relu'))
model.add(keras.layers.Dense(units = 128, activation = 'relu'))
model.add(keras.layers.Dense(units = 128, activation = 'relu'))
model.add(keras.layers.Dense(units = 1, activation = 'linear'))
model.compile(loss='mae', optimizer=tf.optimizers.Adam(learning_rate=1e-4))
# model.compile(optimizer='adam', loss='mse')

# Display the model
model.summary()

early_stop = EarlyStopping(
    monitor='val_loss',           # Monitor validation loss
    patience=70,                  # Stop if no improvement for 10 epochs
    restore_best_weights=True,    # Restore weights from best epoch
    verbose=1                     # Print when stopping
)
epochs=5000
history = model.fit(x_data_train, W_filtered_1830_train, epochs=epochs, verbose=1, validation_data=(x_data_test, W_filtered_1830_test), callbacks=[early_stop])

W_predicted = model.predict(x_data_test)
W_predicted_train = model.predict(x_data_train)

t=0
W_pred_tot = model.predict((np.array([DTB_1830[t,:,:].reshape(-1,1), DTB_1833[t,:,:].reshape(-1,1), DTB_1835[t,:,:].reshape(-1,1), DTB_1837[t,:,:].reshape(-1,1), DTB_183T[t,:,:].reshape(-1,1), DTB_3250[t,:,:].reshape(-1,1), DTB_3253[t,:,:].reshape(-1,1), DTB_3255[t,:,:].reshape(-1,1), DTB_3257[t,:,:].reshape(-1,1), DTB_325T[t,:,:].reshape(-1,1)]).T)[0])
w_pred_tot = W_pred_tot.reshape((500,500))

score = model.evaluate(x_data_test, W_filtered_1830_test, verbose=1)
print('Test loss:', score)

plt.figure(figsize=(12,9))
plt.plot(x_data_test[:, 2], W_filtered_1830_test,'o', label='True')
plt.plot(x_data_test[:, 2], W_predicted,'o', label='Predicted')
plt.xlabel('Δaos_1830BT/30s (K/s)')
plt.ylabel('W_at_BT (mm/hr)')
plt.title('True vs Predicted with Tensorflow')
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(12,9))
plt.plot(W_filtered_1830_train, W_predicted_train,'o', alpha=0.4, label='training')
plt.plot(W_filtered_1830_test, W_predicted,'o', label='testing')
plt.plot([min(W_filtered_1830_test), max(W_filtered_1830_test)], [min(W_filtered_1830_test), max(W_filtered_1830_test)], 'k--')
plt.xlabel('True W_at_BT')
plt.ylabel('Predicted W_at_BT')
plt.title(f'True vs Predicted with Tensorflow (score={score:.4f})')
plt.grid()
plt.legend()
# plt.savefig('trueVSpredict_val_split=0.4, patience=70, 5 couches, learning_rate=1e-5.pdf', dpi=300)
plt.show()

plt.figure(figsize=(12,9))
loss = history.history['loss']
val_loss = history.history['val_loss']
# plt.plot(np.linspace(0, len(loss), len(loss)), loss)
plt.plot(np.linspace(0, len(val_loss), len(val_loss)), val_loss)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title(f"Evolution de loss et validation_loss au cours des epochs. Validation split=0.4 (score={score:.4f})")  
plt.grid()
# plt.savefig('losses_val_split=0.4, patience=70, 5 couches, learning_rate=1e-5.pdf', dpi=300)
plt.show()

#%% figure heatmaps
import matplotlib as mpl

figure, ax = plt.subplots(2,1, figsize=(12,9), layout='constrained')

cmapb = mpl.colors.ListedColormap(['None','black','None'])
cmapw = mpl.colors.ListedColormap(['None','white','None'])
bounds=[-1,-0.1,0.1,1]
norm = mpl.colors.BoundaryNorm(bounds, cmapb.N)

t=0

# norm1 = mpl.colors.Normalize(vmin=max(frame['W_at_BT'][t,:,:].min(), w_pred_tot[:,:].min()), 
#                              vmax=min(frame['W_at_BT'][t,:,:].max(), w_pred_tot[:,:].max()))

norm1 = mpl.colors.Normalize(vmin=-0.2, vmax=0.2)

im1 = ax[0].imshow(frame['W_at_BT'][t,:,:], origin='lower', cmap='Spectral', norm=norm1)
ax[0].imshow(filter[t,:,:]*filter[t+1,:,:], origin='lower', cmap=cmapw, norm=norm)
ax[0].set_title(f'W_at_BT at t={t}')

ax[1].imshow(w_pred_tot[:,:], origin='lower', cmap='Spectral', norm=norm1)
ax[1].imshow(filter[t,:,:]*filter[t+1,:,:], origin='lower', cmap=cmapw, norm=norm)
ax[1].set_title(f'Predicted W at t={t}')

# plt.savefig('Heatmaps_pred_W, monitor=val_loss,patience=70, 5 couches, learning_rate=1e-5.pdf', dpi=300)
cbar = figure.colorbar(im1, ax=ax.ravel().tolist(), shrink=0.6)
cbar.set_label('W_at_BT (mm/hr)')
plt.show()

# %% figure difference W_at_BT - W_predicted
filter = create_convection_filter()

plt.figure(figsize=(12,9))

cmapb = mpl.colors.ListedColormap(['black','None'])
cmapw = mpl.colors.ListedColormap(['None','white','None'])
bounds=[-1,-0.1,0.1,1]
norm = mpl.colors.BoundaryNorm(bounds, cmapw.N)
norm_pos = mpl.colors.BoundaryNorm([-1,-1e-2,1], cmapb.N)

t=0

norm2= mpl.colors.Normalize(vmin=-0.1, vmax=0.1)

plt.imshow(frame['W_at_BT'][t,:,:], origin='lower', cmap='Spectral', norm=norm1, alpha=0.6)
plt.imshow(frame['W_at_BT'][t,:,:]*w_pred_tot[:,:], origin='lower', cmap=cmapb, norm=norm_pos)
plt.imshow(filter[t,:,:]*filter[t+1,:,:], origin='lower', cmap=cmapw, norm=norm)
plt.title(f'W_at_BT at t={t}')
plt.colorbar(label='difference W_at_BT - W_predicted (mm/hr)')
plt.show()

# %%
