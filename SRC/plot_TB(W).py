from matplotlib import testing
import numpy as np
import matplotlib.pyplot as plt
from SRC.extract_data import extract_data, create_reg_array1, create_reg_array2, create_reg_array3, create_train_test_matrix
from SRC.filtre_convection import create_convection_filter
from scipy.optimize import curve_fit
import matplotlib as mpl

frame = extract_data()
filter = create_convection_filter()

train_mat, test_mat = create_train_test_matrix(0.6)

# TB_filtered1830, W_filtered = create_reg_array3('1830', frame, filter, train_mat)
# TB_filtered1837, _ = create_reg_array3('1837', frame, filter, train_mat)
# TB_filtered183T, _ = create_reg_array3('183T', frame, filter, train_mat)
# TB_filtered3250, _ = create_reg_array3('3250', frame, filter, train_mat)
# TB_filtered3257, _ = create_reg_array3('3257', frame, filter, train_mat)
# TB_filtered325T, _ = create_reg_array3('325T', frame, filter, train_mat)

 
# %% Figure
# plt.figure()
# plt.plot(TB_filtered1830, W_filtered, 'o', markersize=1, label='183±1 GHz')
# plt.plot(TB_filtered1837, W_filtered, 'o', markersize=1, label='183±7 GHz')
# plt.plot(TB_filtered183T, W_filtered, 'o', markersize=1, label='183±10 GHz')
# plt.plot(TB_filtered3250, W_filtered, 'o', markersize=1, label='325±1 GHz')
# plt.plot(TB_filtered3257, W_filtered, 'o', markersize=1, label='325±7 GHz')
# plt.plot(TB_filtered325T, W_filtered, 'o', markersize=1, label='325±10 GHz')
# plt.legend()
# plt.xlabel('aos_1830BT (K)')
# plt.ylabel('W_at_BT (mm/hr)')
# plt.show()

#%%
DTB_1830, W_filtered = create_reg_array2('1830', frame, filter, train_mat)
DTB_1837, _ = create_reg_array2('1837', frame, filter, train_mat)
DTB_183T, _ = create_reg_array2('183T', frame, filter, train_mat)
DTB_3250, _ = create_reg_array2('3250', frame, filter, train_mat)
DTB_3257, _ = create_reg_array2('3257', frame, filter, train_mat)
DTB_325T, _ = create_reg_array2('325T', frame, filter, train_mat)

plt.figure()
plt.plot(DTB_1830, W_filtered, 'o', markersize=1, label='183±1 GHz', alpha=0.5)
plt.plot(DTB_1837, W_filtered, 'o', markersize=1, label='183±7 GHz', alpha=0.5)
plt.plot(DTB_183T, W_filtered, 'o', markersize=1, label='183±10 GHz', alpha=0.5)
plt.plot(DTB_3250, W_filtered, 'o', markersize=1, label='325±1 GHz', alpha=0.5)
plt.plot(DTB_3257, W_filtered, 'o', markersize=1, label='325±7 GHz', alpha=0.5)
plt.plot(DTB_325T, W_filtered, 'o', markersize=1, label='325±10 GHz', alpha=0.5)
plt.legend()
plt.xlabel('Δaos_1830BT/30s (K/s)')
plt.ylabel('W_at_BT (mm/hr)')
plt.title('Convection filter: p1 > p7 - 30K')
# plt.savefig('DeltaTB VS WatBT filter-10.pdf', dpi=300)
plt.show()