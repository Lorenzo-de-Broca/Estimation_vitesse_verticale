import numpy as np
import matplotlib.pyplot as plt
from SRC.extract_data import extract_data, create_reg_array1, create_reg_array3, create_train_test_matrix
from SRC.filtre_convection import create_convection_filter
from scipy.optimize import curve_fit
import matplotlib as mpl

frame = extract_data()
filter = create_convection_filter()


TB_filtered1830 = create_reg_array3('1830')
TB_filtered1837 = create_reg_array3('1837')
TB_filtered183T = create_reg_array3('183T')
TB_filtered3250 = create_reg_array3('3250')
TB_filtered3257 = create_reg_array3('3257')
TB_filtered325T = create_reg_array3('325T')

plt.figure()
plt.imshow(testing, origin='lower', cmap='gray')
plt.colorbar()
plt.show()
