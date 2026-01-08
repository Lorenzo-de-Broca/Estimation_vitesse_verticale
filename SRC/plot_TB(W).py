from matplotlib import testing
import numpy as np
import matplotlib.pyplot as plt
from SRC.extract_data import extract_data, create_reg_array1, create_reg_array3, create_train_test_matrix
from SRC.filtre_convection import create_convection_filter
from scipy.optimize import curve_fit
import matplotlib as mpl

frame = extract_data()
filter = create_convection_filter()

train_mat, test_mat = create_train_test_matrix(prop_train=0.6)

TB_filtered1830 = create_reg_array3('1830', frame, filter, train_mat)
TB_filtered1837 = create_reg_array3('1837', frame, filter, train_mat)
TB_filtered183T = create_reg_array3('183T', frame, filter, train_mat)
TB_filtered3250 = create_reg_array3('3250', frame, filter, train_mat)
TB_filtered3257 = create_reg_array3('3257', frame, filter, train_mat)
TB_filtered325T = create_reg_array3('325T', frame, filter, train_mat)