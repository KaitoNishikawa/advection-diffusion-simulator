import numpy as np
from scipy.spatial.distance import cdist

class GPRModel():
    def calculate_covar_matrix(sigma_linear, sigma_matern, length, x1, x2):
        x1 = GPRModel.convert_to_vector(x1)
        x2 = GPRModel.convert_to_vector(x2)
        print(f'x1 shape: {x1}')
        print(f'x2 shape: {x2}')

        linear_term = np.dot(x1.T, x2) + (sigma_linear ** 2)

        r = cdist(x1, x2, metric="euclidean")
        r_over_l_term = (np.sqrt(3) * r) / length
        matern_term = (sigma_matern ** 2) * (1 + r_over_l_term) * np.exp(-1 * r_over_l_term)

        return linear_term + matern_term
    
    def calculate_covar_matrix(x1, x2, sigma_linear, sigma_matern, length):
        x1_x_axis = np.linspace(0, len(x1[0]) - 1, len(x1[0])).astype(int)
        x1_y_axis = np.linspace(0, len(x1) - 1, len(x1)).astype(int)
        x2_x_axis = np.linspace(0, len(x2[0]) - 1, len(x2[0])).astype(int)
        x2_y_axis = np.linspace(0, len(x2) - 1, len(x2)).astype(int)

        x1_x_grid, x1_y_grid = np.meshgrid(x1_x_axis, x1_y_axis)
        x2_x_grid, x2_y_grid = np.meshgrid(x2_x_axis, x2_y_axis)

        x1_vec = np.stack([x1_x_grid.flatten(), x1_y_grid.flatten()], axis=1)
        x2_vec = np.stack([x2_x_grid.flatten(), x2_y_grid.flatten()], axis=1)
        print(f'x1 shape: {x1_vec}')
        print(f'x2 shape: {x2_vec}')
        
        return GPRModel.calculate_covar(sigma_linear, sigma_matern, length, x1_vec, x2_vec)
    
    def convert_to_vector(x1):
        x1_x_axis = np.linspace(0, len(x1[0]) - 1, len(x1[0])).astype(int)
        x1_y_axis = np.linspace(0, len(x1) - 1, len(x1)).astype(int)

        x1_x_grid, x1_y_grid = np.meshgrid(x1_x_axis, x1_y_axis)

        return np.stack([x1_x_grid.flatten(), x1_y_grid.flatten()], axis=1)
    
    def mean_function(x_test, x_train, sigma_linear, sigma_matern, length):


