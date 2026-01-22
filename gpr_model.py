import numpy as np
from scipy.spatial.distance import cdist
import GPy
from calculate import Calculate
from plot_fields import PlotFields

class GPRModel():
    def create_model(x, y):
        if y.ndim == 1:
            y = y[:, None]
        k1 = GPy.kern.Matern32(input_dim=2, variance=1.0, lengthscale=0.5)
        k2 = GPy.kern.Linear(input_dim=2)
        kernel = k1 + k2

        model = GPy.models.GPRegression(x, y, kernel, noise_var=0.01)
        model.optimize()

        return model
    
    def predict(x_train, x_test, model):
        grad = model.kern.gradients_X(model.posterior.woodbury_vector.T, x_test, x_train)
        return grad 
    
    def get_points(scaler_field):
        y_coords = np.array([100, 100, 120, 120])
        x_coords = np.array([50, 70, 50, 70])

        x_train = np.column_stack((x_coords, y_coords))
        y_train = scaler_field[y_coords, x_coords]

        y_index_grid, x_index_grid = np.indices(scaler_field.shape)
        y_indices = y_index_grid.ravel()
        x_indices = x_index_grid.ravel()
        x_test = np.column_stack((x_indices, y_indices))

        grad_truth_x, grad_truth_y = Calculate.calculate_gradient(scaler_field)
        grad_truth_x_train = grad_truth_x[y_coords, x_coords]
        grad_truth_y_train = grad_truth_y[y_coords, x_coords]

        # PlotFields.plot_vector_field(grad_truth_x, grad_truth_y, "last gradient field")
        PlotFields.plot_sampled_vectors(scaler_field, x_coords, y_coords, grad_truth_x_train, grad_truth_y_train)

        return x_train, y_train, x_test, grad_truth_x, grad_truth_y
    
    def run_model(x_train, y_train, x_test, grad_truth_x, grad_truth_y):
        model = GPRModel.create_model(x_train, y_train)
        grad_pred = GPRModel.predict(x_train, x_test, model)

        PlotFields.compare_vector_fields(grad_truth_x, grad_truth_y, grad_pred, scale=100)

        return grad_pred

    # def calculate_covar_matrix(sigma_linear, sigma_matern, lengthscale, x1, x2):
    #     # x1 = GPRModel.convert_to_vector(x1)
    #     # x2 = GPRModel.convert_to_vector(x2)
    #     # print(f'x1 shape: {x1}')
    #     # print(f'x2 shape: {x2}')

    #     linear_term = np.dot(x1, x2.T) + (sigma_linear ** 2)

    #     r = cdist(x1, x2, metric="euclidean")
    #     r_over_l_term = (np.sqrt(3) * r) / lengthscale
    #     matern_term = (sigma_matern ** 2) * (1 + r_over_l_term) * np.exp(-1 * r_over_l_term)

    #     return linear_term + matern_term
    
    
    # def calculate_gradient(x_test, x_train, y_train, sigma_linear, sigma_matern, lengthscale, noise_variance=None):
    #     # x_test = GPRModel.convert_to_vector(x_test)
    #     # x_train = GPRModel.convert_to_vector(x_train)

    #     K = GPRModel.calculate_covar_matrix(sigma_linear, sigma_matern, lengthscale, x_train, x_train)
    #     jitter_val = noise_variance if noise_variance is not None else 3e-7
    #     jitter = jitter_val * np.identity(len(x_train))
    #     K_inv = np.linalg.inv(K + jitter)
    #     alpha = np.dot(K_inv, y_train) 

    #     grad_linear = np.dot(x_train.T, alpha).T

    #     K_matern = GPy.kern.Matern32(input_dim=2, variance=sigma_matern**2, lengthscale=lengthscale)
    #     grad_matern = K_matern.gradients_X(alpha.T, x_test, x_train)

    #     return grad_linear + grad_matern

    # def predict(x_test, x_train, y_train, sigma_linear, sigma_matern, lengthscale, noise_variance=None):
    #     # x_test = GPRModel.convert_to_vector(x_test)
    #     # x_train = GPRModel.convert_to_vector(x_train)

    #     if y_train.ndim == 1:
    #         y_train = y_train[:, None]

    #     K = GPRModel.calculate_covar_matrix(sigma_linear, sigma_matern, lengthscale, x_train, x_train)
    #     jitter_val = noise_variance if noise_variance is not None else 3e-7
    #     jitter = jitter_val * np.identity(len(x_train))
    #     K_inv = np.linalg.inv(K + jitter)

    #     K_star = GPRModel.calculate_covar_matrix(sigma_linear, sigma_matern, lengthscale, x_test, x_train)
        
    #     y_pred = np.dot(K_star, np.dot(K_inv, y_train))
    #     return y_pred.flatten()
