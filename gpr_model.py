import numpy as np
from scipy.spatial.distance import cdist
import GPy

class GPRModel():
    def calculate_covar_matrix(sigma_linear, sigma_matern, lengthscale, x1, x2):
        # x1 = GPRModel.convert_to_vector(x1)
        # x2 = GPRModel.convert_to_vector(x2)
        print(f'x1 shape: {x1}')
        print(f'x2 shape: {x2}')

        linear_term = np.dot(x1, x2.T) + (sigma_linear ** 2)

        r = cdist(x1, x2, metric="euclidean")
        r_over_l_term = (np.sqrt(3) * r) / lengthscale
        matern_term = (sigma_matern ** 2) * (1 + r_over_l_term) * np.exp(-1 * r_over_l_term)

        return linear_term + matern_term
    
    def convert_to_vector(x1):
        x1_arr = np.asarray(x1)
        if x1_arr.ndim == 2 and x1_arr.shape[1] == 2:
            return x1_arr

        x1_x_axis = np.linspace(0, len(x1_arr[0]) - 1, len(x1_arr[0])).astype(int)
        x1_y_axis = np.linspace(0, len(x1_arr) - 1, len(x1_arr)).astype(int)

        x1_x_grid, x1_y_grid = np.meshgrid(x1_x_axis, x1_y_axis)

        return np.stack([x1_x_grid.flatten(), x1_y_grid.flatten()], axis=1)
    
    def calculate_gradient(x_test, x_train, y_train, sigma_linear, sigma_matern, lengthscale, noise_variance=None):
        x_test = GPRModel.convert_to_vector(x_test)
        x_train = GPRModel.convert_to_vector(x_train)

        K = GPRModel.calculate_covar_matrix(sigma_linear, sigma_matern, lengthscale, x_train, x_train)
        jitter_val = noise_variance if noise_variance is not None else 3e-7
        jitter = jitter_val * np.identity(len(x_train))
        K_inv = np.linalg.inv(K + jitter)
        alpha = np.dot(K_inv, y_train) 

        grad_linear = np.dot(x_train.T, alpha).T

        K_matern = GPy.kern.Matern32(input_dim=2, variance=sigma_matern**2, lengthscale=lengthscale)
        grad_matern = K_matern.gradients_X(alpha.T, x_test, x_train)

        return grad_linear + grad_matern

    def optimize_hyperparams(x_train, y_train, max_iters=120):
        x_train = GPRModel.convert_to_vector(x_train)
        y_train = np.asarray(y_train, dtype=float)
        if y_train.ndim == 1:
            y_train = y_train[:, None]

        lin_kern = GPy.kern.Linear(input_dim=2, variances=1.0)
        matern_kern = GPy.kern.Matern32(input_dim=2, variance=1.0, lengthscale=5.0)
        kernel = lin_kern + matern_kern

        model = GPy.models.GPRegression(x_train, y_train, kernel)
        model.Gaussian_noise.variance = 1e-5
        model.optimize(messages=False, max_iters=max_iters)

        lin_var = float(np.mean(lin_kern.variances.values))
        sigma_linear_opt = np.sqrt(lin_var)
        sigma_matern_opt = float(np.sqrt(matern_kern.variance.values[0]))
        length_opt = float(matern_kern.lengthscale.values[0])
        noise_var = float(model.Gaussian_noise.variance.values[0])

        return {
            "sigma_linear": sigma_linear_opt,
            "sigma_matern": sigma_matern_opt,
            "lengthscale": length_opt,
            "noise_variance": noise_var,
        }

    def predict(x_test, x_train, y_train, sigma_linear, sigma_matern, lengthscale, noise_variance=None):
        x_test = GPRModel.convert_to_vector(x_test)
        x_train = GPRModel.convert_to_vector(x_train)

        if y_train.ndim == 1:
            y_train = y_train[:, None]

        K = GPRModel.calculate_covar_matrix(sigma_linear, sigma_matern, lengthscale, x_train, x_train)
        jitter_val = noise_variance if noise_variance is not None else 3e-7
        jitter = jitter_val * np.identity(len(x_train))
        K_inv = np.linalg.inv(K + jitter)

        K_star = GPRModel.calculate_covar_matrix(sigma_linear, sigma_matern, lengthscale, x_test, x_train)
        
        y_pred = np.dot(K_star, np.dot(K_inv, y_train))
        return y_pred.flatten()

