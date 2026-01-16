import matplotlib.pyplot as plt
import numpy as np

from create_fields import CreateFields
from plot_fields import PlotFields
from calculate import Calculate
from gpr_model import GPRModel

scaler_field = CreateFields.create_scaler_field(200, 10)
velo_x = 0.6
velo_y = 0.3

scaler_field_time_series = [scaler_field.copy()]

dt = 0.1
tolerance = 1e-5
max_iterations = 1000
loss_coefficient = 0.05
leak_rate = 0.1
iteration = 0

source_sink_term_1 = CreateFields.create_source_sink_field(200, (120, 70), radius=3, strength=10)
source_sink_term_2 = CreateFields.create_source_sink_field(200, (20, 20), radius=3, strength=10)
source_sink_term_3 = CreateFields.create_source_sink_field(200, (20, 160), radius=3, strength=10)

while True:
    flux_field_x, flux_field_y = CreateFields.create_flux_field(scaler_field, velo_x, velo_y)
    scaler_field_gradient_x, scaler_field_gradient_y = Calculate.calculate_gradient(scaler_field)

    advection_term = Calculate.calculate_divergence(flux_field_x, flux_field_y) * (-1)
    divergence_term = Calculate.calculate_divergence(scaler_field_gradient_x, scaler_field_gradient_y)

    loss_term = -loss_coefficient * scaler_field
    rate = advection_term + divergence_term + source_sink_term_1 + source_sink_term_2 + source_sink_term_3 + loss_term

    change = rate * dt
    max_change = np.max(np.abs(change))
    if max_change < tolerance or iteration > max_iterations:
        break

    scaler_field = scaler_field + change

    scaler_field[0, :] -= leak_rate * scaler_field[0, :] 
    scaler_field[-1, :] -= leak_rate * scaler_field[-1, :] 
    scaler_field[:, 0] -= leak_rate * scaler_field[:, 0] 
    scaler_field[:, -1] -= leak_rate * scaler_field[:, -1] 

    scaler_field_time_series.append(scaler_field.copy())

    iteration += 1



# PlotFields.plot_scaler_field(scaler_field)
# PlotFields.plot_vector_field(flux_field_x, flux_field_y)
# PlotFields.animate_scaler_field(scaler_field_time_series, auto_close=False)

last_field = scaler_field_time_series[-1]
grad_truth_x, grad_truth_y = Calculate.calculate_gradient(last_field)

PlotFields.plot_vector_field(grad_truth_x, grad_truth_y, "last gradient field")

# --- GPR Sampling and Training ---
num_samples = 100

# Calculate gradient magnitude for importance sampling
grad_magnitude = np.sqrt(grad_truth_x**2 + grad_truth_y**2)
prob_distribution = grad_magnitude.flatten()
# Add a small epsilon to avoid zero probabilities and ensure some global coverage
prob_distribution = prob_distribution + 0.3 * np.mean(prob_distribution)
prob_distribution /= prob_distribution.sum()

sample_indices = np.random.choice(last_field.size, size=num_samples, replace=False, p=prob_distribution)
y_coords, x_coords = np.unravel_index(sample_indices, last_field.shape)
# Use (x, y) coordinates to match GPRModel.convert_to_vector
x_train = np.column_stack((x_coords, y_coords)) 

# Sample the gradient vectors at these locations
y_train_gx = grad_truth_x[y_coords, x_coords]
y_train_gy = grad_truth_y[y_coords, x_coords]

# Plot the sampled vectors before training
print("Plotting sampled vectors...")
PlotFields.plot_sampled_vectors(x_coords, y_coords, y_train_gx, y_train_gy)

# Optimize hyperparameters (using x-component as representative)
print("Optimizing hyperparameters...")
params = GPRModel.optimize_hyperparams(x_train, y_train_gx)

print(params)

# Predict the full gradient field components
print("Predicting full field...")
x_test = last_field # Passing the full field grid
gx_pred = GPRModel.predict(x_test, x_train, y_train_gx, **params)
gy_pred = GPRModel.predict(x_test, x_train, y_train_gy, **params)

# Reshape predictions back to grid
gx_pred_grid = gx_pred.reshape(last_field.shape)
gy_pred_grid = gy_pred.reshape(last_field.shape)

# Plot the comparison
print("Plotting results...")
PlotFields.compare_vector_fields(grad_truth_x, grad_truth_y, gx_pred_grid, gy_pred_grid, scale=100)


