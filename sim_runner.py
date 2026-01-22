import matplotlib.pyplot as plt
import numpy as np

from create_fields import CreateFields
from plot_fields import PlotFields
from calculate import Calculate
from gpr_model import GPRModel

default_value = 30

scaler_field = CreateFields.create_scaler_field(200, default_value)
velo_x = 0.1
velo_y = 0

scaler_field_time_series = [scaler_field.copy()]

dt = 0.1
tolerance = 1e-5
max_iterations = 1000
loss_coefficient = 0.01
leak_rate = 0.001
iteration = 0

# source_sink_term_1 = CreateFields.create_source_sink_field(200, (120, 70), radius=3, strength=10)
# source_sink_term_2 = CreateFields.create_source_sink_field(200, (20, 20), radius=3, strength=10)
# source_sink_term_3 = CreateFields.create_source_sink_field(200, (20, 160), radius=3, strength=10)

source_sink_term_1 = CreateFields.create_source_sink_field(200, (140, 140), radius=20, strength=0.01)
source_sink_term_2 = CreateFields.create_source_sink_field(200, (70, 80), radius=20, strength=0.008)

while True:
    flux_field_x, flux_field_y = CreateFields.create_flux_field(scaler_field, velo_x, velo_y)
    scaler_field_gradient_x, scaler_field_gradient_y = Calculate.calculate_gradient(scaler_field)

    advection_term = Calculate.calculate_divergence(flux_field_x, flux_field_y) * (-1)
    divergence_term = Calculate.calculate_divergence(scaler_field_gradient_x, scaler_field_gradient_y)

    loss_term = -loss_coefficient * (scaler_field - default_value)
    # rate = advection_term + divergence_term + source_sink_term_1 + source_sink_term_2 + source_sink_term_3 + loss_term
    rate = advection_term + divergence_term + source_sink_term_1 + source_sink_term_2 + loss_term

    change = rate * dt
    max_change = np.max(np.abs(change))
    if max_change < tolerance or iteration > max_iterations:
        break

    scaler_field = scaler_field + change

    scaler_field[0, :] -= leak_rate * (scaler_field[0, :] - default_value)
    scaler_field[-1, :] -= leak_rate * (scaler_field[-1, :] - default_value)
    scaler_field[:, 0] -= leak_rate * (scaler_field[:, 0] - default_value)
    scaler_field[:, -1] -= leak_rate * (scaler_field[:, -1] - default_value)

    scaler_field_time_series.append(scaler_field.copy())

    iteration += 1



# PlotFields.plot_scaler_field(scaler_field)
# PlotFields.plot_vector_field(flux_field_x, flux_field_y)
# PlotFields.animate_scaler_field(scaler_field_time_series, auto_close=False)

last_field = scaler_field_time_series[-1]

# # --- GPR Sampling and Training ---
# num_samples = 30

# # Calculate gradient magnitude for importance sampling
# grad_magnitude = np.sqrt(grad_truth_x**2 + grad_truth_y**2)
# prob_distribution = grad_magnitude.flatten()
# # Add a small epsilon to avoid zero probabilities and ensure some global coverage
# prob_distribution = prob_distribution + 1 * np.mean(prob_distribution)
# prob_distribution /= prob_distribution.sum()

# sample_indices = np.random.choice(last_field.size, size=num_samples, replace=False, p=prob_distribution)
# sample_indices = np.random.choice(last_field.size, size=num_samples, replace=False)
# print(sample_indices.shape)
# y_coords, x_coords = np.unravel_index(sample_indices, last_field.shape)

x_train, y_train, x_test, grad_truth_x, grad_truth_y = GPRModel.get_points(scaler_field)

current_pos = x_train[-1].astype(float).copy() 
step_size = 3.0  
theta = np.pi / 2
robot_path = [current_pos.copy()]

print(f"Starting Navigation from: {current_pos}")

for n in range(50):  
    model = GPRModel.create_model(x_train, y_train)
    grad_pred = GPRModel.predict(x_train, current_pos.reshape(1, -1), model)
    
    dn = grad_pred[0] 
    
    # Normalize the gradient to maintain constant step speed
    mag = np.linalg.norm(dn)
    if mag > 1e-8:
        dn = dn / mag
    
    direction = dn  
    
    buffer = 10
    if (current_pos[0] < buffer or current_pos[0] > 190 or 
        current_pos[1] < buffer or current_pos[1] > 190):
        
        rot_matrix = np.array([[np.cos(theta), -np.sin(theta)], 
                               [np.sin(theta),  np.cos(theta)]])
        direction = np.dot(rot_matrix, direction)

    new_pos = current_pos + direction * step_size
    
    ix, iy = int(np.clip(new_pos[0], 0, 199)), int(np.clip(new_pos[1], 0, 199))
    new_measurement = scaler_field[iy, ix]
    
    x_train = np.vstack([x_train, [ix, iy]])
    y_train = np.append(y_train, new_measurement)
    
    current_pos = new_pos.copy()
    robot_path.append(current_pos.copy())

PlotFields.animate_robot_path(scaler_field, robot_path)



