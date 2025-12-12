import matplotlib.pyplot as plt
import numpy as np

from create_fields import CreateFields
from plot_fields import PlotFields
from calculate import Calculate
from calculate import Calculate

scaler_field = CreateFields.create_scaler_field(100, 10)
velo_x = 2
velo_y = 2

scaler_field_time_series = [scaler_field.copy()]
source_sink_term = CreateFields.create_source_sink_field(100, (50, 50), radius=2, strength=5.0)

dt = 0.1

for i in range(100):
    flux_field_x, flux_field_y = CreateFields.create_flux_field(scaler_field, velo_x, velo_y)
    scaler_field_gradient_x, scaler_field_gradient_y = Calculate.calculate_gradient(scaler_field)

    advection_term = Calculate.calculate_divergence(flux_field_x, flux_field_y) * (-1)
    divergence_term = Calculate.calculate_divergence(scaler_field_gradient_x, scaler_field_gradient_y)

    rate = advection_term + divergence_term + source_sink_term
    scaler_field = scaler_field + (rate * dt)
    scaler_field_time_series.append(scaler_field.copy())

# PlotFields.plot_scaler_field(scaler_field)
# PlotFields.plot_vector_field(flux_field_x, flux_field_y)
PlotFields.animate_scaler_field(scaler_field_time_series)
