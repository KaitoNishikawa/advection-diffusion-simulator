import numpy as np
import matplotlib.pyplot as plt

class CreateFields():
    def create_scaler_field(array_size, center_size):
        arr = np.zeros((array_size, array_size), dtype=int)

        start_index = (array_size - center_size) // 2  # (100 - 10) / 2 = 45
        end_index = start_index + center_size        # 45 + 10 = 55

        arr[start_index:end_index, start_index:end_index] = 10

        return arr
    
    def create_flux_field(scaler_field, x_velo, y_velo):
        vector_field_x = scaler_field * x_velo
        vector_field_y = scaler_field * y_velo

        return vector_field_x, vector_field_y
