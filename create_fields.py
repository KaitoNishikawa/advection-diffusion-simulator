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
    
    def create_source_sink_field(array_size, center_point, radius, strength):
        source_sink_field = np.zeros((array_size, array_size))
        
        row, col = center_point
    
        # S[row, col] = strength 
        
        r_start = max(0, row - radius)
        r_end = min(array_size, row + radius)
        c_start = max(0, col - radius)
        c_end = min(array_size, col + radius)
        
        source_sink_field[r_start:r_end, c_start:c_end] = strength
        
        return source_sink_field
