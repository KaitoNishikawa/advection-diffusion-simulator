import numpy as np
import matplotlib.pyplot as plt

class CreateFields():
    def create_scaler_field(array_size, center_size, center_value=0):
        """Create a square scalar field with an optional high-value center block."""
        if array_size <= 0:
            raise ValueError("array_size must be positive")
        if center_size < 0:
            raise ValueError("center_size cannot be negative")

        arr = np.zeros((array_size, array_size), dtype=float)

        if center_size > 0 and center_value is not None:
            start_index = (array_size - center_size) // 2
            end_index = start_index + center_size

            start_index = max(0, start_index)
            end_index = min(array_size, end_index)

            arr[start_index:end_index, start_index:end_index] = center_value

        return arr
    
    def create_flux_field(scaler_field, x_velo, y_velo):
        vector_field_x = scaler_field * x_velo
        vector_field_y = scaler_field * y_velo

        return vector_field_x, vector_field_y
    
    def create_source_sink_field(array_size, center_point, radius, strength):
        # x = np.arange(0, array_size)
        # y = np.arange(0, array_size)
        # X, Y = np.meshgrid(x, y)
        
        # # Mapping: center_point is (row, col) -> (y, x)
        # cy, cx = center_point
        
        # # Use radius as the standard deviation (sigma) for the Gaussian
        # sigma = radius if radius > 0 else 1e-5
        
        # # Gaussian distribution
        # dist_sq = (X - cx)**2 + (Y - cy)**2
        # source_sink_field = strength * np.exp(-dist_sq / (2 * sigma**2))
        
        # return source_sink_field
    
        x = np.arange(0, array_size)
        y = np.arange(0, array_size)
        X, Y = np.meshgrid(x, y)
        
        # Mapping: center_point is (row, col) -> (y, x)
        cy, cx = center_point
        
        # Use radius as the standard deviation (sigma) for the Gaussian
        sigma = radius if radius > 0 else 1e-5
        
        # Gaussian distribution
        dist_sq = (X - cx)**2 + (Y - cy)**2
        source_sink_field = strength * np.exp(-dist_sq / (2 * sigma**2))
        
        return source_sink_field
