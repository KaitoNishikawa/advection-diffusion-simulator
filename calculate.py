import numpy as np

class Calculate():    
    def calculate_gradient(field):
        grad_field_y, grad_field_x = np.gradient(field)
        return grad_field_x, grad_field_y

    def calculate_divergence(field_x, field_y):
        garbage, div_field_x = np.gradient(field_x)
        div_field_y, garbage = np.gradient(field_y)

        return div_field_x + div_field_y