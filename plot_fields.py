import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation

class PlotFields():
    def plot_scaler_field(field):
        plt.figure(figsize=(8, 8))
        heatmap = plt.imshow(field, cmap='viridis', interpolation='nearest')
        plt.colorbar(heatmap, fraction=0.046, pad=0.04, ticks=[0, 10], label='Value')
        plt.show()
    
    def plot_vector_field(field_x, field_y):
        x = np.arange(0, len(field_x)) # 0, 1, 2, ..., 99
        y = np.arange(0, len(field_x))
        X, Y = np.meshgrid(x, y)

        skip = 2
        X_plot = X[::skip, ::skip]
        Y_plot = Y[::skip, ::skip]
        Fx_plot = field_x[::skip, ::skip]
        Fy_plot = field_y[::skip, ::skip]

        plt.figure(figsize=(8, 8))
        plt.quiver(X_plot, Y_plot, Fx_plot, Fy_plot, 
           scale=100,  
           color='blue', 
           headwidth=5)
        plt.show()

    def animate_scaler_field(time_series, interval=50, cmap='viridis'):
        ts = np.array(time_series)

        fig, ax = plt.subplots(figsize=(8, 8))
        vmin, vmax = 0, 10
        heatmap = ax.imshow(ts[0], cmap=cmap, interpolation='nearest', vmin=vmin, vmax=vmax)
        cbar = fig.colorbar(heatmap, fraction=0.046, pad=0.04, ticks=[vmin, vmax], label='Value')

        def update(i):
            heatmap.set_array(ts[i])
            ax.set_title(f'Time step {i}')
            return [heatmap]

        anim = animation.FuncAnimation(fig, update, frames=len(ts), interval=interval, blit=False, repeat=True)

        plt.show(block=True)
        return anim