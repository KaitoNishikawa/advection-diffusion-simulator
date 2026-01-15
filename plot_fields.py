import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation

class PlotFields():
    def plot_scaler_field(field):
        plt.figure(figsize=(8, 8))
        heatmap = plt.imshow(field, cmap='viridis', interpolation='nearest')
        plt.colorbar(heatmap, fraction=0.046, pad=0.04, ticks=[0, 10], label='Value')
        plt.show()
    
    def plot_vector_field(field_x, field_y, title, scale=100):
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
           scale=scale,  
           color='blue', 
           headwidth=5)
        plt.title(title)
        plt.show()

    def plot_sampled_vectors(x_coords, y_coords, gx, gy, title="Sampled Vectors"):
        plt.figure(figsize=(8, 8))
        plt.quiver(x_coords, y_coords, gx, gy, scale=100, color='red', headwidth=5)
        plt.title(title)
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.show()

    def compare_vector_fields(truth_x, truth_y, pred_x, pred_y, scale=100):
        x = np.arange(0, len(truth_x))
        y = np.arange(0, len(truth_x))
        X, Y = np.meshgrid(x, y)

        skip = 2 # Increased skip for side-by-side clarity
        X_plot = X[::skip, ::skip]
        Y_plot = Y[::skip, ::skip]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

        # Truth
        ax1.quiver(X_plot, Y_plot, truth_x[::skip, ::skip], truth_y[::skip, ::skip], 
                   scale=scale, color='blue', headwidth=5)
        ax1.set_title("Ground Truth Gradient")
        
        # Prediction
        ax2.quiver(X_plot, Y_plot, pred_x[::skip, ::skip], pred_y[::skip, ::skip], 
                   scale=scale, color='green', headwidth=5)
        ax2.set_title("GPR Predicted Gradient")

        plt.show()

    def animate_scaler_field(time_series, interval=1, cmap='viridis', repeat=False, auto_close=False, blit=False, step=2):
        ts = np.array(time_series)[::step]

        fig, ax = plt.subplots(figsize=(8, 8))
        vmin, vmax = 0, np.max(np.abs(time_series))
        # vmin, vmax = 0, 10
        heatmap = ax.imshow(ts[0], cmap=cmap, interpolation='nearest', vmin=vmin, vmax=vmax)
        cbar = fig.colorbar(heatmap, fraction=0.046, pad=0.04, ticks=[vmin, vmax], label='Value')

        def update(i):
            heatmap.set_array(ts[i])
            ax.set_title(f'Time step {i * step}')
            if auto_close and i == len(ts) - 1:
                try:
                    anim.event_source.stop()
                except Exception:
                    pass
                plt.close(fig)
            return [heatmap]
        anim = animation.FuncAnimation(fig, update, frames=len(ts), interval=interval, blit=blit, repeat=repeat)

        plt.show(block=True)
        return anim