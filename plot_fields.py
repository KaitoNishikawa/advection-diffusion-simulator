import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation

class PlotFields():
    def plot_scaler_field(field):
        plt.figure(figsize=(8, 8))
        heatmap = plt.imshow(field, cmap='viridis', interpolation='nearest')
        plt.colorbar(heatmap, fraction=0.046, pad=0.04, label='Value')
        plt.show()
    
    def plot_vector_field(field_x, field_y, title, scale=100):
        x = np.arange(0, len(field_x)) # 0, 1, 2, ..., 99
        y = np.arange(0, len(field_x))
        X, Y = np.meshgrid(x, y)

        skip = 4
        X_plot = X[::skip, ::skip]
        Y_plot = Y[::skip, ::skip]
        Fx_plot = field_x[::skip, ::skip] * 500
        Fy_plot = field_y[::skip, ::skip] * 500

        plt.figure(figsize=(8, 8))
        plt.quiver(X_plot, Y_plot, Fx_plot, -Fy_plot, 
           scale=scale,  
           color='blue', 
           headwidth=5)
        plt.title(title)
        plt.gca().invert_yaxis()
        plt.show()

    def plot_sampled_vectors(field, x_coords, y_coords, gx, gy, title="Sampled Points", scale=100):
        plt.figure(figsize=(8, 8))
        
        heatmap = plt.imshow(field, cmap='viridis', interpolation='nearest', origin='lower')
        plt.colorbar(heatmap, fraction=0.046, pad=0.04, label='Value')
    
        plt.quiver(x_coords, y_coords, gx * 500, -gy * 500, 
                   scale=scale, color='red', headwidth=5, label='Gradients')        
        plt.title(title)
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.xlim(0, field.shape[1])
        plt.ylim(0, field.shape[0])
        plt.legend()
        plt.gca().invert_yaxis()
        plt.show()

    def compare_vector_fields(truth_x, truth_y, grad_pred, title1="Ground Truth", title2="GPR Prediction", scale=100):
        grid_shape = truth_x.shape
        x = np.arange(0, grid_shape[1])
        y = np.arange(0, grid_shape[0])
        X, Y = np.meshgrid(x, y)

        # Reshape grad_pred if it's flat
        if grad_pred.ndim == 2 and grad_pred.shape[1] == 2:
            pred_x = grad_pred[:, 0].reshape(grid_shape)
            pred_y = grad_pred[:, 1].reshape(grid_shape)
        else:
            # Handle case where it might already be split or wrong shape
            pred_x, pred_y = grad_pred[..., 0], grad_pred[..., 1]

        skip = 5 
        X_plot = X[::skip, ::skip]
        Y_plot = Y[::skip, ::skip]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

        # Truth
        ax1.quiver(X_plot, Y_plot, truth_x[::skip, ::skip] * 500, -truth_y[::skip, ::skip] * 500, 
                   scale=scale, color='blue', headwidth=5)
        ax1.set_title(title1)
        ax1.invert_yaxis()
        
        # Prediction
        ax2.quiver(X_plot, Y_plot, pred_x[::skip, ::skip] * 500, -pred_y[::skip, ::skip] * 500, 
                   scale=scale, color='green', headwidth=5)
        ax2.set_title(title2)
        ax2.invert_yaxis()

        plt.show()

    def animate_scaler_field(time_series, interval=1, cmap='viridis', repeat=False, auto_close=False, blit=False, step=4):
        ts = np.array(time_series)[::step]

        fig, ax = plt.subplots(figsize=(8, 8))
        vmin, vmax = np.min(np.abs(time_series[-1])), np.max(np.abs(time_series[-1]))
        # vmin, vmax = 0, 10
        heatmap = ax.imshow(ts[0], cmap=cmap, interpolation='nearest', vmin=vmin, vmax=vmax)
        cbar = fig.colorbar(heatmap, fraction=0.046, pad=0.04, label='Value')

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
    
    def animate_robot_path(field, path):
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Show the scalar field as a static background heatmap
        im = ax.imshow(field, cmap='viridis', origin='lower', interpolation='nearest')
        fig.colorbar(im, label='Intensity Value')
        
        # Initialize the path line and robot marker
        path_data = np.array(path)
        line, = ax.plot([], [], 'r-', linewidth=2, label='Robot Path')
        marker, = ax.plot([], [], 'ro', markersize=8, label='ASV Position')
        
        ax.set_title("Robot Path Overlay (Hotspot-Seeking)")
        ax.set_xlabel("X coordinate")
        ax.set_ylabel("Y coordinate")
        ax.invert_yaxis()
        ax.legend()

        def init():
            line.set_data([], [])
            marker.set_data([], [])
            return line, marker

        def update(frame):
            # Update line up to current frame
            line.set_data(path_data[:frame, 0], path_data[:frame, 1])
            # Update current position marker
            marker.set_data([path_data[frame, 0]], [path_data[frame, 1]])
            return line, marker

        anim = animation.FuncAnimation(fig, update, frames=len(path_data), 
                            init_func=init, blit=True, interval=100, repeat=False)
        
        plt.show()
        return anim