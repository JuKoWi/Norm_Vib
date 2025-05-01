import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from .vibration_analysis import complete_analysis



"""
The information for a certain animation is the following
    - list of atoms and types with equilibrium positions
    - eigenvector 
    - eigen-frequency

The 
"""

NUM_FRAMES = 100
t = np.linspace(0, 10 * np.pi, NUM_FRAMES)
COLORS = {
   12: "black",
   1.00784: "grey",
   14.003074: "blue" ,
   15.994914: "red" ,
   31.97207117: "orange" ,
   30.97376: "crimson" , 
   18.998403: "chartreuse",
    34.96885: "yellow",
    79.904: "darkorange",
   126.9044726: "blueviolet",
   55.93493554: "khaki"
    }

# the following function takes the full information necessary to animate: the position-frames and the mass-vector
def animate(frames, massvec):
    sphere_sizes = scale_to_range(massvec, 10, 25)
    num_spheres = int(np.shape(frames)[0] / 3)
    num_frames = np.shape(frames)[1]
    data = frames.T
    max_dist = np.max(np.abs(data))


    # Create a figure and a 3D axis
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(-max_dist, max_dist)
    ax.set_ylim(-max_dist, max_dist)
    ax.set_zlim(-max_dist, max_dist)

    spheres = [ax.plot([], [], [], color=COLORS[massvec[3*_]], marker='o', markersize=sphere_sizes[3*_])[0] for _ in range(num_spheres)]
    def update(frame):
        ax.cla()  # Clear the axis to remove trails
        ax.set_xlim(-max_dist, max_dist)
        ax.set_ylim(-max_dist, max_dist)
        ax.set_zlim(-max_dist, max_dist)
        for i, sphere in enumerate(spheres):
            sphere, = ax.plot(data[frame, 3*i], data[frame, 3*i+1], data[frame, 3*i+2], color=COLORS[massvec[3*i]], marker='o', markersize=sphere_sizes[3*i])
        return spheres

    anim = FuncAnimation(fig, update, frames=num_frames, interval=50, blit=False)
    plt.show()


def make_frames(eigenval, eigenvec, vib_index, pos_vec, nospeed=True, heavywiggle=1):
    for i, vec in enumerate(eigenvec):
        vec *= heavywiggle * vec / np.max(np.abs(vec))
    num_comp = len(eigenval)
    omega = eigenval[vib_index]
    if nospeed and omega != 0:
        omega = 1
    frames =[]
    for i in range(num_comp):
        pos = coord_func(omega, pos_vec[i], eigenvec[vib_index, i])
        frames.append(pos)
    frames = np.array(frames)
    return frames
    

def coord_func(omega, eq_x, eigen_x):
    t = np.linspace(0, 10 * np.pi, NUM_FRAMES)
    pos_t = np.zeros((NUM_FRAMES))
    pos_t[:] = eq_x + eigen_x * np.sin(omega * t)
    return pos_t



def scale_to_range(x, new_min, new_max):
    old_min = np.min(x)
    old_max = np.max(x)
    scaled = (x - old_min) / (old_max - old_min)  # normalize to [0,1]
    return scaled * (new_max - new_min) + new_min
