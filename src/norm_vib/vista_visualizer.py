import numpy as np
import pyvista as pv

NUM_FRAMES = 1000
COLORS = {
    12: "#000000",          # black
    1.00784: "#808080",     # grey
    14.003074: "#0000ff",   # blue
    15.994914: "#D22B2B",   # red
    31.97207117: "#ffa500", # orange
    30.97376: "#dc143c",    # crimson
    18.998403: "#7fff00",   # chartreuse
    34.96885: "#ffff00",    # yellow
    79.904: "#ff8c00",      # darkorange
    126.9044726: "#8a2be2", # blueviolet
    55.93493554: "#f0e68c", # khaki
}


def do_plot(frames, mass):
    frames = convert_frames(frames)
    n_steps = np.shape(frames)[0]
    atoms = frames[0]  # positions of atoms at t=0
    n_atoms = int(len(mass)/3)
    hex_atom_colors = [COLORS[mass[3*i]] for i in range(n_atoms)]


    colors = np.array([hex_to_rgb(c) for c in hex_atom_colors])
    print(np.shape(colors))
    plotter = pv.Plotter()
    point_cloud = pv.PolyData(atoms)
    sphere_sizes = scale_to_range(mass, 1, 2)
    point_cloud["scales"] = np.array([sphere_sizes[3*i] for i in range(n_atoms)])

    # Add spheres to represent atoms for the initial frame
    glyphs= point_cloud.glyph(scale="scales", geom=pv.Sphere(radius=0.1))
    num_glyphs = len(glyphs.points)
    repeated_colors = np.repeat(colors, repeats=int(num_glyphs/n_atoms), axis=0)
    print(repeated_colors)
    glyphs.point_data['colors'] = repeated_colors
    # Add the initial spheres to the plot
    plotter.add_mesh(glyphs, scalars="colors", show_scalar_bar=False, rgb=True)

    # Function to update the positions of atoms in the animation
    def update_frame(i):
        new_positions = frames[i]
        point_cloud.points = new_positions
        glyphs = point_cloud.glyph(scale="scales", geom=pv.Sphere(radius=0.1))
        glyphs.point_data["colors"] = repeated_colors
        plotter.clear()
        plotter.add_mesh(glyphs, scalars="colors", show_scalar_bar=False, rgb=True)

    # Animate the motion over the time-dependent positions
    plotter.open_movie("atom_trajectory.gif")  # Save the animation to a video
    # plotter.advanced_time = True

    for i in range(n_steps):
        update_frame(i)
        plotter.write_frame()  # Write the frame to the video

    # Show the animation
    plotter.show()

def hex_to_rgb(hex_color):
    color = pv.Color(hex_color)
    return color.int_rgb
#convert the frames in PyVista format of shape (NUM_STEPS, N, 3)
def convert_frames(frames):
    n_cord, n_time = np.shape(frames)
    new_frames = np.zeros((n_time, int(n_cord/3), 3))
    frames = frames.T
    for i in range(n_time):
        new_frames[i] = np.reshape(frames[i], (-1,3))
    return new_frames

#time dependent positions in the shape (3N, NUM_FRAMES)
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
    
#time dependency of one coordinate
def coord_func(omega, eq_x, eigen_x):
    t = np.linspace(0, 10 * np.pi, NUM_FRAMES)
    pos_t = np.zeros((NUM_FRAMES))
    pos_t[:] = eq_x + eigen_x * np.sin(omega * t)
    return pos_t


#scale the size of the spheres according to masses
def scale_to_range(x, new_min, new_max):
    old_min = np.min(x)
    old_max = np.max(x)
    scaled = (x - old_min) / (old_max - old_min)  # normalize to [0,1]
    return scaled * (new_max - new_min) + new_min
