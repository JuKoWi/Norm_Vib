from norm_vib import (
    file_to_hess,
    complete_analysis,
    make_frames,
    read_xyz,
    pos_vec, 
    animate, 
    correct_com,
    is_approximately_unitary,
    do_plot
)
import numpy as np
"""
Idea of the project: 
    Doing vibrational analysis of a molecule based on its Hessian and calculating the Normal modes
    Displaying the Normal modes (ideally individually)

    - Core is the Normal mode analysis: It consists of diagonalizing a Hessian. Therefore the Hessian and information about masses
        - output are two arrays: one vector of the eigenvalues (elements of the diagonalized matrix) and the respective eigenvectors
        - definitely a class for the actual diagonalization

    - parser script to read job details from a config file
        - config file states path of hessian and xyz file
        -       

"""
def main():
    MOLECULE = "testo"
    NOSPEED = True
    NUM_VIB = 4  #29
    HEAVY_WIGGLE = 4
    hessfile = f"{MOLECULE}.hess"
    xyzfile = f"{MOLECULE}.xyz"
    posvec = pos_vec(read_xyz(xyzfile))
    eigenval, eigenvec, mass = complete_analysis(xyzfile, hessfile)
    posvec = correct_com(posvec, mass)
    frames = make_frames(eigenval, eigenvec, NUM_VIB, posvec, NOSPEED, HEAVY_WIGGLE)
    animate(frames, mass)
    do_plot(frames, mass)

if __name__ == "__main__":
    main()