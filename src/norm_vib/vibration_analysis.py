import numpy as np
from numpy import linalg as LA
from .parser import pos_vec, read_xyz, mass_vec, file_to_hess

Eh = 4.3597447222071e-18 #in J
u = 1.66053906660e-27 #in kg
a0 = 5.29177210544e-11 #in m
c = 2.99792458e10 #in cm/s



# M^-1 matrix with unit 1/kg
def mmat(mass_vec):
    length = len(mass_vec)
    mmat = np.zeros((length, length))
    for i in range(length):
        mmat[i,i] = 1/mass_vec[i]
    return mmat


#returns frequency in 
def solve_eigenprob(hessian, mmat):
    hessian_si = hessian * Eh / (a0**2 * u) #convert to SI J/m^2
    sqrt_mmat = np.sqrt(mmat) # 1/sqrt(kg)
    massw_hessian = sqrt_mmat @ hessian_si @ sqrt_mmat # J/m^2 kg = 1/s^2
    eigenvalue, eigenvector = LA.eig(massw_hessian)
    threshold = 1e25 
    eigenvalue[np.abs(eigenvalue) < threshold] = 0 #correct numerical errors
    freq = np.sqrt(eigenvalue) # 1/s
    eigenvector = sqrt_mmat @ eigenvector
    wavenum = freq_2_wavenum(freq)
    print("The following Eigenmodes were found")
    print("Index \t wavenum/cm^-1")
    for i, nu in enumerate(wavenum):
        print(f"{i} \t {nu}")
    return freq, eigenvector


def freq_2_wavenum(freq):
    return freq /  (2 * np.pi * c) # in cm^-1

def complete_analysis(xyzfile, hessfile):
    xyz = read_xyz(xyzfile)
    hessian = file_to_hess(hessfile)
    massvector = mass_vec(xyz)
    big_m = mmat(massvector)
    result = solve_eigenprob(hessian, big_m)
    return result[0], result[1], massvector

def correct_com(posvec, massvec):
    mw_pos = posvec * massvec
    n_atoms = int(len(mw_pos)/3)
    three_full_mass = np.sum(massvec)
    com = np.zeros((3))
    for i in range(n_atoms):
        com += mw_pos[3*i:3*i+3]
    com = 3 * com / three_full_mass
    correction = np.tile(com, n_atoms)
    return posvec - correction


def is_approximately_unitary(A, tol=1e-6):
    # Compute the conjugate transpose (Hermitian conjugate) of A
    A_conj_T = np.conj(A).T
    
    # Check if A_conj_T * A is close to the identity matrix
    identity = np.eye(A.shape[0])
    return np.allclose(np.dot(A_conj_T, A), identity, atol=tol)