import numpy as np

PSE = {
    "C": 12, 
    "H": 1.00784, 
    "N": 14.003074, 
    "O": 15.994914, 
    "S": 31.97207117, 
    "P": 30.97376, 
    "F": 18.998403,
    "Cl": 34.96885,
    "Br": 79.904,
    "I": 126.9044726,
    "Fe": 55.93493554
    }

def parse_hess(filepath: str):
    hessian_content = ""
    with open(filepath, "r") as f:
        lines = f.readlines()
        reading = False
        for line in lines:

            if "$hessian" in line:
                reading = True
                continue

            if "$vibrational_frequencies" in line:
                reading = False
                break
            
            if reading:
                hessian_content += line
    return hessian_content


# find more flexible way, that also works for incomplete block
def make_hess_mat(hess_string: str):

    rows = hess_string.splitlines()
    dim = int(rows[0])
    del rows[0]

    for i in range(len(rows)):
        if (i % dim) == 0:
            del rows[i]

    block_count = 0
    hessian_numbers = []
    final_mat = None
    for i, row in enumerate(rows):
        numbers = row.split()
        del numbers[0]
        hessian_numbers.append([float(num) for num in numbers])
        
        if ((i+1) % dim) == 0:
            if block_count == 0:
                final_mat = np.array(hessian_numbers)
            else:
                hessian_numbers = np.array(hessian_numbers)
                final_mat = np.append(final_mat, hessian_numbers, axis=1)
            block_count += 1 
            hessian_numbers = []

    return final_mat, dim

def custom_reshape(mat, shape: int):
    num_block = int(np.shape(mat)[0]/shape)
    new_mat = mat[:shape, :]
    for i in range(1, num_block):
        add_mat = mat[shape*i : shape*(i+1), :]
        new_mat = np.append(new_mat, add_mat, axis=1)
    return new_mat

def file_to_hess(filename: str):
    mat = make_hess_mat(parse_hess(filename))[0]
    return mat 

# return array with first column masses and next three columns xyz components
def read_xyz(filename):
    data = []
    with open(filename, "r") as f:
        lines = f.readlines()
        del lines[:2]
        for line in lines:
            data.append(line.split())
    data = np.array(data)
    for i in range(np.shape(data)[0]):
        data[i, 0] = float(PSE[data[i,0]])
    data = np.array(data, dtype=float)
    return data

def pos_vec(mat):
    num_atoms = int(np.shape(mat)[0])
    mat = np.delete(mat, 0, axis=1)
    mat = np.reshape(mat, (num_atoms*3))
    return mat

def mass_vec(mat):
    mat = np.delete(mat, (1,2,3), axis=1)
    mat = np.repeat(mat, 3)
    return mat

