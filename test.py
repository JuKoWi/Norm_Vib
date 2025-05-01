import openbabel
import pybel

# Read a molecule
mol = next(pybel.readfile("xyz", "your_molecule.xyz"))

# Access the underlying OBMol object
obmol = mol.OBMol

# Example: print symmetry classes
for atom in openbabel.OBMolAtomIter(obmol):
    print(f"Atom {atom.GetIdx()} has symmetry class {atom.GetSymmetryClass()}")
