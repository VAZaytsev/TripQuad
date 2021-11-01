The very first version of the code, any notes, remarks, suggestions, etc. is welcome!

To launch just type 

```
python main.py
```

but be sure, that you have all the necessary libraries.

Package ```pyscf``` is used for all chemistry, it is powerful enough for the project. It can perform self-consistent calculations: Hartree-Fock (HF) and Multi-Configuration HF (MCHF). It can also perform Full Configuration Interaction (FCI) calculation and Coupled Cluster Singles Doubles (CCSD).

H2 molecule is used as a test system. For this system, the HF, FCI, and CCSD calculation is performed with the use of the STO-3G basis. Then the one- and two-body integrals are extracted and the Hamiltonian operator is constructed. As a next step, it is mapped to the operations on qubits.

Later all possible excitations are created, but only double excitations are left. For these excitations, the quantum circuit is created and launched on a quantum simulator. To compare, the calculations are performed by classical methods (matrix-vector multiplication).
