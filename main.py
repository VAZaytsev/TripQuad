import cmath
import numpy as np
import math
pi = math.pi

# PySCF
import pyscf
from pyscf import gto, scf, mcscf, cc

from integrals_mod import *
from quantum_mod import *
import classical_mod

# Qiskit
from qiskit import *
from qiskit_nature.problems.second_quantization.electronic.builders.fermionic_op_builder import build_ferm_op_from_ints

from qiskit_nature.operators.second_quantization import FermionicOp
from qiskit_nature.converters.second_quantization import QubitConverter
from qiskit_nature.mappers.second_quantization import JordanWignerMapper, ParityMapper

from qiskit.quantum_info import Statevector

import itertools


#====================================================================

sz_tot = 0

dist = 0.5
x_dist = 0.5*dist*math.sqrt(3)
lbl = str(f'{x_dist: .3f}')
lbl += " 0 " + str(dist)


mol = gto.M(
        atom = 'H 0 0 0; H 0 0 0.735',
        #atom = 'Li 0 0 0; H 0 0 1.5',
        #atom = 'H 0 0 0; H'+lbl+'; H 0 0 0.735',
        #atom = 'H 0 0 0; H 0 0 1.24; H 0 1.24 1.24; H 0 1.24 0',
        basis = 'sto-3g',
        #basis = 'ccpvdz',
        #basis = 'ccpvtz',
        #basis = 'cc-pVQZ',
        #basis = 'cc-pV5Z',
        spin = sz_tot, verbose=0)


nao = mol.nao
nso = 2*nao
print("Number of atomic orbitals = ", nao, flush=True)
print("Number of spin orbitals = ", nso, flush=True)

nelectron = mol.nelectron
na = mol.nelec[0]
nb = mol.nelec[1]
print("Number of electrons = ", nelectron, "alpha = ", na,"beta = ", nb)


# Hartree-Fock
mf = scf.RHF(mol).run()
E_HF = mf.e_tot
print("\nHartree-Fock energy = ", E_HF, "[Ha]", flush=True)


# Nucleus interaction energy
E_Nuc = mf.energy_nuc()
print("Nucleus interaction energy = ",E_Nuc, "[Ha]")


# Complete CI in a given basis
mc_fci = mcscf.CASCI(mf, nao, nelectron).run(verbose=0)
E_FCI = mc_fci.e_tot
print("FCI electronic energy = ", E_FCI, "[Ha]")
print("Correlations = ", f'{abs(E_FCI - E_HF)*1.e3: <.3f}', "[mHa]",
      flush=True)


# CCSD
mc_ccsd = cc.CCSD(mf).run(verbose=0)
E_CCSD = mc_ccsd.e_tot
print("\nCCSD electronic energy = ", E_CCSD, "[Ha]")
print("diff with FCI = ", f'{abs(E_FCI - E_CCSD)*1.e3: <.3f}', "[mHa]",
      flush=True)
      

#exit()
# Calculate one- and two-body integrals
one_b_int, two_b_int = integrals(mol, mc_fci)


# Write the Hamiltonian in terms of the second quantization operators
H_op = build_ferm_op_from_ints(one_body_integrals=one_b_int,
                               two_body_integrals=two_b_int)


# Choose mapping and decide whether the reduction of qubits will
# be performed
mapper = JordanWignerMapper()
#mapper = ParityMapper()
reduction = False
qubit_converter = QubitConverter(mapper=mapper,
                                 two_qubit_reduction=reduction)


# Transform the Hamiltonian into operations on qubits
H_q = qubit_converter.convert(H_op, num_particles=nelectron)

Nq = H_q.num_qubits
print( "\n Nq = ", Nq, flush=True )


H_mtrx = H_q.to_matrix().real # for classical tests


# Initial guess of the ground state
occ_hf_a = [i for i in range(na)]
occ_hf_b = [nao + i for i in range(nb)] 
occ_hf = occ_hf_a + occ_hf_b


print("occupied orbitals = ", occ_hf)


psi_HF_circ = QuantumCircuit(Nq, Nq)
lbl = "0"*Nq
for o in occ_hf:
  psi_HF_circ.x(o)
  lbl = lbl[:Nq-1-o] + "1" + lbl[Nq-o:]
  
psi_HF_vec = Statevector.from_label(lbl).data

# Simple test that the average of the Hamiltonian on the HF
# state vector gives HF energy
print("< hf | H | hf > = ", psi_HF_vec.conj().T.dot( H_mtrx.dot(psi_HF_vec) ).item(0).real + E_Nuc)


# Measure H (Should differ from the Hartree-Fock by the statistical error)
av = average_of_H(psi_HF_circ, H_q)
print(av + E_Nuc, E_HF)


print( psi_HF_circ )


# Double excitations, the excitations can take place only 
# from occupied orbials to vacant ones
orbs_a = [i for i in range(nao)]
orbs_b = [nao+i for i in range(nao)]

list_a = list(itertools.combinations(orbs_a,na))
list_b = list(itertools.combinations(orbs_b,nb))


D_op = []
D_q = []
for occ_a in list_a:
  ann_a = [a for a in occ_hf_a if not a in occ_a]
  crt_a = [a for a in occ_a if not a in occ_hf_a]
  for occ_b in list_b:
    ann_b = [b for b in occ_hf_b if not b in occ_b]
    crt_b = [b for b in occ_b if not b in occ_hf_b]

    ann = ann_a + ann_b
    crt = crt_a + crt_b

    # in this place only the double excitations are retained
    if len(crt) != 2:
      continue


    T = FermionicOp("", register_length=Nq)
    for o in crt:
      T @= FermionicOp("+_"+str(o), register_length=Nq)
    for o in ann:
      T @= FermionicOp("-_"+str(o), register_length=Nq)
    T -= ~T

    T_q = qubit_converter.convert(T.reduce(), num_particles=nelectron)

    D_op.append(T)
    D_q.append(T_q)


# Create Quantum Circuit
Nparam = len(D_q)
print("Number of double excitations = ", len(D_q))


# Add double excitations
# Arbitrary values are assigned to the cluster amplitudes
Nstps = 50
for stp in range(Nstps):
  t_ang = -0.5*pi + stp*pi/Nstps


  ansatz_circ = QuantumCircuit(Nq, Nq)
  psi = psi_HF_vec.copy()
  for D in D_q:
    # Classical
    D_mtrx = (t_ang*D).to_matrix() # for classical tests
    exp_iT = classical_mod.exp_iH(-1j*D_mtrx)
    psi = exp_iT.dot(psi)

    # Quantum Simulator
    for x in D.primitive.to_list():
      ang = -1j * x[1] * t_ang
      ansatz_circ = ansatz_circ.compose(exp_alpha_PS_circ(ang.real, x[0]))


  av_Classical = psi.conj().T.dot( H_mtrx.dot(psi) ).item(0).real    
  av_Quantum = average_of_H(psi_HF_circ.compose(ansatz_circ), H_q)


  # The agreement should not be perfect due to the Trotterisation 
  # and statistical errors
  print(t_ang,
        av_Quantum + E_Nuc, 
        av_Classical + E_Nuc)
