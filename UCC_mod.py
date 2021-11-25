import numpy as np

import itertools

from qiskit_nature.operators.second_quantization import FermionicOp
from qiskit_nature.converters.second_quantization import QubitConverter
from qiskit.opflow.primitive_ops.pauli_sum_op import PauliSumOp

# ===================================================================
class ClusterOperator():
  def __init__(self, ann, crt):
    self.ann = ann
    self.crt = crt
    self.nex = len(crt)
 
  def fer_op(self, nq):
    T = FermionicOp("", register_length=nq)
    for o in self.crt:
      T @= FermionicOp("+_"+str(o), register_length=nq)
    for o in self.ann:
      T @= FermionicOp("-_"+str(o), register_length=nq)
    T -= ~T
    self.T = T.reduce()

  def q_op(self, q_converter, num_part):
    self.Tq = q_converter.convert(self.T, num_particles=num_part)

  def mtrx(self):
    # the matrix is real and antihermitian (T^+ = -T)
    self.mtrx = self.Tq.to_matrix().real

    e_val, e_vec = np.linalg.eig(self.mtrx)

    self.e_val = [-1j, 0, 1j]
    self.P = [ np.zeros(e_vec.shape, dtype=complex) ] * 3
    for i,e in enumerate(e_val):
      for j,val in enumerate(self.e_val):
        if abs(e - val) < 1.e-13:
          out = np.outer( e_vec[:,i], np.conj(e_vec[:,i]) )
          self.P[j] = np.add(self.P[j],
                             np.outer( e_vec[:,i], 
                                      np.conj(e_vec[:,i]) 
                                      )
                             )
          break
# ===================================================================


# ===================================================================
def create_cluster_operators(psi_in_bn):
  nq = len(psi_in_bn)
  nao = int(nq/2)

  # occupied orbitals
  occ_in = sorted([i for i in range(nq) if psi_in_bn[nq-1-i] == "1"],
                  reverse = True)
  #print("occ_in = ", occ_in)

  # vacant orbitals
  vac = sorted([i for i in range(nq) if psi_in_bn[nq-1-i] == "0"],
               reverse = True)
  #print("vac = ", vac)

  # number of electrons
  Nelec = len(occ_in)

  na_in = sum([1 for o in occ_in if o < nao])
  nb_in = sum([1 for o in occ_in if o >= nao])
  #print("na_in = ", na_in, "nb_in = ", nb_in)

  cluster_ops = []
  #Evangelista order is used to excite from the reference state
  #The parity and momentum projection are conserved
  #nes if the deepest active electron
  for nes in range(1,Nelec+1):
    #print("Number of active electrons", nes)
    for nex in range(nes,0,-1):
      #print("Number of excitations = ", nex)
      if nex > len(vac):
        continue

      for add in list(itertools.combinations(occ_in[:nes-1],nex-1)):
        ann = sorted([occ_in[nes-1]] + list(add))
        
        for crt in list(itertools.combinations(vac,nex)):
          
          occ = [i for i in range(nq) if (i in crt) or 
                 ((i in occ_in) and (not i in ann))]
          na = sum([1 for o in occ if o < nao])
          nb = sum([1 for o in occ if o >= nao])

          if na != na_in or nb != nb_in:
            continue

          #print(ann, list(crt), occ)

          cluster_ops.append( ClusterOperator(list(crt), ann) )
          cluster_ops[-1].fer_op(nq)

  return cluster_ops
# ===================================================================
