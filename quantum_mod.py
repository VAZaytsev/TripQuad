import math
pi = math.pi
from qiskit import *

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

Nshots = 10000

err_stat = 2/math.sqrt(Nshots)

backend = Aer.get_backend('qasm_simulator')

# ===================================================================
def ps2meas_basis(ps):
  nq = len(ps)
  circ = QuantumCircuit(nq,nq)

  for i in range(nq):
    if ps[i] == "X":
      circ.h(nq-1-i)
      continue

    if ps[i] == "Y":
      circ.rx(-0.5*pi,nq-1-i)
      continue

  return circ
# ===================================================================


# ===================================================================
def average_of_H(circ, H):
  nq = H.num_qubits

  average = 0
  for x in H.primitive.to_list():
    qc = circ.compose( ps2meas_basis(x[0]) )
    
    for i in range(nq):
      qc.measure(i,i)

    res = execute(qc, backend, shots=Nshots).result()
    counts = res.get_counts(qc)

    for k in counts.keys():
      pwr = sum( [int(k[ii]) if x[0][ii] != "I" else 
                  0 for ii in range(nq)] )

      average += (-1)**pwr * x[1].real * counts[k] / Nshots

  return average
# ===================================================================


# ===================================================================
def exp_alpha_PS_circ(alpha, ps):
# exp (i * alpha * sigma_i) = U U^t exp (i * alpha * sigma_i) U U^t
#                           = U exp (i * alpha * sigma_i_diag) U^t
  nq = len(ps)
  circ = QuantumCircuit(nq,nq)
  indx_q_cnot = []

# Ut step
  for i in range(nq):
    if ps[i] == "I" or ps[i] == "Z":
      continue

    indx_q_cnot.append(nq-1-i)

    if ps[i] == "X":
      circ.h(nq-1-i)
      continue

    if ps[i] == "Y":
      circ.rx(0.5*pi,nq-1-i)
      continue

  if len(indx_q_cnot) == 0:
    return circ

# Ladder of CNOT gates going down
  for i in range(len(indx_q_cnot)-1,0,-1):
    i1 = indx_q_cnot[i]
    i2 = indx_q_cnot[i-1]
    circ.cx(i1,i2)

# Rz rotation for the last qubit
  circ.rz(-2.0*alpha,indx_q_cnot[0])

# Ladder of CNOT gates going up
  for i in range(len(indx_q_cnot)-1):
    i1 = indx_q_cnot[i]
    i2 = indx_q_cnot[i+1]
    circ.cx(i2,i1)

# U step
  for i in range(nq):
    if ps[i] == "I" or ps[i] == "Z":
      continue
    if ps[i] == "X":
      circ.h(nq-1-i)
      continue
    if ps[i] == "Y":
      circ.rx(-0.5*pi,nq-1-i)
      continue

  return circ
# ===================================================================
