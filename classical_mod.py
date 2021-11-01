import numpy as np
import cmath

# ======================================================================
def exp_iH(H):
  sz = H[:,0].size

  eigen_val, eigen_vec = np.linalg.eigh(H)

  mtrx_out = np.zeros((sz,sz), dtype=complex)
  for row in range(sz):
    P = np.outer(eigen_vec[:,row], np.conj(eigen_vec[:,row]) )
    mtrx_out += cmath.exp(1j*eigen_val[row]) * P

  return mtrx_out
# ======================================================================
