import numpy as np
from pyscf import ao2mo, scf, mcscf

# ======================================================================
def integrals(mol, mf_mc):
# The spin orbitals are mapped in the following way:
#       Orbital zero, spin up mapped to qubit 0
#       Orbital one,  spin up mapped to qubit 1
#       Orbital two,  spin up mapped to qubit 2
#            .
#            .
#       Orbital zero, spin down mapped to qubit norbs
#       Orbital one,  spin down mapped to qubit norbs+1
# one_body_integrals ---------------------------------------------------
  if isinstance(mf_mc, scf.hf.SCF):
    ncore = 0
    ncas = mf_mc.nao
  else:
    if isinstance(mf_mc, mcscf.casci.CASCI):
      ncore = mf_mc.ncore
      ncas = mf_mc.ncas
    else:
      print("Wrong input for integrals! Abort!")
      exit()
  nocc = ncore + ncas


  if type(mf_mc.mo_coeff) is tuple:
    mo_coeff = mf_mc.mo_coeff[0,ncore:nocc]
    mo_coeff_B = mf_mc.mo_coeff[1,ncore:nocc]
  else:
    if len(mf_mc.mo_coeff.shape) > 2:
      mo_coeff = mf_mc.mo_coeff[0,ncore:nocc]
      mo_coeff_B = mf_mc.mo_coeff[1,ncore:nocc]
    else:
      mo_coeff = mf_mc.mo_coeff[:,ncore:nocc]
      mo_coeff_B = None

  norbs = mo_coeff.shape[0]
  #print("mo_coeff = ", mo_coeff)
  #print("mo_coeff_B = ", mo_coeff_B)

  if type(mf_mc.mo_energy) is tuple:
    orbs_energy = mf_mc.mo_energy[0]
    orbs_energy_B = mf_mc.mo_energy[1]
  else:
    if len(mf_mc.mo_energy.shape) > 1:
      orbs_energy = mf_mc.mo_energy[0]
      orbs_energy_B = mf_mc.mo_energy[1]
    else:
      orbs_energy = mf_mc.mo_energy
      orbs_energy_B = None
  #print("orbs_energy = ", orbs_energy)
  #print("orbs_energy_B = ", orbs_energy_B)

  hij = mf_mc.get_hcore()

  mohij = np.dot(np.dot(mo_coeff.T, hij), mo_coeff)

  mohij_B = None
  if mo_coeff_B is not None:
    mohij_B = np.dot(np.dot(mo_coeff_B.T, hij), mo_coeff_B)

  if mohij_B is None:
    mohij_B = mohij

  norbs = mohij.shape[0]
  nspin_orbs = 2*norbs

  moh1_qubit = np.zeros([nspin_orbs, nspin_orbs])
  for p in range(nspin_orbs):  # pylint: disable=invalid-name
    for q in range(nspin_orbs):
      spinp = int(p/norbs)
      spinq = int(q/norbs)
      if spinp % 2 != spinq % 2:
        continue
      ints = mohij if spinp == 0 else mohij_B
      orbp = int(p % norbs)
      orbq = int(q % norbs)
      #if abs(ints[orbp, orbq]) > 1e-12:
      moh1_qubit[p, q] = ints[orbp, orbq]

# two_body_integrals ------------------------------------------------
  eri = mol.intor('int2e', aosym=1)
  #print("eri = ", eri)

  if isinstance(mf_mc, scf.hf.SCF):
    mo_eri = ao2mo.incore.full(mf_mc._eri, mo_coeff, compact=False)
  else:
    if isinstance(mf_mc, mcscf.casci.CASCI):
      mo_eri = ao2mo.full(mol, mo_coeff, compact=False)
    else:
      print("Wrong input for integrals! Abort!")
      exit()
  mohijkl = mo_eri.reshape(norbs, norbs, norbs, norbs)

  mohijkl_bb = None
  mohijkl_ba = None
  if mo_coeff_B is not None:
    mo_eri_B = ao2mo.incore.full(mf_mc._eri, mo_coeff_B, compact=False)
    mohijkl_bb = mo_eri_B.reshape(norbs, norbs, norbs, norbs)
    mohijkl_ba = ao2mo.incore.general(mf_mc._eri, (mo_coeff_B, mo_coeff_B, mo_coeff, mo_coeff), compact=False)

    mohijkl_ba = mohijkl_ba.reshape(norbs, norbs, norbs, norbs)

  ints_aa = np.einsum('ijkl->ljik', mohijkl)

  if mohijkl_bb is None or mohijkl_ba is None:
    ints_bb = ints_ba = ints_ab = ints_aa
  else:
    ints_bb = np.einsum('ijkl->ljik', mohijkl_bb)
    ints_ba = np.einsum('ijkl->ljik', mohijkl_ba)
    ints_ab = np.einsum('ijkl->ljik', mohijkl_ba.transpose())

  moh2_qubit = np.zeros([nspin_orbs, nspin_orbs, nspin_orbs, nspin_orbs])
  for p in range(nspin_orbs):  # pylint: disable=invalid-name
    for q in range(nspin_orbs):
      for r in range(nspin_orbs):
        for s in range(nspin_orbs):  # pylint: disable=invalid-name
          spinp = int(p/norbs)
          spinq = int(q/norbs)
          spinr = int(r/norbs)
          spins = int(s/norbs)
          if spinp != spins:
            continue
          if spinq != spinr:
            continue
          if spinp == 0:
            ints = ints_aa if spinq == 0 else ints_ba
          else:
            ints = ints_ab if spinq == 0 else ints_bb
          orbp = int(p % norbs)
          orbq = int(q % norbs)
          orbr = int(r % norbs)
          orbs = int(s % norbs)
          #if abs(ints[orbp, orbq, orbr, orbs]) > 1e-12:
          moh2_qubit[p, q, r, s] = -0.5*ints[orbp, orbq, orbr, orbs]

  return moh1_qubit, moh2_qubit
# ======================================================================
