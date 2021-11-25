import numpy as np
import cmath

import inspect


# ===================================================================
class ansatz_cls:
  def __init__(self):
    self.layers = []

  def clc_nparams(self):
    self.nparams = sum([layer.get_num_params() 
                        for layer in self.layers])

  def act_on_vctr(self, args, vctr_in):
    vctr_out = vctr_in.copy()

    indx = 0 
    for layer in self.layers:
      nargs = layer.get_num_params()
      vctr_out = layer.act_on_vctr(args[indx:indx+nargs], vctr_out)
      indx += nargs
    return vctr_out


  def dact_on_vctr(self, args, vctr_in):
    res = [vctr_in.copy()]*len(args)

    indx = 0 
    for layer in self.layers:
      nargs = layer.get_num_params()

      for i in range(indx):
        res[i] = layer.act_on_vctr(args[indx:indx+nargs], res[i])

      if nargs != 0:
        res[indx:indx+nargs] = layer.dact_on_vctr(args[indx:indx+nargs], res[indx])

      tmp = layer.act_on_vctr(args[indx:indx+nargs], res[-1])
      for i in range(indx+nargs,len(args)):
        res[i] = tmp.copy()

      indx += nargs
    return res
# ===================================================================


# ===================================================================
class layer_cls:
  def __init__(self):
    self.rules = []
    self.ql = []
    self.qr = []
    self.drules = []

  
  def add_gate(self, ql, qr, mtrx_fun, dmtrx_fun=None):
    if ql < qr:
      exit("Wrong values for ql and qr")
      
    if any([qr in range(self.qr[i],self.ql[i]+1) 
            for i in range(len(self.qr))] ):
      exit("ql and qr overlap with previous matrices")

    self.rules.append( mtrx_fun )
    self.ql.append( ql )
    self.qr.append( qr )
    self.drules.append( dmtrx_fun )


  def get_num_params(self):
    nparam = 0
    for rule in self.rules:
      if inspect.isfunction(rule):
        nparam += len(inspect.signature(rule).parameters)

    return nparam

  def sort(self):
    indxs = sorted(range(len(self.qr)), key=lambda i: self.qr[i])
    self.rules = [self.rules[i] for i in indxs]
    self.ql = [self.ql[i] for i in indxs]
    self.qr = [self.qr[i] for i in indxs]
    self.drules = [self.drules[i] for i in indxs]


  def act_on_vctr(self, args, vctr):
    nq = (len(vctr)-1).bit_length()

    tmp = 1
    indx = 0
    l = -1
    for i,rule in enumerate(self.rules):
      if inspect.isfunction(rule):
        nargs = len(inspect.signature(rule).parameters)
        mtrx = rule(*tuple(args[indx:indx+nargs]))
        indx += nargs
      else:
        mtrx = rule

      tmp = np.kron( np.kron( mtrx, np.identity( 2**(self.qr[i]-1-l) ) ), tmp)
      l = self.ql[i]

    tmp = np.kron( np.identity( 2**(nq-1-l ) ), tmp )

    return tmp.dot(vctr).reshape(2**nq,1)


  def dact_on_vctr(self, args, vctr):
    nq = (len(vctr)-1).bit_length()
    
    res = [vctr.copy()] * len(args)

    indx = 0
    l = -1
    tmp_arr = [1] * len(args)
    for i,rule in enumerate(self.rules):
      nargs = 0
      if inspect.isfunction(rule):
        nargs = len(inspect.signature(rule).parameters)
        mtrx = rule(*tuple(args[indx:indx+nargs]))

        if self.drules[i] != None:
          dmtrx = self.drules[i](*tuple(args[indx:indx+nargs]))
          
          if nargs == 1:
            tmp_arr[indx] = np.kron( np.kron( dmtrx, np.identity( 2**(self.qr[i]-1-l) ) ), tmp_arr[indx])
          else:
            for ii,d in enumerate(dmtrx):
              j = ii + indx
              tmp_arr[j] = np.kron( np.kron( d, np.identity( 2**(self.qr[i]-1-l) ) ), tmp_arr[j])
        else:
          for ii in range(nargs):
            j = ii + indx
            tmp_arr[j] = np.zeros(( 2**(self.ql[i]+1), 2**(self.ql[i]+1) ))
      else:
        mtrx = rule


      for j,tmp in enumerate(tmp_arr):
        if j in range(indx,indx+nargs):
          continue
        tmp_arr[j] = np.kron( np.kron( mtrx, np.identity( 2**(self.qr[i]-1-l) ) ), tmp)

      indx += nargs
      l = self.ql[i]

    for i,tmp in enumerate(tmp_arr):
      tmp_arr[i] = np.kron( np.identity( 2**(nq-1-l ) ), tmp )
      res[i] = tmp_arr[i].dot( res[i] ).reshape(2**nq,1)
    
    return res
# ===================================================================


# ===================================================================
def UCC_ansatz(ClusterOps, exc):
  ansatz = ansatz_cls()

  def mtrx_fun(ClOp):
    def f(x):
      res = np.zeros( ClOp.P[0].shape, dtype=complex )
      for i, e in enumerate(ClOp.e_val):
        res += cmath.exp(e * x) * ClOp.P[i]
      return res
    return f

  def dmtrx_fun(ClOp):
    def f(x):
      res = np.zeros( ClOp.P[0].shape, dtype=complex )
      for i, e in enumerate(ClOp.e_val):
        res += e * cmath.exp(e * x) * ClOp.P[i]
      return res
    return f

  nq = ClusterOps[0].Tq.num_qubits

  for ClusterOp in ClusterOps:
    if ClusterOp.nex in exc:
      layer = layer_cls()
      layer.add_gate(nq-1, 0, 
                     mtrx_fun(ClusterOp), dmtrx_fun(ClusterOp))
      ansatz.layers.append(layer)

  ansatz.clc_nparams()
  return ansatz
# ===================================================================
