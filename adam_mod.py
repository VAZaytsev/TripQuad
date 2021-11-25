import numpy as np

class Adam_cls():
  def __init__(self, n, 
               eta=0.01, 
               beta1=0.9, 
               beta2=0.999, 
               epsilon=1.e-8):
    self.name = "Adam"

    self.m = np.zeros(n)
    self.v = np.zeros(n)

    self.eta = eta # learning rate
    self.beta1 = beta1
    self.beta2 = beta2
    self.epsilon = epsilon
    
    self.t = 1
    self.f = None
    self.converged = False
    

  def update(self, x, dx):
    # biased first momentum estimate
    self.m = self.beta1*self.m + (1-self.beta1)*dx

    # biased second raw momentum estimate
    self.v = self.beta2*self.v + (1-self.beta2)*(dx**2)

    # bias-corrected first momentum estimate
    m_corr = self.m/(1-self.beta1**self.t)
    
    # bias-corrected second raw momentum estimate
    v_corr = self.v/(1-self.beta2**self.t)

    # update
    self.t += 1
    
    return x - self.eta*(m_corr/(np.sqrt(v_corr) + self.epsilon))
