
# coding: utf-8

# In[4]:

from Classes import * 
from Simu import *
from NLL import *
import numpy as np
import matplotlib.pyplot as plt
import scipy as sc
from scipy import integrate
import sympy as sy
from scipy import linalg
import pylab as pl
import math as ma
import time 


# Classes et fonctions associées à l'évaluation de la negative-log-likelihood et ses dérivées partielles

# Fonctions ----------------------------
def Comp_Phi(base,T,J):
    N=len(T)
    args=np.zeros(((J+1)*N,2))
    args[:,1]=np.kron( np.ones(J+1,float) , T )
    args[:,0]=np.kron( np.ones(N,float) , np.arange(J+1)*1./J ) * args[:,1]
    Phi = np.apply_along_axis(base.val,axis=1,arr=args)
    return Phi;

# Classes ------------------------------
class KBeta:
    def __init__(self,L,T,k,Beta,J):
        self.k=k
        self.Beta=Beta
        self.L=L
        self.J=J
        self.T=T
    def val(self):
        N=len(self.L)
        D=len(self.L[0])
        Ints=[[ [] for j in range(D)] for i in range(N)]
        J=self.J
        for i in range(N):
            for j in range(D):
                Beta_ij= np.reshape ( self.Beta[(i*(J+1)):(i*(J+1)+J+1),j]  , (1,J+1) )
                t=self.L[i][j]
                grid=self.T[i]*1.*np.arange(J+1)/J
                vecarg=expandnp([t,grid])
                fonc_kij=self.k[i][j]
                K_ij=np.reshape( np.apply_along_axis(fonc_kij.val,axis=1,arr=vecarg) , (J+1,J+1) )
                K_ij[0,:]=K_ij[0,:]*0.5
                K_ij[J,:]=K_ij[J,:]*0.5
                Ints[i][j]=(self.T[i]/J)*Beta_ij.dot(K_ij)
        return Ints;
#------------------------------
# PhiK :
# Calcule, pour tout i de 1 à N et pour tout j de 1 à D, l'intégrale Phi(s,Ti)*K^{ijT}_*(s) ds entre 0 et Ti (il s'agit donc d'une matrice d'intégrales de taille (Kbeta**2,m_ij).
def PhiK(L,T,k,Phi_mat,J):
    N=len(L)
    D=len(L[0])
    Ints=[[ [] for j in range(D)] for i in range(N)]
    for i in range(N):
        for j in range(D):
            Phi_mat_i= Phi_mat[(i*(J+1)):(i*(J+1)+J+1),:].T  
            t=L[i][j]
            grid=T[i]*1.*np.arange(J+1)/J
            vecarg=expandnp([t,grid])
            fonc_kij=k[i][j]
            K_ij=np.reshape( np.apply_along_axis(fonc_kij.val,axis=1,arr=vecarg) , (J+1,len(t)) )
            K_ij[0,:]=K_ij[0,:]*0.5
            K_ij[J,:]=K_ij[J,:]*0.5
            Ints[i][j]=(T[i]/J)*Phi_mat_i.dot(K_ij)
    return Ints;
#------------------------------
# kbb :
# Calcule, pour tout i de 1 à N et pour tout j de 1 à D, l'intégrale k^i_j(s,t)*Beta_j(t,T_i)*Beta_j(s,T_i) dsdt sur [0,Ti]² (il s'agit d'un réel).
class kbb:
    def __init__(self,T,k,Beta,J):
        self.k=k
        self.Beta=Beta
        self.J=J
        self.T=T
    def val(self):
        N=len(self.k)
        D=len(self.k[0])
        Ints=[[ [] for j in range(D)] for i in range(N)]
        J=self.J
        for i in range(N):
            for j in range(D):
                Beta_ij= self.Beta[(i*(J+1)):(i*(J+1)+J+1),j]
                grid=self.T[i]*1.*np.arange(J+1)/J
                vecarg=expandnp([grid,grid])
                k_ij = np.apply_along_axis(self.k[i][j].val,axis=1,arr= vecarg).reshape(J+1,J+1)
                Beta_ij[0]=Beta_ij[0]*1./2
                Beta_ij[J]=Beta_ij[J]*1./2
                Ints[i][j]=  Beta_ij.dot(k_ij).dot(Beta_ij)  * 1.*self.T[i]**2/(J**2)

        return Ints; 

#------------------------------
# On définit une fonction qui calcule les matrices d'intégrales associées aux différentes dérivées secondes 
# de notre pénalité J22. Celle-ci utilise la dérivation symbolique automatique de sympy et l'intégration
# multiple scipy.integrate.dlbquad

def J22_Ints(Phi,Tmax):
    Kbeta=int(np.sqrt(len(Phi)))
    s=sy.Symbol('s')
    t=sy.Symbol('t')
    def gfun(x): return 0;
    def hfun(x): return x;
    
    Phi_mat=Phi*(Phi.T)
    
    Phi_mat_dsds=Phi_mat.diff(s,s)
    Phi_mat_dsdt=Phi_mat.diff(s,t)
    Phi_mat_dtdt=Phi_mat.diff(t,t)

    Is=np.zeros((Kbeta**2,Kbeta**2),float)
    Ic=np.zeros((Kbeta**2,Kbeta**2),float)
    It=np.zeros((Kbeta**2,Kbeta**2),float)
    for i in range(Kbeta**2):
        for j in range(Kbeta**2):
            func=sy.lambdify((s,t),Phi_mat_dsds[i,j],'numpy')
            Is[i,j]=integrate.dblquad(func, 0., Tmax, gfun, hfun, epsabs=1e-07)[0]
            func=sy.lambdify((s,t),Phi_mat_dsdt[i,j],'numpy')
            Ic[i,j]=integrate.dblquad(func, 0., Tmax, gfun, hfun, epsabs=1e-07)[0]
            func=sy.lambdify((s,t),Phi_mat_dtdt[i,j],'numpy')
            It[i,j]=integrate.dblquad(func, 0., Tmax, gfun, hfun, epsabs=1e-07)[0]
    
    return (Is,Ic,It);

