
# coding: utf-8

# In[ ]:


import Classes as cg
import numpy as np
import matplotlib.pyplot as plt
import scipy as sc
from scipy import integrate
import sympy as sy
from scipy import linalg
import pylab as pl
import math as ma
import time 

# Fonctions ---------------------------------------------------------------

def temps(N,D,n_t_moy):
    n_t=np.zeros((N,D),int)
    T=np.random.beta(5,5,N)
    L=[[[] for j in range(D)] for i in range(N)]
    for i in range(N):
        if (n_t_moy*T[i]-2 >0):
            n_t[i,:]=np.random.poisson( (n_t_moy*T[i]) -2,D) +2
        else:
            n_t[i,:]=np.random.poisson( 0.1,D) +2
        for j in range(D):
            L[i][j]=np.sort( np.random.uniform(0,T[i],int(n_t[i,j])) )
    return (T,L,n_t);

#-----------------

def X_Params(varu,N,D):
    U=np.random.normal(0,np.sqrt(varu),N*D).reshape((N,D))
    V1=np.zeros((N,D,10))
    V2=np.zeros((N,D,10))
    for k in range(10):
        V1[:,:,k]=np.reshape(np.random.normal(0,2./(k+1),N*D),(N,D))
        V2[:,:,k]=np.reshape(np.random.normal(0,2./(k+1),N*D),(N,D))
    return (U,V1,V2);

#----------------

def Xf(t,u,v1,v2):
    if t.shape==():
        x=2*ma.pi*(np.arange(10)+1)*t
        res= sum( v1*np.sin(x)+v2*np.cos(x) )+u
    else:
        l=len(t)
        res=np.zeros(l)
        for i in range(l):
            x=2*ma.pi*(np.arange(10)+1)*t[i]
            res[i]=sum( v1*np.sin(x)+v2*np.cos(x) )+u
    return res;

#----------------

def X_obs(U,V1,V2,L,std):
    
    N=len(L)
    D=len(L[0])
    Xdata=[[[] for j in range(D)] for i in range(N)]
    for i in range(N):
        for j in range(D):
            if std>0:
                Xdata[i][j]=Xf(L[i][j],U[i,j],V1[i,j,:],V2[i,j,:])+np.random.normal(0,std,len(L[i][j]))
            else:
                Xdata[i][j]=Xf(L[i][j],U[i,j],V1[i,j,:],V2[i,j,:])
    return Xdata;

#----------------

def simu_YC(Alpha,beta,Z,U,V1,V2,T,Sigma):
    N=len(T)
    D=len(U[0])
    Y=np.zeros(N,float)
    for i in range(N):
        Y[i]=Z[i].dot(Alpha)
        for j in range(D):
            def Fbeta(t):
                return U[i,j]+np.sum(V1[i,j,:]*np.sin( (np.arange(10)+1.)*2.*ma.pi*t/100.)+V2[i,j,:]*np.cos((np.arange(10)+1.)*2.*ma.pi*t/100.))*beta.val([t,T[i]]);
            Y[i]= Y[i] + integrate.nquad(Fbeta,[[0,T[i]]],opts=[{'epsabs':5e-04}])[0] / T[i]
    # Erreur sur le label
    if Sigma>0:
        Y = Y + np.random.normal(0,Sigma,N)
    # Labels censurés et temps de censures
    C=np.random.normal(np.mean(Y)+np.std(Y)/2,np.sqrt(np.std(Y)),N)
    Yc=np.zeros(N,float)
    Delta=np.zeros(N)
    for i in range(N):
        if C[i]==min(C[i],Y[i]):
            Yc[i]=C[i]
            Delta[i]=0
        else:
            Yc[i]=Y[i]
            Delta[i]=1
    return (Y,C,Yc,Delta);