
# coding: utf-8

# In[1]:
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

# Fonctions ---------------------------------------------------------------

def temps(N,D,n_t_moy):
    n_t=np.random.poisson(n_t_moy,N)
    M_t=np.zeros((N,D))
    T=np.random.beta(5,5,N)
    L=[[[] for j in range(D)] for i in range(N)]
    for i in range(N):
        M_t[i,:]=np.random.poisson(n_t[i],D)
        it=0
        while np.min(M_t[i,:])<=2 and it<=100:
            it=it+1
            M_t[i,:]=np.random.poisson(n_t[i],D)
        for j in range(D):
            L[i][j]=np.sort( np.random.uniform(0,1,int(M_t[i,j])) )
            stock =L[i][j][L[i][j]<T[i]]
            if len(stock)<=2:
                T[i]=np.max(np.concatenate((np.array([T[i]]),L[i][j]),axis=0))
            else:
                L[i][j]=stock
                M_t[i,j]=len(L[i][j])
                
    return (T,L,M_t);

#-----------------

def X_Params(varu1,varu2,N,D):
    u=np.random.normal(0,varu1,N)
    U=np.zeros((N,D))
    for i in range(N):
        U[i,:]=np.random.normal(0,varu2,D)
    V1=np.zeros((N,D,10))
    V2=np.zeros((N,D,10))
    for k in range(10):
        V1[:,:,k]=np.reshape(np.random.normal(0,4./(k+1)**2,N*D),(N,D))
        V2[:,:,k]=np.reshape(np.random.normal(0,4./(k+1)**2,N*D),(N,D))
    return (U,V1,V2);

#----------------

def X_obs(U,V1,V2,L):
    from Classes import * 
    N=len(L)
    D=len(L[0])
    var_mes=1
    Xdata=[[[] for j in range(D)] for i in range(N)]
    for i in range(N):
        for j in range(D):
            Xdata[i][j]=X_fonc(L[i][j],U[i,j],V1[i,j,:],V2[i,j,:]).val()+np.random.normal(0,var_mes,len(L[i][j]))
    return Xdata;


#----------------

def simu_YC(Alpha,Kbeta,b,Z,U,V1,V2,T,Sigma):
    N=len(T)
    D=len(b[0])
    t=sy.Symbol('t')
    s=sy.Symbol('s')
    syPhi=sy.ones(Kbeta**2,1)
    syb=sy.ones(1,Kbeta*Kbeta)
    k=0
    for i in range(Kbeta):
        for j in range(Kbeta):
            syPhi[k]=(t**i)*(s**j) 
            syb[k]=sy.Symbol('b'+str(k))
            k+=1
    syBeta=syb*syPhi

    Y=np.zeros(N,float)
    for i in range(N):
        Y[i]=Z[i].dot(Alpha)
        for j in range(D):
            replacements=[ (sy.sympify('b'+str(k)),b[k,j]) for k in range(Kbeta**2)]
            replacements.append((s,T[i]))
            syBeta_subs=syBeta.subs(replacements)[0]
        betaf=sy.lambdify(t,syBeta_subs,'numpy')
        def Fbeta(t):
            return U[i,j]+np.sum(V1[i,j,:]*np.sin( (np.arange(10)+1.)*2.*ma.pi*t/100.) +V2[i,j,:]*np.cos((np.arange(10)+1.)*2.*ma.pi*t/100.))*betaf(t);
        
        Y[i]= Y[i] + integrate.nquad(Fbeta,[[0,T[i]]],opts=[{'epsabs':5e-04}])[0] / T[i]

    Y = Y + np.random.normal(0,Sigma,N)
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