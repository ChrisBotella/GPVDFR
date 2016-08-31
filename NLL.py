
# coding: utf-8

# In[4]:

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
import pyGPs


# fonctions associées à l'évaluation de la negative-log-likelihood et ses dérivées partielles

# Fonctions ----------------------------
def Comp_Phi(Phi_fonc,T,J):
    N=len(T)
    Kbeta=len(Phi_fonc)
    args=np.zeros(((J+1)*N,2))
    args[:,1]=np.kron( T , np.ones(J+1,float)  )
    args[:,0]=np.kron( np.ones(N,float) , np.arange(J+1)*1./J ) * args[:,1]
    Phi=np.zeros(((J+1)*N,Kbeta))
    for b in range(Kbeta):
        Phi[:,b] = np.apply_along_axis( Phi_fonc[b],0,*(args.T) ) 
    return Phi;

#------------------------------
# PhiK :
# Calcule, pour tout i de 1 à N et pour tout j de 1 à D, l'intégrale Phi(s,Ti)*K^{ijT}_*(s) ds entre 0 et Ti 
# et retouner donc une matrice d'intégrales de taille (Kbeta,m_ij) où Kbeta est la taille de la base fonctionelle.
def PhiK(L,T,k,Phi_mat,J):
    N=len(L)
    D=len(L[0])
    Ints=[[ [] for j in range(D)] for i in range(N)]
    for i in range(N):
        for j in range(D):
            Phi_mat_i= Phi_mat[(i*(J+1)):(i*(J+1)+J+1),:].T  
            t=L[i][j]
            grid=T[i]*1.*np.arange(J+1)/J
            vecarg=cg.expandnp([t,grid])
            K_ij= np.apply_along_axis(k[i][j],0,*(vecarg.T)).reshape(J+1,len(t))
            K_ij[0,:]=K_ij[0,:]*0.5
            K_ij[J,:]=K_ij[J,:]*0.5
            Ints[i][j]=(T[i]/J)*Phi_mat_i.dot(K_ij)
    return Ints;
#------------------------------
# kbb :
# Calcule, pour tout i de 1 à N et pour tout j de 1 à D, les intégrales k^i_j(s,t)*Phi(t,T_i)*Beta_j(s,T_i) dsdt (vecteur de taille Kbeta) et k^i_j(s,t)*Beta_j(t,T_i)*Beta_j(s,T_i) dsdt sur [0,Ti]² (il s'agit d'un réel).
def kbb(T,k,Phi_mat,b,J):
    N=len(k)
    D=len(k[0])
    I2=[[ [] for j in range(D)] for i in range(N)]
    I3=np.zeros((N,D,len(b)),float)
    Beta=Phi_mat.dot(b)
    for i in range(N):
        for j in range(D):
            Beta_ij= Beta[(i*(J+1)):(i*(J+1)+J+1),j]
            # on calcule pour k_ij, la matrice des k(s,t) pour s,t dans la grille de [0,Ti]
            grid=T[i]*1.*np.arange(J+1)/J
            vecarg=cg.expandnp([grid,grid])
            k_ij = np.apply_along_axis(k[i][j],0, *(vecarg.T)).reshape(J+1,J+1)
            k_ij[0,:]=k_ij[0,:]*0.5
            k_ij[J,:]=k_ij[J,:]*0.5
            k_ij[:,0]=k_ij[:,0]*0.5
            k_ij[:,J]=k_ij[:,J]*0.5
            # On calcule alors l'intégrale Phi(s,Ti) k(s,t) Beta(t,Ti) sur [0,Ti]²
            I3[i,j,:]=  Beta_ij.dot(k_ij).dot(Phi_mat[(i*(J+1)):(i*(J+1)+J+1),:])*(T[i]**2) /(J**2)
            # Ainsi que l'intégrale k(s,t) Beta(s,Ti) Beta(t,Ti) sur [0,Ti]²
            #I2[i][j]=  Beta_ij.dot(k_ij).dot(Beta_ij)  * 1.*(T[i]**2)/(J**2)
            I2[i][j]=  I3[i,j,:].dot(b)
    return (I2,I3); 

#---------------------------------


##############################################################################################
#--------------------------------- Fonctions concernant L_YX_noC ------------------------------
##############################################################################################

#-----------------------------

# Calculs d'inférence GP sur les données fonctionnelles, et des quantités nécessaires au calcul de E(Y|X) et V(Y|X)  
def Pre_Comp_YX(L,T,Xdata,Y,Kbeta,J):
    t=sy.Symbol('t')
    s=sy.Symbol('s')
    # Récupération des variables et paramètres 
    N=len(L)
    D=len(L[0])
    # ----------------- Construction de la base fonctionnelle 
    syPhi=sy.ones(Kbeta**2,1)
    syb=sy.ones(1,Kbeta**2)
    v=[np.arange(Kbeta),np.arange(Kbeta)]
    expo=cg.expandnp(v)
    Phi_fonc=[ [] for j in range(Kbeta**2) ]
    for x in range(len(expo[:,0])):
        syPhi[x]=(t**expo[x,0])*(s**expo[x,1])
        Phi_fonc[x]=sy.lambdify((t,s),syPhi[x],'numpy')
        syb[x]=sy.Symbol('b'+str(x))
    syBeta=syb*syPhi
    Phi_mat=Comp_Phi(Phi_fonc,T,J)
    I_pen=J22_fast(syPhi,np.max(T),50)[3]
    # ----------------- Construction des noyaux et leurs dérivées
    # Construction de la forme du noyau
    el1=sy.Symbol('el1')
    per1=sy.Symbol('per1')
    sig1=sy.Symbol('sig1')
    args1=[el1,per1,sig1]
    el2=sy.Symbol('el2')
    sig2=sy.Symbol('sig2')
    args2=[el2,sig2]
    syk= cg.sy_Periodic((s,t),*args1) + cg.sy_RBF((s,t), *args2)
    args= [t,s] + args1 + args2
    # Dérivation et construction des fonctions vectorielles associées 
    k_fonc= sy.lambdify(tuple(args),syk,'numpy')
    n_par=len(args)-2
    k_der=[ [] for i in range(n_par) ]
    for i in range(n_par):
        func=syk.diff(args[i+2])
        k_der[i]=sy.lambdify(tuple(args),func,'numpy')
    return (Phi_mat,k_fonc,k_der,I_pen);

def Comp_Psi(L,k_fonc,Theta,Gamma,i_subset=None):
    N=len(L)
    D=len(L[0])
    Psi=[ [ [] for j in range(D)] for i in range(N)]
    Det=np.ones((N,D),float)
    if i_subset is None:
        veci=range(N)
    else:
        veci=i_subset
    for i in veci:
        for j in range(D):
            t_ij=L[i][j]
            m_ij=len(t_ij)
            vec=cg.expandnp([t_ij,t_ij])
            args= list(vec.T) + list(Theta[i,j,:])            
            K=np.apply_along_axis(k_fonc,0,*args).reshape((m_ij,m_ij))
            Psi[i][j]=sc.linalg.inv(K+np.eye(m_ij)*Gamma[i,j]**2)
            Det[i,j]=sc.linalg.det(K+np.eye(m_ij)*Gamma[i,j]**2,check_finite=False)
            if Det[i,j]==0.:
                Det[i,j]=0.01
    return (Psi,Det);

def Comp_Pen(I_pen,b,D,Kbeta):
    # Pénalité et dérivées partielle
    dPen=np.zeros(D*Kbeta,float)
    Pen=0.
    for j in range(D):
        dPen[j*Kbeta:(j+1)*Kbeta]=2*I_pen.dot(b[j*Kbeta:(j+1)*Kbeta]).reshape(-1)
        Pen+=0.5*b[j*Kbeta:(j+1)*Kbeta].dot(dPen[j*Kbeta:(j+1)*Kbeta])
    return (Pen,dPen);

# Version qui ne renvoie pas le gradient relatif à Sigma2
def L_YX(P,Z,L,T,Xdata,Y,Phi_mat,k_fonc,k_der,Stochastic=False,Gradient=True,I_pen=None,Lambda=None,i_liste=None):
    # Récupération des variables et paramètres 
    N=len(L)
    D=len(L[0])
    p=Z.shape[1]
    Kbeta=Phi_mat.shape[1] # Kbeta est ici la taille de la base complète
    J=int(-1+Phi_mat.shape[0]/N)
    Alpha=P[0:p]
    b=P[p:(D*Kbeta+p)].reshape((Kbeta,D))
    Sigma2=1.
    Eta=P[(D*Kbeta+p):(N*D+D*Kbeta+p)].reshape( (N,D) )# Sous forme d'array de taille (N,D)
    Gamma=P[(N*D+D*Kbeta+p):(2*N*D+D*Kbeta+p)].reshape( (N,D) )# Sous forme d'array de taille (N,D)
    n_par=len(k_der)
    Theta=P[(2*N*D+D*Kbeta+p):((2+n_par)*N*D+D*Kbeta+p)].reshape( (N,D,n_par) )# Sous forme d'array de taille (N,D,n_par)
    # --------------------------Options    
    if Stochastic:
        # Si l'option stochastique est activée, on ne tient pas compte de la liste de i soumise
        I=np.random.randint(0,N)
        i_liste=[I]
    # Calcul des Psi, on ne calcule que ceux dont l'indice doit être évalué
    if i_liste is not None:
        (Psi,Det)=Comp_Psi(L,k_fonc,Theta,Gamma,i_subset=i_liste)
        seq_i=i_liste
        isit= np.array([ (k in i_liste) for k in range(N) ])
    else:
        (Psi,Det)=Comp_Psi(L,k_fonc,Theta,Gamma)
        seq_i=range(N)
        isit=np.ones(N,float)
    # Vecteurs contenant respectivement les espérances de Yi | Xi notées Ei et leurs variances Vi
    (EY,VY)=(np.zeros(N,float),np.ones(N,float))
    Beta=Phi_mat.dot(b)
    # Matrices contenant les Int_[0,Ti] E(F_ij(s)|X_ij)Beta_j(s,Ti) ds et les les Int_[0,Ti]² Cov(F_ij(s),F_ij(t)|X_ij) Beta_j(s,Ti)Beta_j(t,Ti) dsdt
    (EY_Int,VY_Int)=(np.zeros((N,D),float),np.ones((N,D),float))
    # Matrice des dEi/dbj et dVi/dbj
    (dEdb,dVdb)=( np.zeros((N,D,Kbeta),float),np.zeros((N,D,Kbeta),float) )
    # Gradient
    (dEdEta,dLXdEta)=( np.zeros((N,D),float), np.zeros((N,D),float) )
    (dEdGamma,dVdGamma,dLXdGamma)=( np.zeros((N,D),float),np.zeros((N,D),float),np.zeros((N,D),float)) 
    (dEdTheta,dVdTheta,dLXdTheta) = ( np.zeros((N,D,n_par),float),np.zeros((N,D,n_par),float),np.zeros((N,D,n_par),float) )
    Dkij=np.zeros((n_par,J+1,J+1),float)
    XPsiX= np.zeros((N,D),float)
    #On prépare la matrice de rankisation
    Id=np.eye(Kbeta)
    add_diag=np.zeros(D,float)
    for j in range(D):
        add_diag[j]=b[:,j].dot(Id).dot(b[:,j])
    # Calcul des quantités pour dL/dEtaij
    Grad=np.zeros(len(P),float)
    Un=np.ones(J+1,float)
    Un[0]=0.5
    Un[J]=0.5    
    for i in seq_i:
        grid=T[i]*1.*np.arange(J+1)/J
        vecarg=cg.expandnp([grid,grid])
        Beta_i=Beta[(i*(J+1)):(i*(J+1)+J+1),:]
        Phi_mat_i= Phi_mat[(i*(J+1)):(i*(J+1)+J+1),:].T
        for j in range(D):        
            Xij=Xdata[i][j]
            t_ij=L[i][j]
            m_ij=len(t_ij)
            # Moyenne de F_ij 
            Etaij=Eta[i,j]
            # Evaluation du noyau k_ij sur la grille
            k_ij=np.apply_along_axis(k_fonc,0,*(list(vecarg.T) + list(Theta[i,j,:]) ) ).reshape((J+1,J+1))
            k_ij[0,:]=k_ij[0,:]*0.5
            k_ij[J,:]=k_ij[J,:]*0.5
            k_ij[:,0]=k_ij[:,0]*0.5
            k_ij[:,J]=k_ij[:,J]*0.5
            vecarg2=cg.expandnp([t_ij,grid])
            K_ij= np.apply_along_axis(k_fonc,0,*(list(vecarg2.T) + list(Theta[i,j,:]) ) ).reshape(J+1,m_ij)
            K_ij[0,:]=K_ij[0,:]*0.5
            K_ij[J,:]=K_ij[J,:]*0.5
            KPsi=K_ij.dot(Psi[i][j])
            PsiX=Psi[i][j].dot(Xij-Etaij)
            # Calcul de la composante de Ei selon la jème VFT
            Phi_EspFij= Phi_mat_i.dot(  Etaij*Un + KPsi.dot(Xij-Etaij) ) /J
            EY_Int[i,j]= b[:,j].dot( Phi_EspFij ) 
            # Calcul de la composante de Vi selon la jème VFT
            V_i= Phi_mat_i.dot( k_ij - KPsi.dot(K_ij.T) ).dot( Phi_mat_i.T )
            Phi_CovFij= V_i.dot( b[:,j] ) /(J**2)
            # Ajout d'une matrice diagonale faible pour rankiser V^i_j
            extra_butter= add_diag[j]*0.01*np.trace(V_i)/Kbeta
            VY_Int[i,j]= b[:,j].dot(Phi_CovFij) #+ extra_butter
            XPsiX[i,j]= (Xij-Etaij).dot(PsiX)
            
            if Gradient:
                # Calcul de l'intégrale de dEi/dbj
                dEdb[i,j,:]=Phi_EspFij
                # Calcul de l'intégrale de dEi/dbj
                dVdb[i,j,:]=2*Phi_CovFij
                # Evaluation de dérivées du noyau sur la grille 
                vecarg3=cg.expandnp([t_ij,t_ij])
                DK_ij=np.zeros((n_par,J+1,m_ij),float)
                DKt_ij=np.zeros((n_par,m_ij,m_ij),float)
                for npa in range(n_par):
                    Dkij[npa,:,:]= np.apply_along_axis( k_der[npa],0,*(list(vecarg.T) + list(Theta[i,j,:]) ) ).reshape((J+1,J+1))
                    DK_ij[npa,:,:]=np.apply_along_axis(k_der[npa],0,*(list(vecarg2.T) + list(Theta[i,j,:]) ) ).reshape(J+1,m_ij)
                    DKt_ij[npa,:,:]=np.apply_along_axis(k_der[npa],0,*(list(vecarg3.T) + list(Theta[i,j,:]) ) ).reshape(m_ij,m_ij)
                Dkij[:,0,:]=Dkij[:,0,:]*0.5
                Dkij[:,J,:]=Dkij[:,J,:]*0.5
                Dkij[:,:,0]=Dkij[:,:,0]*0.5
                Dkij[:,:,J]=Dkij[:,:,J]*0.5
                DK_ij[:,0,:]=DK_ij[:,0,:]*0.5
                DK_ij[:,J,:]=DK_ij[:,J,:]*0.5
                # Calcul des quantités pour dL/dThetaij
                BetaDK_ij= Beta_i[:,j].dot(DK_ij) 
                BetaKPsiDKt_ij= Beta_i[:,j].dot( np.transpose(KPsi.dot(DKt_ij),(1,0,2)) )
                dEdTheta[i,j,:]= ( BetaDK_ij + BetaKPsiDKt_ij ).dot(PsiX) /J
                dVdTheta[i,j,:]= ( Beta_i[:,j].dot( Dkij ) + ( - 2* BetaDK_ij  + BetaKPsiDKt_ij  ).dot(KPsi.T) ).dot(Beta_i[:,j]) /J**2
                dLXdTheta[i,j,:] = - PsiX.T.dot(DKt_ij).dot(PsiX) + np.trace( DKt_ij.dot(Psi[i][j]) ,axis1=1, axis2=2 )
                # Calcul des quantités pour dL/dGammaij
                dEdGamma[i,j]= Beta_i[:,j].dot(KPsi.dot(PsiX)) /J
                dVdGamma[i,j]= Beta_i[:,j].dot( k_ij - KPsi.dot(KPsi.T) ).dot(Beta_i[:,j]) /J**2
                dLXdGamma[i,j]= -(PsiX.T).dot(PsiX) + np.trace( Psi[i][j] )
                # Calcul des quantités pour dL/dEtaij
                dEdEta[i,j]= Beta_i[:,j].dot( Un +   KPsi.dot(np.ones(m_ij)) ) / J
                dLXdEta[i,j]=-2*PsiX.T.dot(np.ones(m_ij))+2*np.ones(m_ij).dot(Psi[i][j]).dot(np.ones(m_ij))
            
        # On calcule l'espérance de Y sachant X avec l'espérance de Y et le terme de conditionnement gaussien
        EY[i]= Z[i].dot(Alpha) + np.sum( EY_Int[i,:] ) 
        # On calcule la variance de Y sachant X avec le terme de variance issu des VFT X^i_j et de l'erreur résiduelle 
        VY[i]= np.sum(  VY_Int[i,:] ) + Sigma2 
    # Calcul de la vraisemblance     
    LYX = np.sum( isit*( (Y-EY)**2/VY + np.log(VY) ) ) + np.sum( XPsiX.reshape(-1) )  + np.sum(np.log(Det.reshape(-1)))
    if Lambda is not None:
        (Pen,dPen)=Comp_Pen(I_pen,b.reshape(-1),D,Kbeta)
        LYX+=Lambda*Pen
    if Gradient:
        # Calcul du Gradient    
        a= isit*(  -2*(Y-EY)/VY )
        b= isit*( (np.ones(N,float) - (Y-EY)**2/VY )/VY  )
        # Par rapport à Alpha
        Grad[0:p]= np.ones(N,float).dot( a.repeat(p,axis=0).reshape((N,p)) *Z).reshape(-1)  
        # Par rapport à b_j pour tout j de 1 à D
        dL_YlXdb=np.zeros((Kbeta,D))
        A=a.repeat(Kbeta,axis=0).reshape((N,Kbeta))
        B=b.repeat(Kbeta,axis=0).reshape((N,Kbeta))
        for j in range(D): 
            Grad[(p+j*Kbeta):(p+(j+1)*Kbeta)]=np.ones(N,float).dot( A*dEdb[:,j,:]+B*dVdb[:,j,:]) 
        if Lambda is not None:
            Grad[p:(p+D*Kbeta)]+=Lambda*dPen
        # Par rapport à Eta, Theta et Gamma
        # Eta
        A=(cg.expandnp([ a , np.ones(D) ]).T)[0].reshape(D,N).T
        B=(cg.expandnp([ b , np.ones(D) ]).T)[0].reshape(D,N).T 
        Grad[(D*Kbeta+p):(N*D+D*Kbeta+p)]= ( A*dEdEta + dLXdEta ).reshape(-1)
        # Gamma
        Grad[(D*Kbeta+p+N*D):(D*Kbeta+p+2*N*D)]= ( A*dEdGamma + B*dVdGamma + dLXdGamma ).reshape(-1)
        # Theta
        A=np.transpose( (cg.expandnp([ a , np.ones(n_par*D) ]).T)[0].reshape(n_par,D,N), (2,1,0) )
        B=np.transpose( (cg.expandnp([ b , np.ones(n_par*D) ]).T)[0].reshape(n_par,D,N), (2,1,0) )
        Grad[(p+D*Kbeta+2*N*D):((2+n_par)*N*D+D*Kbeta+p)]= (A*dEdTheta + B*dVdTheta + dLXdTheta).reshape(-1)
        # On renvoie le critère ainsi que le gradient
        if Stochastic:
            return (LYX , Grad , I );
        else:
            return (LYX , Grad , None); 
    else:
        if Stochastic:
            return (LYX , None , I);
        else:
            return (LYX , None , None);
        
##############################################################################################
#--------------------------------- Fonctions concernant L_YlX_noC ------------------------------
##############################################################################################


# Calculs d'inférence GP sur les données fonctionnelles, et des quantités nécessaires au calcul de E(Y|X) et V(Y|X)  
def Pre_Comp(L,T,Xdata,Y,Kbeta,J):
    t=sy.Symbol('t')
    s=sy.Symbol('s')
    # Récupération des variables et paramètres 
    N=len(L)
    D=len(L[0])
    # ----------------- INFERENCES DES PARAMS LONGITUDINAUX
    #print("[   ] Inférence des paramètres fonctionnels")
    model = pyGPs.GPR()
    kern1=pyGPs.cov.RBF(log_ell=0.0, log_sigma=0.0) 
    kern2=pyGPs.cov.Periodic(log_ell=0., log_p=0., log_sigma=0.)
    kern=pyGPs.cov.SumOfKernel(kern1,kern2)
    m=pyGPs.mean.Const()
    model.setPrior(mean=m,kernel=kern) 
    model.setNoise(log_sigma=-2.30258)
    Theta=np.zeros((N,D,len(model.covfunc.hyp)),float)
    Gamma=np.zeros((N,D),float)
    moy_est=np.zeros((N,D),float)
    for i in range(N):
        for j in range(D):
            y=np.asarray(Xdata[i][j])
            x=np.asarray(L[i][j])
            try:
                model.optimize(x, y)
                moy_est[i,j]=model.meanfunc.hyp[0]
                Theta[i,j,:]=np.array(np.exp(model.covfunc.hyp))
                Gamma[i,j]=np.exp(model.likfunc.hyp)
            except:
                #Problème d'inférence, paramètres défaut attribués
                moy_est[i,j]=np.mean(x)
                Theta[i,j,:]=np.array([.05,np.std(x)**2,.05,1.,0.])
                Gamma[i,j]=1.
                pass
                
    # ----------------- RECUPERATION DES QUANTITES D'INTERET
    #print("[-  ]  Récupération des quantités d'intérêt")
    # Construction de la forme du noyau
    el1=sy.Symbol('el1')
    sig1=sy.Symbol('sig1')
    args1=[el1,sig1]
    el2=sy.Symbol('el2')
    per2=sy.Symbol('per2')
    sig2=sy.Symbol('sig2')
    args2=[el2,per2,sig2]
    syk= cg.sy_RBF((s,t), *args1) + cg.sy_Periodic((s,t),*args2)
    args= [t,s] + args1 + args2
    k_fonc= sy.lambdify(tuple(args),syk,'numpy')
    Psi=Comp_Psi(L,k_fonc,Theta,Gamma)[0]
    # ----------------- Construction de la base fonctionnelle 
    #print("[-- ]  Calcul des quantités d'intérêt")
    syPhi=sy.ones(Kbeta**2,1)
    syb=sy.ones(1,Kbeta**2)
    v=[np.arange(Kbeta),np.arange(Kbeta)]
    expo=cg.expandnp(v)
    Phi_fonc=[ [] for j in range(Kbeta**2) ]
    for x in range(len(expo[:,0])):
        syPhi[x]=(t**expo[x,0])*(s**expo[x,1])
        Phi_fonc[x]=sy.lambdify((t,s),syPhi[x],'numpy')
        syb[x]=sy.Symbol('b'+str(x))
    syBeta=syb*syPhi
    I_pen=J22_fast(syPhi,np.max(T),50)[3]
    # ----------------- Construction de l et V
    Un=np.ones(J+1,float)
    Un[0]=0.5
    Un[J]=0.5
    Phi_mat=Comp_Phi(Phi_fonc,T,J)
    l=np.zeros((N,D*Kbeta**2),float)
    V=[[] for i in range(N)]
    vl=[ [] for j in range(D) ]
    for i in range(N):
        Phi_i=Phi_mat[(i*(J+1)):(i*(J+1)+J+1),:].T
        for j in range(D):  
            Xij=Xdata[i][j]
            # Moyenne de F_ij estimée en amont
            Etaij=moy_est[i,j]
            t=L[i][j]
            grid=T[i]*1.*np.arange(J+1)/J
            vec=cg.expandnp([t,grid])
            args= list(vec.T) + list(Theta[i,j,:])     
            K_ij= np.apply_along_axis(k_fonc,0,*args).reshape(J+1,len(t))
            K_ij[0,:]=K_ij[0,:]*0.5
            K_ij[J,:]=K_ij[J,:]*0.5
            KPsi=K_ij.dot(Psi[i][j])
            l[i,(j*Kbeta**2):((j+1)*Kbeta**2)]= Phi_i.dot( KPsi.dot(Xij-Etaij) + Etaij*Un ).reshape(-1) /J
            # on calcule pour k_ij, la matrice des k(s,t) pour s,t dans la grille de [0,Ti]
            vec=cg.expandnp([grid,grid])
            args= list(vec.T) + list(Theta[i,j,:])     
            k_ij = np.apply_along_axis(k_fonc,0,*args).reshape(J+1,J+1)
            k_ij[0,:]=k_ij[0,:]*0.5
            k_ij[J,:]=k_ij[J,:]*0.5
            k_ij[:,0]=k_ij[:,0]*0.5
            k_ij[:,J]=k_ij[:,J]*0.5
            Cov_FF=k_ij-KPsi.dot(K_ij.T)
            vl[j]=Phi_i.dot(Cov_FF).dot(Phi_i.T)/J**2
        V[i]=sc.sparse.block_diag(tuple(vl))
        # On ajoute une matrice diagonale pour rendre V[i] définie positive, mais on fait en sorte que ses valeurs propres soient petites par rapport à celles de V[i] pour ne pas trop affecter la vraisemblances sur l'espace autour de 0.
        #V[i]=V[i]+np.eye(D*Kbeta**2)*0.01*np.trace(V[i].toarray())/(D*Kbeta**2)
    return (l,V,I_pen);

def L_YlX(P,Z,V,l,Y,D):
    # Récupération des variables et paramètres
    N=len(l[:,0])
    p=Z.shape[1]
    Kbeta=len(V[0].toarray()[0,:])/D # Kbeta est ici la taille de la base complète
    Sigma2=1.
    Alpha=P[0:p]
    b=P[p:(D*Kbeta+p)]
    L_YlX=0.
    dL_YlXda=np.zeros(p,float)
    dL_YlXdb=np.zeros(len(b),float)
    for i in range(N):
        demiVYi=V[i].toarray().dot(b)
        VYi=Sigma2 + b.dot(demiVYi)
        Yi_EYi=Y[i]-Z[i,:].dot(Alpha)-l[i,:].dot(b)
        L_YlX+=  (Yi_EYi)**2/VYi + np.log(VYi)
        dL_YlXda+= -2*Z[i,:]*Yi_EYi/VYi
        dL_YlXdb+= (2./VYi)*( -l[i,:]*Yi_EYi + demiVYi*(1-Yi_EYi**2/VYi ) ) 
    return ( L_YlX ,np.concatenate((dL_YlXda,dL_YlXdb)) );


def L_YlX_pen(P,Z,V,l,Y,D,I_pen,Lambda):
    # Récupération des variables et paramètres
    N=len(l[:,0])
    p=Z.shape[1]
    Kbeta=len(V[0].toarray()[0,:])/D # Kbeta est ici la taille de la base complète
    Sigma2=1.
    Alpha=P[0:p]
    b=P[p:(D*Kbeta+p)]
    dPen=np.zeros(D*Kbeta,float)
    Pen=0.
    for j in range(D):
        dPen[j*Kbeta:(j+1)*Kbeta]=2*I_pen.dot(b[j*Kbeta:(j+1)*Kbeta]).reshape(-1)
        Pen+=0.5*b[j*Kbeta:(j+1)*Kbeta].dot(dPen[j*Kbeta:(j+1)*Kbeta])
    L_YlX=0.
    dL_YlXda=np.zeros(p,float)
    dL_YlXdb=np.zeros(len(b),float)
    for i in range(N):
        demiVYi=V[i].toarray().dot(b)
        VYi=Sigma2 + b.dot(demiVYi)
        Yi_EYi=Y[i]-Z[i,:].dot(Alpha)-l[i,:].dot(b)
        L_YlX+=  (Yi_EYi)**2/VYi + np.log(VYi) + Lambda*Pen
        dL_YlXda+= -2*Z[i,:]*Yi_EYi/VYi
        dL_YlXdb+= (2./VYi)*( -l[i,:]*Yi_EYi + demiVYi*(1-Yi_EYi**2/VYi ) ) + Lambda*dPen
    return ( L_YlX ,np.concatenate((dL_YlXda,dL_YlXdb)) );

def S_L_YlX_pen(P,Z,V,l,Y,D,I_pen,Lambda):
    # Récupération des variables et paramètres
    N=len(l[:,0])
    i=np.random.randint(0,N) 
    p=Z.shape[1]
    Kbeta=len(V[0].toarray()[0,:])/D # Kbeta est ici la taille de la base complète
    Sigma2=1.
    Alpha=P[0:p]
    b=P[p:(D*Kbeta+p)]
    dPen=np.zeros(D*Kbeta,float)
    Pen=0.
    for j in range(D):
        dPen[j*Kbeta:(j+1)*Kbeta]=2*I_pen.dot(b[j*Kbeta:(j+1)*Kbeta]).reshape(-1)
        Pen+=0.5*b[j*Kbeta:(j+1)*Kbeta].dot(dPen[j*Kbeta:(j+1)*Kbeta])

    demiVYi=V[i].toarray().dot(b)
    VYi=Sigma2 + b.dot(demiVYi)
    Yi_EYi=Y[i]-Z[i,:].dot(Alpha)-l[i,:].dot(b)
    L_YlX=  (Yi_EYi)**2/VYi + np.log(VYi) + Lambda*Pen
    dL_YlXda= -2*Z[i,:]*Yi_EYi/VYi
    dL_YlXdb= (2./VYi)*( -l[i,:]*Yi_EYi + demiVYi*(1-Yi_EYi**2/VYi ) ) + Lambda*dPen
    return ( L_YlX ,np.concatenate((dL_YlXda,dL_YlXdb)), i  );

def S_L_YlX_pen_V(P,Z,V,l,Y,D,I_pen,Lambda,i):
    # Récupération des variables et paramètres
    N=len(l[:,0])
    i=np.random.randint(0,N) 
    p=Z.shape[1]
    Kbeta=len(V[0].toarray()[0,:])/D # Kbeta est ici la taille de la base complète
    Sigma2=1.
    Alpha=P[0:p]
    b=P[p:(D*Kbeta+p)]
    Pen=0.
    for j in range(D):
        Pen+=0.5*b[j*Kbeta:(j+1)*Kbeta].dot(I_pen.dot(b[j*Kbeta:(j+1)*Kbeta]).reshape(-1))
    demiVYi=V[i].toarray().dot(b)
    VYi=Sigma2 + b.dot(demiVYi)
    Yi_EYi= Y[i]-Z[i,:].dot(Alpha)-l[i,:].dot(b)
    L_YlX=  (Yi_EYi)**2/VYi + np.log(VYi) + Lambda*Pen
    return L_YlX;

#####################################################################################################
#---------------------------------- Méthode des Moindres Carrés ------------------------------------
#####################################################################################################

# Critère des moindres carrés et gradient
def SCE_pen(P,Z,V,l,Y,D,I_pen,Lambda):
    # Récupération des variables et paramètres
    N=len(l[:,0])
    p=Z.shape[1]
    Kbeta=len(V[0].toarray()[0,:])/D # Kbeta est ici la taille de la base complète
    Sigma2=1.
    Alpha=P[0:p]
    b=P[p:(D*Kbeta+p)]
    dPen=np.zeros(D*Kbeta,float)
    Pen=0.
    for j in range(D):
        dPen[j*Kbeta:(j+1)*Kbeta]=2*I_pen.dot(b[j*Kbeta:(j+1)*Kbeta]).reshape(-1)
        Pen+=0.5*b[j*Kbeta:(j+1)*Kbeta].dot(dPen[j*Kbeta:(j+1)*Kbeta])
    Y_EY=Y-Z.dot(Alpha)-l.dot(b)
    SCE=  (Y_EY.T).dot(Y_EY) + Lambda*Pen
    dSCEda= -2*(Z.T).dot(Y_EY)
    dSCEdb=  -2.*(l.T).dot(Y_EY) + Lambda*dPen
    grad=np.concatenate((dSCEda,dSCEdb))
    return ( SCE ,grad );


# Estimateur des moindres carrés + pénalité ridge (avec inverse de moore penrose)
def MCO_pen_est(Z,l,Y,I_pen,Lambda):
    N=len(l[:,0])
    p=Z.shape[1]
    X=np.hstack((Z,l))
    tI_pen=sc.linalg.block_diag(np.zeros((p,p),float),I_pen)
    P_est= np.linalg.pinv(X.T.dot(X)+Lambda*tI_pen).dot(X.T).dot(Y)
    return P_est;
def MCO_pen_best(l,Y,I_pen,Lambda):
    b_est= np.linalg.pinv(l.T.dot(l)+Lambda*I_pen).dot(l.T).dot(Y)
    return b_est;


# Estimateur des moindres carrés
def MCO_est(Z,l,Y):
    X=np.hstack((Z,l))
    P_est= np.linalg.pinv(X).dot(Y)
    return P_est;
# Estimateur des moindres carrés de b
def MCO_best(l,Y):
    b_est=np.linalg.pinv(l).dot(Y)
    return b_est;


# Estimateur des moindres carrés + pénalité sur variance Y|X (avec inverse de moore penrose)
def MCO_penvar_est(Z,V,l,Y):
    p=Z.shape[1]
    # Récupération des variables et paramètres
    X=np.hstack((Z,l))
    Vp=sc.linalg.block_diag( np.zeros((p,p),float) , sum(V).toarray() ) 
    P_est= np.linalg.pinv(X.T.dot(X)+Vp).dot(X.T).dot(Y)
    return P_est;
def MCO_penvar_best(V,l,Y):
    # Récupération des variables et paramètres
    Vp= sum(V).toarray()  
    b_est= np.linalg.pinv(l.T.dot(l)+Vp).dot(l.T).dot(Y)
    return b_est;


def MCO_2pen_est(Z,V,l,Y,I_pen,Lambda):
    p=Z.shape[1]
    # Récupération des variables et paramètres
    X=np.hstack((Z,l))
    Vp=sc.linalg.block_diag( np.zeros((p,p),float) , sum(V).toarray() ) 
    tI_pen=sc.linalg.block_diag(np.zeros((p,p),float),I_pen)
    Pen=Vp+Lambda*tI_pen
    P_est= np.linalg.pinv(X.T.dot(X)+Pen).dot(X.T).dot(Y)
    return P_est;
def MCO_2pen_best(Z,V,l,Y,I_pen,Lambda):
    Vp= sum(V).toarray()
    Pen=Vp+Lambda*I_pen
    b_est= np.linalg.inv(l.T.dot(l)+Pen).dot(l.T).dot(Y)
    return b_est;

            
#####################################################################################################
#------------------------ Algorithme de descente de gradient bricolé -------------------------------
#####################################################################################################

# args : liste d'arguments supplémentaires à transmettre à la fonction
def GD_switch(func,Po,args,max_it=20,reach=1,depth=11,verbose=False):
    Pn=Po+np.zeros(len(Po),float)
    seq_entiers=[1,3]
    if reach-depth<-10:
        depth=reach+10
    gamme=np.zeros(depth*len(seq_entiers))
    vec=reach-np.fliplr(np.atleast_2d(np.arange(depth)))[0]
    k=0
    for i in vec:
        for j in seq_entiers:
            gamme[k]=j*10**i
            k+=1
    l=len(gamme)
    stock=np.zeros(l,float)
    vraisemblances=[0.]
    for i in range(max_it):
        V=func(Pn,*args)
        if verbose:
            print("Norme b "+str(np.sqrt(Pn.dot(Pn)))+" Vrais "+ str(V[0])+" ecart moyen à 0 dans gradient : "+ str(np.sqrt(V[1].dot(V[1]))/len(Pn) )  )
        vraisemblances.append(V[0])
        switch=1
        j=0
        while switch==1 and j<=l-1:
            Pstock=Pn-gamme[j]*V[1]/np.sqrt(V[1].dot(V[1]))
            stock[j]=func(Pstock,*args)[0]        
            if j>=1 and stock[j]>=stock[j-1]:
                switch=0
            j+=1
        which_min=[ x for x in range(l) if stock[x]==np.nanmin(stock) ][0]
        Pn=Pn-gamme[which_min]*V[1]/np.sqrt(V[1].dot(V[1]))
    del vraisemblances[0]
    return (Pn,V[0],V[1],vraisemblances);

def GD(Po, func, args,func_V=None,max_it=20,reach=1,depth=11,verbose=False):
    Pn=Po+np.zeros(len(Po),float)
    seq_entiers=[1,3]
    if reach-depth<-10:
        depth=reach+10
    gamme=np.zeros(depth*len(seq_entiers))
    vec=reach-np.fliplr(np.atleast_2d(np.arange(depth)))[0]
    k=0
    for i in vec:
        for j in seq_entiers:
            gamme[k]=j*10**i
            k+=1
    longueur=len(gamme)
    stock=np.zeros(longueur,float)
    vraisemblances=[0.]
    for i in range(max_it):
        V=func(Pn,*args)
        if i==0:
            curr_V=V[0]
        if verbose:
            em_b=np.sqrt(Pn.dot(Pn)/len(Pn))
            em_g=np.sqrt(V[1].dot(V[1])/len(Pn))
            print("Ecart moyen b :"+str(em_b)+" Vrais "+str(curr_V)+" Ecart moyen gradient :"+str(em_g))
        vraisemblances.append(curr_V)
        for j in range(longueur):
            Pstock=Pn-gamme[j]*V[1]/np.sqrt(V[1].dot(V[1]))
            if func_V==None:
                stock[j]=func(Pstock,*args)[0]
            else:
                if len(V)==3:
                    stock[j]=func_V( Pstock,*(args + [V[2]]) )
                else:
                    stock[j]=func_V(Pstock,*args)
        which_min=[ x for x in range(longueur) if stock[x]==np.nanmin(stock) ][0]
        Pn=Pn-gamme[which_min]*V[1]/np.sqrt(V[1].dot(V[1]))
        curr_V=stock[which_min]
    del vraisemblances[0]
    return (Pn,curr_V,V[1],vraisemblances);

def GD_new(Po,func, args,func_V=True,max_it=20,reach=1,depth=11,verbose=False):
    Pn=Po+np.zeros(len(Po),float)
    seq_entiers=[1,3]
    if reach-depth<-10:
        depth=reach+10
    gamme=np.zeros(depth*len(seq_entiers))
    vec=reach-np.fliplr(np.atleast_2d(np.arange(depth)))[0]
    k=0
    for i in vec:
        for j in seq_entiers:
            gamme[k]=j*10**i
            k+=1
    longueur=len(gamme)
    stock=np.zeros(longueur,float)
    vraisemblances=[0.]
    for i in range(max_it):
        V=func(Pn,**args)
        if i==0:
            curr_V=V[0]
        if verbose:
            em_b=np.sqrt(Pn.dot(Pn)/len(Pn))
            em_g=np.sqrt(V[1].dot(V[1])/len(Pn))
            print("Ecart moyen b :"+str(em_b)+" Vrais "+str(curr_V)+" Ecart moyen gradient :"+str(em_g))
        vraisemblances.append(curr_V)
        for j in range(longueur):
            Pstock=Pn-gamme[j]*V[1]/np.sqrt(V[1].dot(V[1]))
            if not func_V:
                stock[j]=func(Pstock,**args)[0]
            else:
                if V[2] is None:
                    stock[j]=func(Pstock,**dict(args.items()+ {'Gradient':False}.items() )  )[0]
                else:
                    dd=dict(args)
                    if 'Stochastic' in dd:
                        del dd['Stochastic']
                    stock[j]=func( Pstock,**dict(dd.items()+{'Gradient':False,'i_liste':[V[2]]}.items() ) )[0]
                    
        which_min=[ x for x in range(longueur) if stock[x]==np.nanmin(stock) ][0]
        Pn=Pn-gamme[which_min]*V[1]/np.sqrt(V[1].dot(V[1]))
        curr_V=stock[which_min]
    del vraisemblances[0]
    return (Pn,curr_V,V[1],vraisemblances);

def GD_rate(Po,func, args,max_it=20,verbose=False,tol=1.):
    Pn=Po+np.zeros(len(Po),float)
    vraisemblances=[0.]
    evalpente=np.zeros(5,float)
    for i in range(5):
        evalpente[i]= np.max( func(Pn,**args)[1] )
    rate_ref=1./np.max(evalpente)
    rate=1./np.max(evalpente)
    for i in range(max_it):
        V=func(Pn,**args)
        # Si l'écart moyen du gradient à 0 passe en dessous de tol, on termine
        em_g=np.sqrt(V[1].dot(V[1])/len(Pn))
        curr_V=V[0]
        vraisemblances.append(curr_V)
        if em_g<tol:
            del vraisemblances[0]
            return (Pn,curr_V,V[1],vraisemblances);
        
        if verbose:
            em_b=np.sqrt(Pn.dot(Pn)/len(Pn))
            em_g=np.sqrt(V[1].dot(V[1])/len(Pn))
            print("Ecart moyen b :"+str(em_b)+" Vrais "+str(curr_V)+" Ecart moyen gradient :"+str(em_g))
            
        # Si le déplacement moyen s'emballe, on diminue drastiquement le taux d'apprentissage
        if rate*np.mean(V[1])>1e2:
            rate_ref=rate_ref/(1e2)
        Pn=Pn-rate*V[1]
        rate=rate_ref * (1.- i/max_it)
    
    del vraisemblances[0]
    return (Pn,curr_V,V[1],vraisemblances);

# Procédure pour lancer plusieurs initialisations aléatoires d'une routine d'optimisation
def multi_init(optim_func,P,D,p,Kbeta,args,n_ini=10):
    Po=P+np.zeros(len(P),float)
    Pf=[[ [] for j in range(n_ini) ] for i in range(4)]
    for i in range(n_ini):
        Po[p:(D*Kbeta**2+p)]= np.random.uniform(-10,10,D*Kbeta**2)
        (Pf[0][i],Pf[1][i],Pf[2][i],Pf[3][i])= optim_func(Po,**args)
    which_min=[ x for x in range(n_ini) if Pf[1][x]==np.nanmin(Pf[1]) ][0]
    return (Pf[0][which_min],Pf[1][which_min],Pf[2][which_min],Pf[3][which_min]);

# Procédure d'inférence par SGD pour L_YX
def LYX_optim(Z,l,L,T,Xdata,Y,Phi_mat,k_fonc,k_der,I_pen,Lambda,n_it=100,n_ini=2,tol=0.5):
    N=len(Y)
    D=len(L[0])
    p=Z.shape[1]
    n_par=len(k_der)
    # taille totale de la base
    Kbeta=int(np.sqrt( Phi_mat.shape[1] ))
    # Construction du paramètre
    N_param=p+D*Kbeta**2+(2+n_par)*N*D
    Po=np.zeros( N_param,float )
    Po[(N*D+D*Kbeta**2+p):N_param]=np.ones((1+n_par)*D*N, float )
    bf=MCO_pen_best(l,Y,I_pen,Lambda)
    Pi=np.concatenate((np.zeros(p,float),bf,Po[(p+D*Kbeta**2):N_param]))
    (Pl, Vl , Gl, vrais )=( [[]for i in range(2)],[[]for i in range(2)],[[]for i in range(2)],[[]for i in range(2)])
    # 1 initialisation sur résultat des MCO_pénalisé
    args={'Z':Z,'L':L,'T':T,'Xdata':Xdata,'Y':Y,'Phi_mat':Phi_mat,'k_fonc':k_fonc,'k_der':k_der,'Stochastic':True,'I_pen':I_pen,'Lambda':Lambda}                
    (Pl[0],Vf,Gl[0],vrais[0])=GD_rate(  Pi , L_YX , args,n_it,verbose=False )
    # On évalue le critère complet à la fin de l'optimisation
    a = L_YX(Pl[0],Z,L,T=T,Xdata=Xdata,Y=Y,Phi_mat=Phi_mat,k_fonc=k_fonc,k_der=k_der,
            Stochastic=False,Gradient=True,I_pen=I_pen,Lambda=Lambda)
    (Vl[0],Gl[0])=( a[0] , a[1] )
    # 2 Initialisations aléatoires
    args2={ 'func': L_YX , 'args':args , 'max_it':n_it, 'verbose':False,'tol':tol}
    (Pl[1],Vf,Gl[1],vrais[1]) =multi_init( GD_rate ,Pi ,D,p,Kbeta,args2,n_ini=n_ini)
    # On évalue le critère complet à la fin de l'optimisation
    a = L_YX(Pl[1],Z,L,T=T,Xdata=Xdata,Y=Y,Phi_mat=Phi_mat,k_fonc=k_fonc,k_der=k_der,
             Stochastic=False,Gradient=True,I_pen=I_pen,Lambda=Lambda)
    (Vl[1],Gl[1])=( a[0] , a[1] )
    # On choisi la paramètre dont le critère est le meilleur
    which_min=[ x for x in range(2) if Vl[x]==np.nanmin(Vl)][0]
    return ( Pl[which_min] , Vl[which_min] , Gl[which_min] , vrais[which_min] );


# Procédure d'inférence par SGD pour L_YlX
def LYlX_optim(func,Z,l,V,Y,I_pen,Lambda,n_it=100,n_ini=2,tol=0.5):
    N=len(Y)
    p=Z.shape[1]
    # taille totale de la base
    Kbeta=int(np.sqrt(I_pen.shape[1] ))
    D=1
    N_param=p+D*Kbeta**2
    args={'Z':Z,'V':V,'l':l,'Y':Y,'D':D,'I_pen':I_pen,'Lambda':Lambda}
    (Pl, Vl , Gl, vrais )=( [[]for i in range(2)],[[]for i in range(2)],[[]for i in range(2)],[[]for i in range(2)])
    # 1 initialisation sur résultat des MCO_pénalisé
    bf=MCO_pen_best(l,Y,I_pen,Lambda)
    Pi=np.concatenate((np.zeros(p,float),bf))
    (Pl[0],Vf,Gf,vrais[0])= GD_rate(  Pi , func , args,n_it ,verbose=False, tol=tol )
    # On évalue le critère complet à la fin de l'optimisation
    (Vl[0],Gl[0]) = L_YlX_pen(Pl[0],Z,V,l,Y,D,I_pen,Lambda)
    # 2 Initialisations aléatoires
    args2={ 'func': func , 'args':args , 'max_it':n_it, 'verbose':False,'tol':0.5}
    (Pl[1],Vf,Gf,vrais[1]) = multi_init( GD_rate ,Pi ,D,p,Kbeta,args2,n_ini=n_ini)
    # On évalue le critère complet à la fin de l'optimisation
    (Vl[1],Gl[1]) = L_YlX_pen(Pl[1],Z,V,l,Y,D,I_pen,Lambda)
    # On choisi la paramètre dont le critère est le meilleur
    which_min=[ x for x in range(2) if Vl[x]==np.nanmin(Vl)][0]
    return ( Pl[which_min] , Vl[which_min] , Gl[which_min] , vrais[which_min] );

#####################################################################################################"
#------------------------- Apprentissage Hyperparamètre par cross-validation

def parmi(n,N):
    chosen=[0.]
    Ns=range(N)
    # tirage sans remise
    for j in range(n):
        ind=np.random.randint(0,N-j)
        chosen.append(Ns[ind])
        del Ns[ind]
    del chosen[0]
    chosen=np.asarray(chosen)
    chosen=np.sort(chosen)
    bol=np.in1d( np.arange(N), chosen )
    bol = bol.astype(int)
    bol = 1-bol
    bol=bol.astype(bool)
    notchosen= np.arange(N)[bol]
    return (chosen,notchosen);

# Méthode d'inférence par les moindres carrés pénalisés
def MCO_pen_learn(Z,l,Y,I_pen,n_test=5):
    p=Z.shape[1]
    N=len(Y)
    Lambda=N*np.logspace( -3. , 3.0, num=n_test)
    rmse=[ 0. for i in range(len(Lambda))]
    Res=[ [] for i in range(len(Lambda))]
    n_learn=int(round(0.7*N))
    for i in range(len(Lambda)):
        (ind_learn,ind_test)=parmi(n_learn,N)
        (l_learn , l_test ) = ( l[ind_learn,:],l[ind_test,:] )
        (Z_learn , Z_test ) = ( Z[ind_learn,:],Z[ind_test,:] )
        ( Y_learn , Y_test ) = ( Y[ind_learn] , Y[ind_test] )
        Res[i]=  np.concatenate( ( np.zeros(p,float), MCO_pen_best(l_learn,Y_learn,I_pen,Lambda[i]) ) )
        rmse[i] = rMSE( Pred_YlX( np.concatenate((np.zeros(p,float),Res[i])) , Z_test, l_test ), Y_test )
    which_min=[ x for x in range(len(Lambda)) if rmse[x]==np.nanmin(rmse)][0]
    return  (Res[which_min],rmse[which_min],Lambda[which_min]);

# Procédure d'inférence par les moindres carrés pénalisés (double pénalité)
def MCO_2pen_learn(Z,V,l,Y,I_pen,n_test=5):
    p=Z.shape[1]
    N=len(Y)
    Lambda=N*np.logspace( -3. , 3.0, num=n_test)
    rmse=[ 0. for i in range(len(Lambda))]
    Res=[ [] for i in range(len(Lambda))]
    n_learn=int(round(0.7*N))
    for i in range(len(Lambda)):
        (ind_learn,ind_test)=parmi(n_learn,N)
        (Z_learn , Z_test ) = ( Z[ind_learn,:],Z[ind_test,:] )
        (l_learn , l_test ) = ( l[ind_learn,:],l[ind_test,:] )
        ( Y_learn , Y_test ) = ( Y[ind_learn] , Y[ind_test] )
        V_learn=[ [] for j in range(n_learn) ]
        for j in range(n_learn):
            V_learn[j]=V[ind_learn[j]]
        Res[i]=  np.concatenate( ( np.zeros(p,float), MCO_2pen_best(Z_learn,V_learn,l_learn,Y_learn,I_pen,Lambda[i]) ) )
        rmse[i] = rMSE( Pred_YlX( Res[i] , Z_test, l_test ), Y_test )
    which_min=[ x for x in range(len(Lambda)) if rmse[x]==np.nanmin(rmse)][0]
    return  (Res[which_min],rmse[which_min],Lambda[which_min]);

# Méthode d'inférence par maximum de vraisemblance en 2 étapes
def LYlX_learn(Z,V,l,Y,I_pen,n_it=100,n_ini=10,tol=0.5,n_test=5):
    N=len(Y)
    Lambda=N*np.logspace( -3. , 3.0, num=n_test)
    rmse=[ 0. for i in range(len(Lambda))]
    Res=[ [] for i in range(len(Lambda))]
    n_learn=int(round(0.7*N))
    for i in range(len(Lambda)):
        (ind_learn,ind_test)=parmi(n_learn,N)
        (Z_learn , Z_test ) = ( Z[ind_learn,:],Z[ind_test,:] )
        (l_learn , l_test ) = ( l[ind_learn,:],l[ind_test,:] )
        ( Y_learn , Y_test ) = ( Y[ind_learn] , Y[ind_test] )
        V_learn=[ [] for j in range(n_learn) ]
        for j in range(n_learn):
            V_learn[j]=V[ind_learn[j]]
        Res[i]=LYlX_optim(S_L_YlX_pen,Z_learn,l_learn,V_learn,Y_learn,I_pen,Lambda[i],n_it,n_ini,tol)
        rmse[i] = rMSE( Pred_YlX( Res[i][0] , Z_test, l_test ), Y_test )
    which_min=[ x for x in range(len(Lambda)) if rmse[x]==np.nanmin(rmse)][0]
    return  (Res[which_min],rmse[which_min],Lambda[which_min]);

# Méthode d'inférence méthode jointe
def LYX_learn(Z,l,L,T,Xdata,Y,Phi_mat,k_fonc,k_der,I_pen,n_it=100,n_ini=10,tol=0.5,n_test=5):
    p=Z.shape[1]
    D=len(L[0])
    Kbeta=int(np.sqrt(Phi_mat.shape[1]))
    N=len(Y)
    J=int(Phi_mat.shape[0]/N)-1
    Lambda=N*np.logspace( -3. , 3.0, num=n_test)
    rmse=[ 0. for i in range(len(Lambda))]
    Res=[ [] for i in range(len(Lambda))]
    n_learn=int(round(0.7*N))
    for i in range(len(Lambda)):
        (ind_learn,ind_test)=parmi(n_learn,N)
        (Z_learn , Z_test ) = ( Z[ind_learn,:],Z[ind_test,:] )
        (l_learn , l_test ) = ( l[ind_learn,:],l[ind_test,:] )
        ( Y_learn , Y_test ) = ( Y[ind_learn] , Y[ind_test] )
        T_learn=T[ind_learn]
        L_learn=[ [] for j in range(n_learn) ]
        Xdata_learn=[ [] for j in range(n_learn) ]
        Phi_mat_learn = np.zeros((n_learn*(J+1),Kbeta**2),float )
        for j in range(n_learn):
            L_learn[j]=L[ind_learn[j]]
            Xdata_learn[j]=Xdata[ind_learn[j]]
            Phi_mat_learn[(j*(J+1)):(j*(J+1)+J+1),:]=Phi_mat[(ind_learn[j]*(J+1)):((ind_learn[j]+1)*(J+1))]
        Res[i]=list( LYX_optim(Z_learn,l_learn,L_learn,T_learn,
                         Xdata_learn,Y_learn,Phi_mat_learn,k_fonc,k_der,I_pen,Lambda[i],n_it,n_ini,tol) )
        Res[i][0]=Res[i][0][0:(p+D*Kbeta**2)]
        rmse[i] = rMSE( Pred_YlX( Res[i][0] , Z_test, l_test ), Y_test )
    which_min=[ x for x in range(len(Lambda)) if rmse[x]==np.nanmin(rmse)][0]
    return  (Res[which_min],rmse[which_min],Lambda[which_min]);



#####################################################################################################
#------------------------ Prédiction -------------------------------
#####################################################################################################
def Pred_YlX(P,Z,l):
    p=Z.shape[1]
    Alpha=P[0:p]
    lb=l.shape[1]
    b=P[p:(lb+p)]
    Pred=Z.dot(Alpha)+l.dot(b)
    return Pred;

def rMSE(Pred,Y):
    N=len(Y)
    return np.sqrt((Pred-Y).dot(Pred-Y)/N);

#####################################################################################################
#---------------------------------- Calcul de la pénalité ------------------------------------------
#####################################################################################################
# On définit une fonction qui calcule les matrices d'intégrales associées aux différentes dérivées secondes 
# de notre pénalité J22. Celle-ci utilise la dérivation symbolique automatique de sympy et l'intégration
# multiple scipy.integrate.dlbquad

def J22(syPhi,Tmax):
    # taille de la base fonctionnelle
    Kbeta=len(syPhi)
    # symbole associé au petit t (c'est à dire l'instant dans la période de suivi)
    t=sy.Symbol('t')
    # smbole associé au grand T (c'est à dire la durée de suivie)
    s=sy.Symbol('s')
    deb=time.clock()
    Phi=sy.ones(Kbeta,1)
    for i in range(Kbeta):
        Phi[i]=syPhi[i]
    Phi_dsds=Phi.diff(s,s)
    Phi_dsdt=Phi.diff(s,t)
    Phi_dtdt=Phi.diff(t,t)
    
    Is=np.zeros((Kbeta,Kbeta),float)
    Ic=np.zeros((Kbeta,Kbeta),float)
    It=np.zeros((Kbeta,Kbeta),float)
    def gfun(x): return 0;
    def hfun(x): return x;
    for i in range(Kbeta):
        for j in range(Kbeta):
            func=sy.lambdify((t,s),Phi_dsds[j]*Phi_dsds[i],'numpy')
            Is[i,j]=integrate.dblquad(func, 0., Tmax, gfun, hfun, epsabs=5e-03)[0]
            func=sy.lambdify((t,s),Phi_dsdt[j]*Phi_dsdt[i],'numpy')
            Ic[i,j]=integrate.dblquad(func, 0., Tmax, gfun, hfun, epsabs=5e-03)[0]
            func=sy.lambdify((t,s),Phi_dsdt[j]*Phi_dtdt[i],'numpy')
            It[i,j]=integrate.dblquad(func, 0., Tmax, gfun, hfun, epsabs=5e-03)[0]
    print(str(time.clock()-deb)+" secondes de calcul")
    I_pen=Is+It+2*Ic
    return (Is,Ic,It,I_pen);


def J22_fast(syPhi,Tmax,J):
    # taille de la base fonctionnelle
    Kbeta=len(syPhi)
    # symbole associé au petit t (c'est à dire l'instant dans la période de suivi)
    t=sy.Symbol('t')
    # smbole associé au grand T (c'est à dire la durée de suivie)
    s=sy.Symbol('s')
    
    deb=time.clock()
    Phi=sy.ones(Kbeta,1)
    for i in range(Kbeta):
        Phi[i]=syPhi[i]
    # dérivation de la base fonctionnelle
    ( Phi_dsds , Phi_dsdt , Phi_dtdt )=( Phi.diff(s,s) , Phi.diff(s,t) , Phi.diff(t,t) )
    
    (Phi_mat_dsds,Phi_mat_dsdt,Phi_mat_dtdt)=(np.zeros(((J+1)**2,Kbeta),float),np.zeros(((J+1)**2,Kbeta),float),np.zeros(((J+1)**2,Kbeta),float))
    # Grille d'intégration carré (on retire le triangle supérieur plus loin)
    t_arg=sc.linspace(0,Tmax,J+1)
    s_arg=t_arg
    args=cg.expandnp([t_arg,s_arg]).T
    for i in range(Kbeta):
        func=sy.lambdify((t,s),Phi_dsds[i],'numpy')    
        Phi_mat_dsds[:,i]=np.apply_along_axis(func,0,*args)
        func=sy.lambdify((t,s),Phi_dsdt[i],'numpy')
        Phi_mat_dsdt[:,i]=np.apply_along_axis(func,0,*args)
        func=sy.lambdify((t,s),Phi_dtdt[i],'numpy')
        Phi_mat_dtdt[:,i]=np.apply_along_axis(func,0,*args)
        
    (Is,Ic,It)=( np.zeros((Kbeta,Kbeta),float) , np.zeros((Kbeta,Kbeta),float) , np.zeros((Kbeta,Kbeta),float) )
    Un=np.ones(J+1,float)
    #matrice triangulaire inférieure d'intégration
    a=cg.expandnp([np.arange(J+1),np.arange(J+1)])
    triang= np.asarray( a[:,0]<=a[:,1] , float ).reshape((J+1,J+1))
    # Calcul des intégrales
    for i in range(Kbeta):
        for j in range(Kbeta):
            if i<=j:
                Is[i,j]=Un.dot(Phi_mat_dsds[:,j].reshape((J+1,J+1))*Phi_mat_dsds[:,i].reshape((J+1,J+1))*triang).dot(Un)*(Tmax**2)/(J*(J+1))
                Ic[i,j]=Un.dot(Phi_mat_dsdt[:,j].reshape((J+1,J+1))*Phi_mat_dsdt[:,i].reshape((J+1,J+1))*triang).dot(Un)*(Tmax**2)/(J*(J+1))
                It[i,j]=Un.dot(Phi_mat_dtdt[:,j].reshape((J+1,J+1))*Phi_mat_dtdt[:,i].reshape((J+1,J+1))*triang).dot(Un)*(Tmax**2)/(J*(J+1))
            else:
                ( Is[i,j] , Ic[i,j] , It[i,j] )=( Is[j,i] , Ic[j,i] , It[j,i] )
    I_pen=Is+It+2*Ic
    I_pen=I_pen + np.eye(Kbeta)*0.001*np.trace(I_pen)/Kbeta
    return (Is,Ic,It,I_pen);


#####################################################################################################
#------------------------ Critère ISE -------------------------------
#####################################################################################################

def construct_beta(Kbeta):
    # symbole associé au petit t (c'est à dire l'instant dans la période de suivi)
    t=sy.Symbol('t')
    # smbole associé au grand T (c'est à dire la durée de suivie)
    s=sy.Symbol('s')
    syPhi=sy.ones(Kbeta,1)
    syb=sy.ones(1,Kbeta)
    b=[ [] for k in range(Kbeta)]
    v=[np.arange(np.sqrt(Kbeta)),np.arange(np.sqrt(Kbeta))]
    expo=cg.expandnp(v)
    for x in range(len(expo[:,0])):
        syPhi[x]=(t**expo[x,0])*(s**expo[x,1])
        syb[x]=sy.Symbol('b'+str(x))
        b[x]= sy.Symbol('b'+str(x))
    syBeta=syb*syPhi
    syBeta=syBeta[0,0]
    arg= [t,s] + b
    Beta_fonc_est=sy.lambdify(tuple(arg),syBeta,'numpy')
    return Beta_fonc_est;

# Calcul de l'Integrated Square Error    
def ISE(numbeta,Beta_fonc_est,b_learned,T,nb=40,plot_beta=False,resolution=100):
    ise=0.
    Tmax=np.max(T)
    Tmin=np.min(T)
    Tgrid=np.linspace(Tmin,Tmax,nb,retstep=True)
    step=Tgrid[1]
    Tgrid=Tgrid[0]
    grille=np.vstack((Tgrid,np.arange(nb)))
    count=0
    for T_,i in grille.T:
        for t_ in np.arange( 0.01,T_,step,float ):
            count+=1
            true_b=cg.beta_fonc(t_,T_,numbeta)
            est_b=Beta_fonc_est( *( [t_,T_] + list(b_learned[:,0]) ) )
            ise= ise + (true_b - est_b )**2
    ise=ise/count
    print("ISE : "+str(ise))
    if plot_beta:
        x=np.linspace( 0.1, np.max(T), num=resolution)
        y=np.linspace( 0.1, np.max(T), num=resolution)
        arr=[x,y]
        Grid=cg.expandnp(arr)
        x=Grid[:,0]
        y=Grid[:,1]
        arrV=np.zeros(resolution**2,float)
        n=0
        for i in range(resolution):
            for j in range(resolution):
                if j<=i:
                    arrV[n] = Beta_fonc_est[0](x[n],y[n])
                n+=1
        Xfig=x.reshape((resolution,resolution))
        Yfig=y.reshape((resolution,resolution))
        Zfig=arrV.reshape((resolution,resolution))
        fig=plt.figure().add_subplot(111)
        plt.imshow(Zfig, vmin=Zfig.min(), vmax=Zfig.max(), origin='lower',
                   extent=[ Yfig.min(), Yfig.max(),Xfig.min(),Xfig.max()])
        plt.colorbar()
    return ise;
