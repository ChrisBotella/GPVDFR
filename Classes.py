
# coding: utf-8

# In[ ]:

import numpy as np
import scipy as sp
from scipy import linalg
import sympy as sy
import pylab as pl
import math as ma
import time 

# FONCTIONS ----------------------------------------

def poly(vec,a,b):
    i=vec[0]
    j=vec[1]
    return (a**i)*(b**j);

def prod(vec):
    "Produit des valeurs d'un vecteur"
    P=1
    l=len(vec)
    if l==0:
        return 1;
    else:
        for i in range(len(vec)):
            
            P=P*vec[i]
        return P;

# Fonction générale renvoyant la matrice croisée d'évaluation vectorielle d'un objet
def val_mat(obj,vecarg):
    val=obj.val(vecarg)
    Vec=np.reshape(val,(len(val),1))
    Mat=Vec.dot(Vec.T)
    return Mat;

def expandnp(arr):
    d=len(arr)
    J=np.zeros(d,int)
    for i in range(d):
        J[i]=len(arr[i])
    P=prod(J)    
    Tab=np.zeros((P,d))
    for j in range(d):
        v=np.array([arr[j]])
        rep=prod(J[0:j]) 
        V=np.repeat(v,rep,axis=1)
        rep=P/prod(J[0:(j+1)])
        V2=np.repeat(V,rep,axis=0)
        M=np.reshape(V2,(1,P))
        Tab[:,j]=M
    return Tab;

#---------------------------
# renvoie le tableau de valeurs de obj (fonction de 2 variables) pour tous les couples de valeurs dans t
def cov(obj,t):
    # Calculer la matrice de variance-covariance d'une série de temps à partir d'une fonction de covariance
    vec=expandnp(np.array([t,t])).T
    res=np.apply_along_axis(obj.val,axis=0,arr=vec)
    res=np.reshape(res,(len(t),len(t)))
    return res;


# CLASSES --------------------------------------------------------------------------------------------------

class X_fonc:
    " Fonction temporelle paramétrique de référence proposée dans Gellar et al., 2014"
    def __init__(self,t,u,v1,v2):
        self.t=np.asarray(t)
        if u.size==1 and v1.size==10 and v1.size==10:
            self.u=u
            self.v1=v1
            self.v2=v2
        else:
            print "Erreur dans les dimensions de paramètres"            
    def val(self):
        t=self.t
        u=self.u
        v1=self.v1
        v2=self.v2
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

#------------------
    
class beta:
    " Fonctions paramètre de référence proposées dans Gellar et al. , 2014 "
    import math as ma
    def __init__(self,num=1,J=1):
        self.num=num
    def val(self,vecarg):
        # vecarg doit être un np.array simple (1D) dont chaque élément est une variable de la fonction
        t=vecarg[0]
        T=vecarg[1]
        num=self.num
        J=self.J
        if num==1:
            return(10*t/T-5);
        if num==2:
            return((1-2*T/J)*(5-40*(t/T-0.5)^2));
        if num==3:
            return(5-10*(T-t)/J);
        if num==4:
            return(sin(2*ma.pi*T/J)*(5-10*(T-t)/J));
        

#------------------
        
class gauss:
    " Fonction gaussienne "
    def __init__(self,mu,sig):
        self.dim=1
        self.mu=mu
        self.sig=sig
    def val(self,vecarg):
        return np.exp( -(vecarg-self.mu)**2/(2*self.sig**2 )) / (ma.sqrt(2*ma.pi)*self.sig);
#-----------------------
# Noyau exponentiel quadratique
class RBF:
    def __init__(self,el,sig):
        self.dim=2
        self.sig=sig
        self.el=el
    def val(self,vecarg):
        return self.sig**2*np.exp(-(vecarg[1]-vecarg[0])**2/(2*self.el**2));
# Noyau exponentiel quadratique périodique
class Periodic:
    def __init__(self,el,per,sig):
        self.dim=2
        self.sig=sig
        self.el=el
        self.per=per
    def val(self,vecarg):
        r=vecarg[1]-vecarg[0]
        return self.sig**2*np.exp(-(r**2/(2*self.el**2))- np.sin(2*ma.pi*r/self.per)**2/2 );

#-----------------------    
# Renvoie un objet fonction somme de deux objets fonctions 
class Sum:
    def __init__(self,obj1,obj2):
        self.obj1=obj1
        self.obj2=obj2
    def val(self,vecarg):
        return self.obj1.val(vecarg)+self.obj2.val(vecarg);
    
# Renvoie un objet fonction produit de deux objets fonctions 
class Prod:
    def __init__(self,obj1,obj2):
        self.obj1=obj1
        self.obj2=obj2
    def val(self,vecarg):
        return self.obj1.val(vecarg)*self.obj2.val(vecarg);
#------------------
class Base:
    "Base polynomiale canonique des fonctions de R^2 dans R," 
    "premier element du vecteur entrée est la dimension de la base polynomiale sur le premier axe et de même pour le deuxième élément"
    import numpy as np
    def __init__(self,darg=np.zeros(2)):
        self.dim=2
        self.dt =darg[0]
        self.dT =darg[1]
    
    def val(self,vecarg):
        t=vecarg[0]
        T=vecarg[1]
        v=np.array([np.arange(self.dt),np.arange(self.dT)])
        grid=expandnp(v)
        Phi=np.apply_along_axis(poly , axis=1, arr=grid , a=t, b=T )
        return Phi;
    
#------------------

class Integ:
    "Integrale d'une classe sur un intervalle multidimensionnel parréllépipédique. Renvoie un np.array dont la longueur correspond la taille de la valeur de sortie de la classe."
    def __init__(self,obj,borne_inf,borne_sup,J):
        dim_int=np.array([])
        for i in range(len(borne_inf)):
            if borne_sup[i]!=borne_inf[i]:
                dim_int=np.append(dim_int,i)
        self.dim=obj.dim
        self.obj=obj
        self.borne_inf=borne_inf.astype(float)
        self.borne_sup=borne_sup.astype(float)
        self.dim_int=dim_int
        self.J=J
    
    def val(self,vec=None,dim=None):
        # vec est le np.array contenant les variables de la fonction à évaluer, 
        # Pour toute variable sur laquelle la fonction n'est pas intégrée il s'agit 
        # de la valeur constante qu'elle doit prendre
        # Pour la variable intégrée il s'agit de la borne inférieure d'intégration 
        # /!\ Une dimension à la fois pour l'intégration /!\
        J=self.J
        # Si c'est le premier appel de la fonction, on calcule la masse volumique
        if vec is None:
            dim=self.dim_int[0]
            vec=self.borne_inf
            Prod=1./(J**len(self.dim_int))
            for i in self.dim_int:
                Prod=Prod*(self.borne_sup[i]-self.borne_inf[i])
            
        Un=np.ones(J+1,float)
        Un[0]=0.5
        Un[J]=0.5
        
        x=np.repeat(np.array([vec]),[J+1],axis=0)
        x[:,dim]= vec[dim]+np.arange(J+1)*1.*(self.borne_sup[dim]-vec[dim])/J 
        # Si on est au niveau d'une feuille, on évalue et somme le volume de chaque feuille
        if dim==self.dim_int[len(self.dim_int)-1]:
            F=np.apply_along_axis( self.obj.val , axis=1, arr=x )
        else:
            next_dim= self.dim_int[np.argwhere(dim==self.dim_int)[0][0]+1]
            F=np.apply_along_axis( self.val , axis=1, arr=x, dim=next_dim )

        # si c'est le premier appel, on multiplie le volume total par la masse volumique "Prod", sinon on renvoie simplement le volume de la branche
        if dim==self.dim_int[0]:
            vol=Un.dot(F)
            return Prod*vol;
        else:
            return Un.dot(F);
        
#------------------

class Integ_cub:
    "Integrale d'une classe sur un intervalle uni-dimensionnel. Renvoie un np.array dont la longueur correspond la taille de la valeur de sortie de la classe."
    def __init__(self,obj,borne_inf,borne_sup,J=100):
        dim_int=np.array([])
        for i in range(len(borne_inf)):
            if borne_sup[i]!=borne_inf[i]:
                dim_int=np.append(dim_int,i)
        self.dim=obj.dim
        self.obj=obj
        self.borne_inf=borne_inf.astype(float)
        self.borne_sup=borne_sup.astype(float)
        self.dim_int=dim_int
        self.J=J
    
    def val(self):
        J=self.J
        d=len(self.dim_int)
        mini_Tab=np.ones((J+1,d),float)
        # construction du vecteur de coefficients de la grille de l'hyper-rectangle
        mini_Tab[0,:]=0.5
        mini_Tab[J,:]=0.5
        coef=expandnp(mini_Tab.T)
        l=(J+1)**d
        Coef=np.ones((1,l),float)
        for k in range(d):
            Coef=Coef*coef[:,k]
        # construction vecteur de la grille pour la fonction intégrée
        i=0
        for dim in self.dim_int:
            mini_Tab[:,i]= self.borne_inf[dim] + np.arange(J+1)*1.*(self.borne_sup[dim]-self.borne_inf[dim])/J
            i=i+1
        Tab=expandnp(mini_Tab.T)
        del mini_Tab
        i=0
        for coo in range(len(self.borne_inf)):
            if sum(coo==self.dim_int)==0:
                if coo==0:
                    B_Tab=np.reshape(np.repeat(self.borne_inf[coo],l),(l,1))
                else:
                    B_Tab=np.hstack((B_Tab,np.reshape(np.repeat(self.borne_inf[coo],l),(l,1))))
            else:
                if coo==0:
                    B_Tab=np.reshape(Tab[:,i],(l,1))
                else:
                    B_Tab=np.hstack((B_Tab,np.reshape(Tab[:,i],(l,1))))
                i=i+1
        del Tab
        F=np.apply_along_axis( self.obj.val , axis=1, arr=B_Tab )
        del B_Tab
        I=Coef.dot(F)
        Prod=1./(J**len(self.dim_int))
        for i in self.dim_int:
            Prod=Prod*(self.borne_sup[i]-self.borne_inf[i])
        return Prod*I;
#---------------------

class Integ_MC:
    "Intégrale d'une classe avec méthode de Monte Carlo basique"
    def __init__(self,obj,borne_inf,borne_sup,n=100):
        dim_int=np.array([])
        for i in range(len(borne_inf)):
            if borne_sup[i]!=borne_inf[i]:
                dim_int=np.append(dim_int,i)
        self.dim=obj.dim
        self.obj=obj
        self.borne_inf=borne_inf.astype(float)
        self.borne_sup=borne_sup.astype(float)
        self.dim_int=dim_int
        self.n=n
    def val(self):
        n=self.n
        x=np.zeros((n,len(self.dim_int)))
        for i in range(len(self.dim_int)):
            x[:,i]=np.random.uniform(self.borne_inf[i],self.borne_sup[i],n)
        F=np.apply_along_axis(self.obj.val,axis=1,arr=x)
        Vol=1.
        for i in self.dim_int:
            Vol=Vol*(self.borne_sup[i]-self.borne_inf[i])
        return Vol*np.ones((1,n),float).dot(F)/n;
        

