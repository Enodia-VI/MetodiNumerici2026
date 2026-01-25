#Zeri di funzione

import math

import numpy as np
import scipy

import SolveTriangular
from SolveTriangular import Lsolve


def sign(x):
  """
  Funzione segno che restituisce 1 se x è positivo, 0 se x è zero e -1 se x è negativo.
  """
  return math.copysign(1, x)

def metodo_bisezione(fname, a, b, tolx):
 
 fa=fname(a)
 fb=fname(b)
 if sign(fa)*sign(fb)>=0 : #to do
     print("Non è possibile applicare il metodo di bisezione \n")
     return None, None,None

 it = 0
 v_xk = []

# se si vuole calcolare il numero di iterazioni massime
 maxit = math.ceil(math.log2((b-a)/tolx)) - 1
 
 
 while abs(b-a) > tolx: #to do
     # siccome con (a+b)/2 si rischia di uscire dall'intervallo
     # con numeri finiti, si preferisce usare a+(b-a)/2
    xk = a+(b-a)/2 #to do
    v_xk.append(xk)
    it += 1
    fxk=fname(xk)
    if fxk==0:
      return xk, it, v_xk

    if sign(fa)*sign(fxk) > 0: # to do
      a = xk #to do
      fa= fxk #to do
    elif sign(fxk)*sign(fb)>0:# to do
      b = xk
      fb=fxk

 
 return xk, it, v_xk

def falsa_posizione(fname,a,b,tolx,tolf,maxit):
    fa=fname(a)
    fb=fname(b)
    if sign(fa*fb)>0:#to do:
       print("Metodo di bisezione non applicabile")
       return None,None,None

    it=0
    v_xk=[]
    fxk=1+tolf
    errore=1+tolx
    xprec=a
    while it<maxit and abs(fxk)> tolf and errore>tolx :     #TODO
        xk= a-fa*(b-a)/(fb-fa)                              #TODO
        v_xk.append(xk)
        it+=1
        fxk=fname(xk)                                       #TODO
        if fxk==0:
            return xk,it,v_xk

        # radice in [a,xk]
        if sign(fa*fxk)<0 :                                 #TODO
           b= xk#to do
           fb= fxk                                          #TODO

        # radice in [xk,b]
        elif sign(fxk*fb)<0 :                               #TODO
           a=xk                                             #TODO
           fa=fxk                                           #TODO
        if xk!=0:
            errore=abs(xk-xprec)/abs(xk)                    #TODO
        else:
            errore=abs(xk-xprec)                            #TODO
        xprec=xk
    return xk,it,v_xk

def corde(fname,coeff_ang,x0,tolx,tolf,nmax):
    
     # coeff_ang è il coefficiente angolare della retta che rimane fisso per tutte le iterazioni
        xk=[]
        
        it=0
        errorex=1+tolx
        erroref=1+tolf
        while it<max and erroref>=tolf and errorex>= tolx :  #TODO
           
           fx0=fname(x0)                                    #TODO
           d=fx0 /coeff_ang                                 #TODO
          
           x1= x0 - d                                       #TODO
           fx1=fname(x1)                                    #TODO
           if x1!=0:
                errorex=abs(d)/abs(x1)                      #TODO
           else:
                errorex=abs(d)                              #TODO
           
           erroref=np.abs(fx1)                              #TODO
           
           x0=x1
           it=it+1
           xk.append(x1)
          
        if it==nmax:
            print('Corde : raggiunto massimo numero di iterazioni \n')
            
        
        return x1,it,xk
    
def newton(fname,fpname,x0,tolx,tolf,nmax):
  
        xk=[]
       
        it=0
        errorex=1+tolx
        erroref=1+tolf
        while it<nmax and erroref>=tolf and errorex>=tolx : #TODO
           
           fx0=fname(x0)
           if abs(fpname(x0))<= np.spacing(1) :             #TODO
                print(" derivata prima nulla in x0")
                return None, None,None
           d=fx0/fpname(x0)                                 #TODO

           x1= x0-d                                         #TODO
           fx1=fname(x1)
           erroref=np.abs(fx1)
           if x1!=0:
                errorex=abs(d)/ abs(x1)                     #TODO
           else:
                errorex=abs(d)                              #TODO

           it=it+1
           x0=x1
           xk.append(x1)
          
        if it==nmax:
            print('Newton: raggiunto massimo numero di iterazioni \n')
            
        
        return x1,it,xk

def newton_modificato(fname,fpname,m,x0,tolx,tolf,nmax):
  
        #m è la molteplicità dello zero
    
        xk=[]
       
        it=0
        errorex=1+tolx
        erroref=1+tolf
        while it<nmax and erroref>=tolf and errorex>= tolx: #TODO
           
           fx0=fname(x0)
           if abs(fpname(x0)) <=np.spacing(1):              #TODO
                print(" derivata prima nulla in x0")
                return None, None,None
           d=fx0/fpname(x0)                                 #TODO

           x1=x0 - m*d                                      #TODO
           fx1=fname(x1)
           erroref=np.abs(fx1)
           if x1!=0:
                errorex=abs(d)/abs(x1)                      #TODO
           else:
                errorex=abs(d)                              #TODO

           it=it+1
           x0=x1
           xk.append(x1)
          
        if it==nmax:
            print('Newton modificato: raggiunto massimo numero di iterazioni \n')
            
        
        return x1,it,xk
    
def secanti(fname,xm1,x0,tolx,tolf,nmax):
        xk=[]

        # !! N.B.: xm1 e' x-1 quindi e' il secondo membro
        
        it=0
        errorex=1+tolx
        erroref=1+tolf
        while it<nmax and erroref>=tolf and errorex>=tolx:  #TODO
            
            fxm1=fname(xm1)                                 #TODO
            fx0=fname(x0)                                   #TODO
            d=fx0*(x0-xm1)/(fx0-fxm1)#to do

            x1=x0 -d                                        #TODO
          
            
            fx1=fname(x1)
            xk.append(x1);
            if x1!=0:
                errorex=abs(d)/abs(x1)                      #TODO
            else:
                errorex=abs(d)                              #TODO
                
            erroref=np.abs(fx1)                             #TODO
            # !! N.B: i passaggi fondamentali del metodo
            xm1=x0                                          #TODO
            x0=x1                                           #TODO
            
            it=it+1;
           
       
        if it==nmax:
           print('Secanti: raggiunto massimo numero di iterazioni \n')
        
        return x1,it,xk
    
def stima_ordine(xk,iterazioni):
     #Vedi dispensa allegata per la spiegazione

      k=iterazioni-4
      p=np.log(abs(xk[k+2]-xk[k+3])/abs(xk[k+1]-xk[k+2]))/np.log(abs(xk[k+1]-xk[k+2])/abs(xk[k]-xk[k+1]));
     
      ordine=p
      return ordine


#Soluzione di sistemi di equazioni non lineari
def newton_raphson(initial_guess, F_numerical, J_Numerical, tolX, tolF, max_iterations):
    

    X= np.array(initial_guess, dtype=float)
    
   

    it=0
    
    erroreF=1+tolF
    erroreX=1+tolX
    
    errore=[]
    
    while it<max_iterations and erroreF>=tolF and erroreX>=tolX:    #TODO
        
        jx = J_Numerical(X[0],X[1])                                 #TODO
        
        if np.linalg.det(jx) == 0:                                  #TODO
            print("La matrice dello Jacobiano calcolata nell'iterato precedente non è a rango massimo")
            return None, None,None
        
        fx = F_numerical(X[0],X[1])                                 #TODO
        fx = fx.squeeze() 
        
        s = np.linalg.solve(jx, -fx)                                #TODO
        
        Xnew=X + s                                                  #TODO
        
        normaXnew=np.linalg.norm(Xnew,1)
        if normaXnew !=0:
            erroreX=np.linalg.norm(s,1)/normaXnew               #TODO
        else:
            erroreX=np.linalg.norm(s,1)                         #TODO
        
        errore.append(erroreX)
        fxnew=F_numerical(Xnew[0],Xnew[1])                          #TODO
        erroreF= np.linalg.norm(fxnew.squeeze(),1)
        X=Xnew
        it=it+1
    
    return X,it,errore

def newton_raphson_corde(initial_guess, F_numerical, J_Numerical, tolX, tolF, max_iterations):
    

    X= np.array(initial_guess, dtype=float)
    
   

    it=0
    
    erroreF=1+tolF
    erroreX=1+tolX
    
    errore=[]
    
    while it<max_iterations and erroreF>tolF and erroreX>tolX:      #TODO
        
        if it==0:                                                   #TODO
            jx = J_Numerical(X[0],X[1])                             #TODO
        
            if np.linalg.det(jx)==0:                                #TODO
                print("La matrice dello Jacobiano calcolata nell'iterato precedente non è a rango massimo")
                return None, None,None
        
        fx = F_numerical(X[0],X[1])                                 #TODO
        fx = fx.squeeze() 
        
        s = np.linalg.solve(jx,-fx)                                 #TODO
        
        Xnew=X+s#to do
        
        normaXnew=np.linalg.norm(Xnew,1)
        if normaXnew !=0:
            erroreX=np.linalg.norm(s,1)/normaXnew               #TODO
        else:
            erroreX=np.linalg.norm(s,1)                         #TODO
        
        errore.append(erroreX)
        fxnew=F_numerical(Xnew[0],Xnew[1])                          #TODO
        erroreF= np.linalg.norm(fxnew.squeeze(),1)
        X=Xnew
        it=it+1
    
    return X,it,errore


def newton_raphson_sham(initial_guess, update, F_numerical, J_Numerical, tolX, tolF, max_iterations):
    
    

    X= np.array(initial_guess, dtype=float)
    
   

    it=0
    
    erroreF=1+tolF
    erroreX=1+tolX
    
    errore=[]
    
    while it<max_iterations and erroreF>tolF and erroreX>tolX:      #TODO
        
        if it%update == 0:                                          #TODO
            jx = J_Numerical(X[0],X[1])                             #TODO
        
            if np.linalg.det(jx) == 0:                              #TODO
                print("La matrice dello Jacobiano calcolata nell'iterato precedente non è a rango massimo")
                return None, None,None
        
        fx = F_numerical(X[0],X[1])                                 #TODO
        fx = fx.squeeze() 
        
        s = np.linalg.solve(jx,-fx)                                 #TODO
        
        Xnew=X+s#to do
        
        normaXnew=np.linalg.norm(Xnew,1)
        if normaXnew !=0:
            erroreX=np.linalg.norm(s,1)/normaXnew                   #TODO
        else:
            erroreX=np.linalg.norm(s,1)                         #TODO
        
        errore.append(erroreX)
        fxnew=F_numerical(Xnew[0],Xnew[1])                          #TODO
        erroreF= np.linalg.norm(fxnew.squeeze(),1)
        X=Xnew
        it=it+1
    
    return X,it,errore




#Minimo di una funzion enon lineare

def newton_raphson_minimo(initial_guess, grad_func, Hessian_func, tolX, tolF, max_iterations):
    

    X= np.array(initial_guess, dtype=float)
    
    it=0
    
    erroreF=1+tolX
    erroreX=1+tolF
    
    errore=[]
    
    while it<max_iterations and erroreF>tolF and erroreX<tolX:      #TODO

        Hx = Hessian_func(X[0],X[1])                                #TODO
        
        if np.linalg.det(Hx) == 0 :                                 #TODO
            print("La matrice Hessiana calcolata nell'iterato precedente non è a rango massimo")
            return None, None,None
        
        gfx =  grad_func(X[0],X[1])                                 #TODO
        gfx = gfx.squeeze() 
        
        s = np.linalg.solve(Hx,-gfx)                                #TODO
        
        Xnew=X+s#to do
        
        normaXnew=np.linalg.norm(Xnew,1)
        if normaXnew!=0:
            erroreX=np.linalg.norm(s,1)/normaXnew                   #TODO
        else:
            erroreX=np.linalg.norm(s,1)                             #TODO
            
        errore.append(erroreX)
        gfxnew=grad_func(X[0],X[1])                                 #TODO
        erroreF= np.linalg.norm(gfxnew.squeeze(),1)
        X=Xnew
        it=it+1
    
    return X,it,errore

#Metodi Iterativi basati sullo splitting della matrice: jacobi, gauss-Seidel - Gauss_seidel SOR
def jacobi(A,b,x0,toll,it_max):
    errore=1000
    d=np.diag(A)
    n=A.shape[0]
    invM=np.diag(1/d)
    E=np.tril(A,-1)                                             #TODO
    F=np.triu(A,1)                                          #TODO
    N=-(E+F)                                                    #TODO
    T=np.dot(invM,N)                                            #TODO
    autovalori=np.linalg.eigvals(T)
    raggiospettrale=np.max(np.abs(autovalori))                  #TODO
    print("raggio spettrale jacobi", raggiospettrale)

    # chicca teorica, bello da imparare
    if raggiospettrale>=1 :
        print("raggio spettrale>1: metodo esplode")

    it=0
    
    er_vet=[]
    while it<=it_max and errore>=toll:
        x=(b+N@x0)/d.reshape(n,1)                               #TODO
        errore=np.linalg.norm(x-x0)/np.linalg.norm(x)           #TODO
        er_vet.append(errore)
        x0=x.copy()
        it=it+1
    return x,it,er_vet


def gauss_seidel(A,b,x0,toll,it_max):
    errore=1000
    d=np.diag(A)                                                    #TODO
    D=np.diag(d)                                                    #TODO
    E=np.tril(A,-1)                                                 #TODO
    F=np.tril(A,1)                                              #TODO
    M=D+E                                                           #TODO
    N=-F                                                            #TODO
    invM=np.linalg.inv(M) # aggiunta perche assente
    T=invM@N                                                        #TODO
    autovalori=np.linalg.eigvals(T)
    raggiospettrale=np.max(abs(autovalori))                         #TODO
    print("raggio spettrale Gauss-Seidel ",raggiospettrale)
    it=0
    er_vet=[]
    while it<=it_max and errore>=toll:                              #TODO
        temp = b-F@x0
        x,flag=Lsolve(M,temp)                                       #TODO
        errore=np.linalg.norm(x-x0)/np.linalg.norm(x)               #TODO
        er_vet.append(errore)
        x0=x.copy()
        it=it+1
    return x,it,er_vet

def gauss_seidel_sor(A,b,x0,toll,it_max,omega):
    errore=1000
    d=np.diag(A)                                                    #TODO
    D=np.diag(d)                                                    #TODO
    E=np.tril(A,-1)                                                 #TODO
    F=np.tril(A,1)                                              #TODO
    Momega=D+omega*E
    Nomega=(1-omega)*D-omega*F
    T=np.dot(np.linalg.inv(Momega), Nomega)                         #TODO
    autovalori=np.linalg.eigvals(T)
    raggiospettrale=np.max(np.abs(autovalori))                      #TODO
    print("raggio spettrale Gauss-Seidel SOR ", raggiospettrale)
    
    M=D+E                                                           #TODO
    N=-F                                                            #TODO
    it=0
    xold=x0.copy()
    xnew=x0.copy()
    er_vet=[]
    while it<=it_max and errore>=toll:
        temp=b-np.dot(F,xold)
        xtilde,flag= Lsolve(M,temp)                                 #TODO
        xnew=(1-omega)*xold+omega*xtilde                            #TODO
        errore=np.linalg.norm(xnew-xold)/np.linalg.norm(xnew)       #TODO
        er_vet.append(errore)
        xold=xnew.copy()
        it=it+1
    return xnew,it,er_vet


#Metodi di Discesa ------------------------------------------------------

def steepestdescent(A,b,x0,itmax,tol):
 
    n,m=A.shape
    if n!=m:
        print("Matrice non quadrata")
        return [],[]
    
    
   # inizializzare le variabili necessarie
    x = x0

     
    r = A@x-b                                                       #TODO
    p = -r                                                          #TODO
    it = 0
    nb=np.linalg.norm(b)
    errore=np.linalg.norm(r)/nb
    vec_sol=[]
    vec_sol.append(x.copy())
    vet_r=[]
    vet_r.append(errore)
     
# utilizzare il metodo del gradiente per trovare la soluzione
    while it<itmax and errore>= tol:                                #TODO
        it=it+1
        Ap=A@p                                                      #TODO
       
        alpha =-(r.T@p)/(p.T@Ap)                                    #TODO
                
        x = x + alpha*p                                             #TODO
        
         
        vec_sol.append(x.copy())
        r= r + alpha*Ap                                             #TODO
        errore=np.linalg.norm(r)/nb
        vet_r.append(errore)
        p = -r                                                      #TODO
        
    iterates_array = np.vstack([arr.T for arr in vec_sol])
    return x,vet_r,iterates_array,it

def conjugate_gradient(A,b,x0,itmax,tol):
    n,m=A.shape
    if n!=m:
        print("Matrice non quadrata")
        return [],[]
    
    
   # inizializzare le variabili necessarie
    x = x0
    
    r = A@x-b                                                       #TODO
    p = -r                                                          #TODO
    it = 0
    nb=np.linalg.norm(b)
    errore=np.linalg.norm(r)/nb
    vec_sol=[]
    vec_sol.append(x0.copy())
    vet_r=[]
    vet_r.append(errore)
# utilizzare il metodo del gradiente coniugato per calcolare la soluzione
    while it<itmax and errore<tol: #to do
        it=it+1
        Ap=A@p                                                      #TODO A.dot(p)
        alpha = -(r.T@p)/(r.T@Ap) #to do
        x = x + alpha*p                                             #TODO
        vec_sol.append(x.copy())
        rtr_old=r.T@r                                               #TODO
        r= r+alpha*Ap                                               #TODO
        gamma=r.T@r/rtr_old                                         #TODO
        errore=np.linalg.norm(r)/nb
        vet_r.append(errore)
        p = -r+gamma*p                                              #TODO
   
    iterates_array = np.vstack([arr.T for arr in vec_sol])
    return x,vet_r,iterates_array,it

#Soluzione di sistemi sovradeterminati -----------------------------------

def eqnorm(A,b):
 
    G=A.T@A                                                         #TODO
    #cond = np.linalg.cond(G)
    f=A.T@b                                                         #TODO
    
    L=scipy.linalg.cholesky(G,lower=True)
    U=L.T

    z, flag = SolveTriangular.Lsolve(L,f)

    if flag == 0:
        x,flag = SolveTriangular.Usolve(U,z)

    return x


def qrLS(A,b):
    n=A.shape[1]  # numero di colonne di A
    Q,R=scipy.linalg.qr(A)
    h=Q.T                                                           #TODO
    x,flag = SolveTriangular.Usolve(R[0:n,:],h[0:n]) #to do
    residuo=np.linalg.norm(h[n:])**2                                #TODO
    return x,residuo



def SVDLS(A,b):
    m,n=A.shape  #numero di righe e  numero di colonne di A
    U,s,VT=scipy.linalg.svd(A)
    
    V=VT.T
    thresh=np.spacing(1)*m*s[0] ##Calcolo del rango della matrice, numero dei valori singolari maggiori di una soglia
    k=np.count_nonzero(s>thresh)                                    #TODO
    
    d=U.T@b                                                         #TODO
    d1=d[:k].reshape(k,1)                                           #TODO
    s1=s[:k].reshape(k,1)                                           #TODO
    
    c=d1/s1                                                         #TODO
    x=V[:,:k]@c                                                     #TODO
    residuo=np.linalg.norm(d[k:])**2                                #TODO
    return x,residuo
     

#-----------Interpolazione

def plagr(xnodi,j):
    
    xzeri=np.zeros_like(xnodi)
    n=xnodi.size
    if j==0:
       xzeri==xnodi[1:n]                                        #TODO
    else:
       xzeri=np.append(xnodi[0:j], xnodi[j+1:n])                #TODO
    
    num= np.poly(xzeri)
    den= np.polyval(num, xnodi[j])
    
    p= num/den
    
    return p



def InterpL(x, y, xx):
     
     n=x.size                                                   #TODO
     m=xx.size                                                  #TODO
     L=np.zeros((m,n))
     for j in range(n) :                                        #TODO
        p=plagr(x,j)                                            #TODO
        L[:,j]=np.polyval(p,xx)                                 #TODO
    
    
     return L@y                                                 #TODO