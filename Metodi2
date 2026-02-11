#Zeri di funzione

import math
def sign(x):
  """
  Funzione segno che restituisce 1 se x è positivo, 0 se x è zero e -1 se x è negativo.
  """
  return math.copysign(1, x)

def metodo_bisezione(fname, a, b, tolx):

 fa=fname(a)
 fb=fname(b)
 if sign(fa)*sign(fb) > 0:#to do
     print("Non è possibile applicare il metodo di bisezione \n")
     return None, None,None

 it = 0
 v_xk = []



 while abs(b-a) > tolx:#to do
    xk = a + (b-a)/2#to do
    v_xk.append(xk)
    it += 1
    fxk=fname(xk)
    if fxk==0:
      return xk, it, v_xk

    if sign(fb) *sign(fxk) <0: # to do
      a = xk#to do
      fa= fxk#to do
    elif sign(fa) * sign(fxk) <0:# to do
      b = xk
      fb= fxk


 return xk, it, v_xk

def falsa_posizione(fname,a,b,tolx,tolf,maxit):
    fa=fname(a)
    fb=fname(b)
    if sign(fa*fb) > 0 :#to do:
       print("Metodo di bisezione non applicabile")
       return None,None,None

    it=0
    v_xk=[]
    fxk=1+tolf
    errore=1+tolx
    xprec=a
    while it<maxit and abs(fxk) > tolf and errore>tolx: #to do:
        xk= a - (b-a) / (fb-fa)#to do
        v_xk.append(xk)
        it+=1
        fxk=fname(xk)# to do
        if fxk==0:
            return xk,it,v_xk

        if sign(fa*fk) <0:#to do
           b= xk#to do
           fb= fxk#to do
        elif sign(fb*fk) <0:#to do
           a= xk#to do
           fa= fxk#to do
        if xk!=0:
            errore= abs(xk-xprec)/abs(xk)#to do
        else:
            errore= abs(xk-xprec)#to do
        xprec=xk
    return xk,it,v_xk

def corde(fname,coeff_ang,x0,tolx,tolf,nmax):

     # coeff_ang è il coefficiente angolare della retta che rimane fisso per tutte le iterazioni
        xk=[]

        it=0
        errorex=1+tolx
        erroref=1+tolf
        while it<nmax and erroref>= tolf and errorex>=tolx:#to do

           fx0= fname(x0)# to do
           d= f(x)/coeff_ang # to do

           x1= x0 - d#to do
           fx1=fname(x1)#
           if x1!=0:
                errorex=abs(d)/abs(x1)#to do
           else:
                errorex=abs(d)#to do

           erroref=np.abs(fx1)#to do

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
        while it<nmax and errorex>=tolx and erroref>=tolf:#to do

           fx0=fname(x0)
           if abs(fpname(x0)) < np.spacing(1):#to do
                print(" derivata prima nulla in x0")
                return None, None,None
           d= fx0/fpname(x0)#to do

           x1= x1 - d#to do
           fx1=fname(x1)
           erroref=np.abs(fx1)
           if x1!=0:
                errorex=abs(d)/abs(x1)#to do
           else:
                errorex=abs(d)#to do

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
        while it<nmax and errorex>=tolx and erroref>=tolf:#to do

           fx0=fname(x0)
           if abs(fpname(x0)) < np.spacing(1):#to do
                print(" derivata prima nulla in x0")
                return None, None,None
           d= fx0 / fpname(x0)#to do

           x1=x0 - m*d#to do
           fx1=fname(x1)
           erroref=np.abs(fx1)
           if x1!=0:
                errorex=abs(d)/abs(x1)#to do
           else:
                errorex=abs(d)#to do

           it=it+1
           x0=x1
           xk.append(x1)

        if it==nmax:
            print('Newton modificato: raggiunto massimo numero di iterazioni \n')


        return x1,it,xk

def secanti(fname,xm1,x0,tolx,tolf,nmax):
        xk=[]

        it=0
        errorex=1+tolx
        erroref=1+tolf
        while it<nmax and errorex>=tolx and errroref>=tolf:#to do

            fxm1=fname(xm1)#to do
            fx0=fname(x0)#to do
            d=fx0 * (x0-xm1)/(fx0-fxm1)#to do

            x1=x0 - d#to do


            fx1=fname(x1)
            xk.append(x1);
            if x1!=0:
                errorex=abs(d)/abs(x1)#to do
            else:
                errorex=abs(d)#to do

            erroref=abs(fx1)#to do
            xm1=x0#to do
            x0= x1#to do

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

    while it<max_iterations and erroreF>=tolF and erroreX>=tolX:#to do

        jx = J_Numerical(X[0],X[1])#to do

        if np.linalg.det(jx) == 0 :#to do
            print("La matrice dello Jacobiano calcolata nell'iterato precedente non è a rango massimo")
            return None, None,None

        fx = F_numerical(X[0],X[1])#To do
        fx = fx.squeeze()

        s = np.linalg.solve(jx,-fx)#to do

        Xnew= X + s#to do

        normaXnew=np.linalg.norm(Xnew,1)
        if normaXnew !=0:
            erroreX=np.linalg.norm(s,1)/normaXnew#to do
        else:
            erroreX=np.linalg.norm(s,1)#to do

        errore.append(erroreX)
        fxnew=F_numerical(Xnew[0],Xnew[1])#to do
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

    while it<nmax and erroreX>=tolX and erroreF>=tolF:#to do

        if it == 0:#to do
            jx = J_Numerical(X[0],X[1]) #to do

            if np.linalg.det(jx) == 0:#to do
                print("La matrice dello Jacobiano calcolata nell'iterato precedente non è a rango massimo")
                return None, None,None

        fx = F_numerical(X[0],X[1])#To do
        fx = fx.squeeze()

        s = np.linalg.solve(jx,-fx)#to do

        Xnew= X + s#to do

        normaXnew=np.linalg.norm(Xnew,1)
        if normaXnew !=0:
            erroreX=np.linal.norm(s,1)/normaXnew#to do
        else:
            erroreX=np.linal.norm(s,1)#to do

        errore.append(erroreX)
        fxnew=F_numerical(X[0],X[1])#to do
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

    while it<nmax and erroreX>=tolX and erroreF>=tolF:#to do

        if it%update == 0: # to to:
            jx = J_Numerical(X[0],X[1])#to do

            if np.linalg.det(jx) == 0:#to do
                print("La matrice dello Jacobiano calcolata nell'iterato precedente non è a rango massimo")
                return None, None,None

        fx = F_numerical(X[0],X[1])#To do
        fx = fx.squeeze()

        s = np.linalg.solve(jx,-fx)#to do

        Xnew=X+s#to do

        normaXnew=np.linalg.norm(Xnew,1)
        if normaXnew !=0:
            erroreX=np.linalg.norm(s,1)/normaXnew#to do
        else:
            erroreX=np.linalg.norm(s,1)#to do

        errore.append(erroreX)
        fxnew=F_numerical(Xnew[0],Xnew[1])#to do
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

    while it<nmax and erroreX>=tolX and erroreF>=tolF:#to do:

        Hx = Hessian_func(X[0],X[1])#to do

        if np.linalg.det(Hx) == 0:#to do
            print("La matrice Hessiana calcolata nell'iterato precedente non è a rango massimo")
            return None, None,None

        gfx =  grad_func(X[0],X[1])#to do
        gfx = gfx.squeeze()

        s = np.linalg.det(Hx,-gfx)#to do

        Xnew=X+s#to do

        normaXnew=np.linalg.norm(Xnew,1)
        if normaXnew!=0:
            erroreX=np.linalg.norm(s,1)/normaXnew#to do
        else:
            erroreX=np.linalg.norm(s,1)#to do

        errore.append(erroreX)
        gfxnew=grad_func(Xnew[0],Xnew[0])#to do
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
    E=np.tril(A,-1)#to do
    F=np.triu(A,1)#to do
    N=-(E+F)#to do
    T=invM@N#to do
    autovalori=np.linalg.eigvals(T)
    raggiospettrale=np.max(np.abs(autovalori))#to do
    print("raggio spettrale jacobi", raggiospettrale)
    it=0

    er_vet=[]
    while it<=it_max and errore>=toll:
        x= (b+N@x0) / d.reshape(n,1) #to do
        errore= np.linalg.norm(x-x0)/np.linalg.norm(x)#to do
        er_vet.append(errore)
        x0=x.copy()
        it=it+1
    return x,it,er_vet


def gauss_seidel(A,b,x0,toll,it_max):
    errore=1000
    d=np.diag(A)#to do
    D=np.diag(d)#to do
    E=np.tril(A,-1)#to do
    F=np.triu(A,1)#to do
    M=E+D#to do
    N=-F#to do
    T=np.dot(np.linalg.inv(M),N)#to do
    autovalori=np.linalg.eigvals(T)
    raggiospettrale=np.max(np.abs(autovalori))#to do
    print("raggio spettrale Gauss-Seidel ",raggiospettrale)
    it=0
    er_vet=[]
    while it<=it_max and errore>=toll:#to do
        temp = b - F@x0
        x,flag = SolveTriangular.Lsolve(M,temp)
        errore=np.linalg.norm(x-x0)/np.linalg.norm(x)#to do
        er_vet.append(errore)
        x0=x.copy()
        it=it+1
    return x,it,er_vet

def gauss_seidel_sor(A,b,x0,toll,it_max,omega):
    errore=1000
    d=np.diag(A)#to do
    D=np.diag(d)#to do
    E=np.tril(A,-1)#to do
    F=np.tril(A,1)#to do
    Momega=D+omega*E
    Nomega=(1-omega)*D-omega*F
    T=np.dot(np.linalg.inv(Momega),Nomega)#to do
    autovalori=np.linalg.eigvals(T)
    raggiospettrale=np.max(np.abs(autovalori))#to do
    print("raggio spettrale Gauss-Seidel SOR ", raggiospettrale)

    M=E+D#to do
    N=-F#to do
    it=0
    xold=x0.copy()
    xnew=x0.copy()
    er_vet=[]
    while it<=it_max and errore>=toll:
        temp = (b-F@x0)
        xtilde,flag = SolveTriangular.Lsolve(M,temp)#to do
        xnew=(1-omega) * xold + omega * xtilde#to do
        errore=np.linalg.norm(xnew-xold)/np.linalg.norm(xnew)#to do
        er_vet.append(errore)
        xold=xnew.copy()
        it=it+1
    return xnew,it,er_vet


#Metodi di Discesa

def steepestdescent(A,b,x0,itmax,tol):

    n,m=A.shape
    if n!=m:
        print("Matrice non quadrata")
        return [],[]


   # inizializzare le variabili necessarie
    x = x0


    r = A@x-b#to do
    p = -r#to do
    it = 0
    nb=np.linalg.norm(b)
    errore=np.linalg.norm(r)/nb
    vec_sol=[]
    vec_sol.append(x.copy())
    vet_r=[]
    vet_r.append(errore)

# utilizzare il metodo del gradiente per trovare la soluzione
    while it<itmax and errore>=tol:#to do
        it=it+1
        Ap=A@p#to do

        alpha = - (r.T@p) / (p.T@Ap)# to do

        x = x + alpha *p#to do


        vec_sol.append(x.copy())
        r= r + alpha *Ap#to do
        errore=np.linalg.norm(r)/nb
        vet_r.append(errore)
        p = -r#to do

    iterates_array = np.vstack([arr.T for arr in vec_sol])
    return x,vet_r,iterates_array,it

def conjugate_gradient(A,b,x0,itmax,tol):
    n,m=A.shape
    if n!=m:
        print("Matrice non quadrata")
        return [],[]


   # inizializzare le variabili necessarie
    x = x0

    r = A@x-b#to do
    p = -r #to do
    it = 0
    nb=np.linalg.norm(b)
    errore=np.linalg.norm(r)/nb
    vec_sol=[]
    vec_sol.append(x0.copy())
    vet_r=[]
    vet_r.append(errore)
# utilizzare il metodo del gradiente coniugato per calcolare la soluzione
    while it<itmax and errore>=tol:#to do
        it=it+1
        Ap=A@p#to do A.dot(p)
        alpha = - (r.T@p) / (p.T@Ap)#to do
        x = x + alpha*p#to do
        vec_sol.append(x.copy())
        rtr_old=r.T@r#to do
        r= r + alpha*Ap#to do
        gamma= r.T@r / rtr_old#to do
        errore=np.linalg.norm(r)/nb
        vet_r.append(errore)
        p = -r#to do

    iterates_array = np.vstack([arr.T for arr in vec_sol])
    return x,vet_r,iterates_array,it

#Soluzione di sistemi sovradeterminati

def eqnorm(A,b):

    G= A.T@A#to do
    f= A.T@b#to do

    L= scipy.linalg.cholesky(G, lower=True)
    U= L.T

    z,flag = SolveTriangular.Lsolve(L,f)
    if flag ==0:
        x,flag = SolveTriangular.Usolve(U,z)

    return x


def qrLS(A,b):
    n=A.shape[1]  # numero di colonne di A
    Q,R=spLin.qr(A)
    h=Q.T#to do
    x,flag = SolveTriangular.Lsolve(R[0:n,:], h[0:n]) #to do
    residuo=np.linalg.norm(h[n:])**2#to do
    return x,residuo



def SVDLS(A,b):
    m,n=A.shape  #numero di righe e  numero di colonne di A
    U,s,VT=spLin.svd(A)

    V=VT.T
    thresh=np.spacing(1)*m*s[0] ##Calcolo del rango della matrice, numero dei valori singolari maggiori di una soglia
    k=np.count_nonzero(s>thres)#to do

    d=U.T@b#to do
    d1=d[:k].reshape(k,1)#to do
    s1=s[:k].reshape(k,1)#to do

    c=d1/s1#to do
    x=V[:,:k]@c#to do
    residuo=np.linalg.norm(d[k:])**2#to do
    return x,residuo


#-----------Interpolazione

def plagr(xnodi,j):

    xzeri=np.zeros_like(xnodi)
    n=xnodi.size
    if j==0:
       xzeri=xnodi[1:n]#to do
    else:
       xzeri=np.append(xnodi[0:j], xnodi[j+1:n])#to do )

    num= np.poly(xzeri)
    den= np.polyval(num,xnodi[j])

    p= num/den

    return p



def InterpL(x, y, xx):

     n=x.size#to do
     m=xx.size#to do
     L=np.zeros((m,n))
     for j in range(n): #to do :
        p=plagr(x,j)#to do
        L[:,j]=np.polyval(p,xx)#to do


     return L@y#to do


# PER CALCOLARE LA COSTANTE DI LEBESQUE:
def Lebesque(points,xx):

    L_matrix = np.zeros((len(xx),len(points)))

    for j in range(len(points)):
        p = plagr(points,j)
        L_matrix[:,j] = np.polyval(p,xx)

    lebes_f = np.sum(np.abs(L_matrix), axis=1)
    return np.max(lebes_f)

# PER CALCOLARE L'INVERSA SENZA FUNZIONE:
def inverse(A,b):
    m,n = A.shape
    if m!=n:
        print("Matrice non quadrata")
        return

    X=np.zeros((n,n))

    P,L,U = scipy.linalg.lu(A)

    for i in range(n):
        y,flagL = SolveTriangular.Lsolve(L, P.T@b[:,i])
        x,flagU = SolveTriangular.Usolve(U,y)

        if flagL==0 and flagU==0:
            X[:,i] = x.reshape(n,)

    return X
