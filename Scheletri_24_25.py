"""
=====================================================================================
GUIDA PRATICA: COMANDI PER ANALIZZARE MATRICI E SCEGLIERE I METODI
=====================================================================================

# ANALISI DIMENSIONI E TIPO DI SISTEMA
print("Dimensioni matrice A:", A.shape)           # (m, n) - se m=n sistema quadrato, se m>n sovradeterminato
m, n = A.shape
if m == n:
    print("Sistema quadrato")
elif m > n:
    print("Sistema sovradeterminato")

# CONTROLLO SIMMETRIA
print("A è simmetrica?", np.allclose(A, A.T))     # True se simmetrica

# CALCOLO NUMERO DI CONDIZIONAMENTO
cond_A = np.linalg.cond(A)
print("Numero di condizionamento:", cond_A)
if cond_A < 1e2:
    print("Matrice ben condizionata")
elif cond_A < 1e6:
    print("Matrice mediamente mal condizionata")
else:
    print("Matrice altamente mal condizionata")

# CONTROLLO DEFINITA POSITIVA (solo per matrici simmetriche)
if np.allclose(A, A.T):
    autovalori = np.linalg.eigvals(A)
    if np.all(autovalori > 0):
        print("Matrice definita positiva")
    else:
        print("Matrice NON definita positiva")

# CONTROLLO DIAGONALE DOMINANZA
def is_diag_dominant(A):
    D = np.diag(np.abs(A))  # Elementi diagonali (valore assoluto)
    S = np.sum(np.abs(A), axis=1) - D  # Somma elementi non diagonali per riga
    return np.all(D > S)  # Diagonale dominanza stretta

if is_diag_dominant(A):
    print("Matrice a diagonale strettamente dominante")

# CALCOLO RANGO
rango = np.linalg.matrix_rank(A)
print(f"Rango della matrice: {rango} (dimensione: {min(A.shape)})")
if rango == min(A.shape):
    print("Matrice a rango massimo")

# CONTROLLO SPARSITÀ
sparsity = np.count_nonzero(A) / A.size
print(f"Densità matrice: {sparsity:.2%}")
if sparsity < 0.1:
    print("Matrice sparsa")
else:
    print("Matrice densa")

# SCELTA DEL METODO - GUIDA RAPIDA:
# 
# SISTEMI QUADRATI (m = n):
# - Ben condizionata, piccola, densa → Fattorizzazione di Gauss (scipy.linalg.solve)
# - Mal condizionata → Fattorizzazione QR (scipy.linalg.qr)
# - Simmetrica e definita positiva → Cholesky (scipy.linalg.cholesky)
# - Grande e sparsa, diag. dominante → Jacobi, Gauss-Seidel, SOR
# - Simmetrica, definita positiva, grande → Gradiente Coniugato, Steepest Descent
#
# SISTEMI SOVRADETERMINATI (m > n):
# - Ben condizionata, rango max → Equazioni Normali (eqnorm)
# - Mediamente mal condizionata → QR Least Squares (qrLS)
# - Altamente mal condizionata o non rango max → SVD Least Squares (SVDLS)

"""

"""
=====================================================================================
SCHEMA DI SELEZIONE DEI METODI PER RISOLVERE SISTEMI LINEARI Ax = b
=====================================================================================

if m=n

    if piccole dimensioni e densa
        if ben condizionata
            Gauss
        if mal condizionata
            fattorizzazione QR
        if simmetrica e definita positiva
            cholesky

    if grandi dimensioni e sparsa
        if diagonale strettamente dominante
            Jacobi, Gauss Seidel, Gauss Seidel SOR
        if simmetrica e definita positiva
            Gauss Seidel Gauss Seidel SOR Metodo di discesa del Gradiente-Gradiente Coniugato
if m>n

    if Matrice ben condizionata e a rango massimo
        Uso il metodo delle Equazioni normali
    if Matrice mediamente mal condizionata e a rango massimo
        Uso il metodo QRLS
    if Matrice altamente mal condizionata non a rango massimo
        Uso il metodo SVDLS
        
"""

#Zeri di funzione

import math
import numpy as np
import scipy.linalg as spLin
def sign(x):
  """
  Funzione segno che restituisce 1 se x è positivo, 0 se x è zero e -1 se x è negativo.
  """
  return math.copysign(1, x)

def metodo_bisezione(fname, a, b, tolx):
    """
    Metodo di bisezione per trovare lo zero di una funzione.
    
    Parametri:
    fname: funzione di cui trovare lo zero
    a, b: estremi dell'intervallo iniziale [a,b]
    tolx: tolleranza sull'errore
    
    Restituisce:
    xk: approssimazione dello zero
    it: numero di iterazioni
    v_xk: vettore delle approssimazioni successive
    
    Uso: x, it, v_x = metodo_bisezione(lambda x: x**2 - 2, 0, 2, 1e-6)
    """
    fa=fname(a)
    fb=fname(b)
    if fa*fb > 0:  # Controllo che f(a) e f(b) abbiano segni opposti
        print("Non è possibile applicare il metodo di bisezione \n")
        return None, None,None

    it = 0
    v_xk = []

    while (b-a) >= tolx:  # Criterio di arresto basato sulla lunghezza dell'intervallo
        xk = (a+b)/2  # Punto medio dell'intervallo
        v_xk.append(xk)
        it += 1
        fxk=fname(xk)
        if fxk==0:
            return xk, it, v_xk

        if fa*fxk < 0:  # Lo zero è nell'intervallo [a, xk]
            b = xk
            fb= fxk
        elif fb*fxk < 0:  # Lo zero è nell'intervallo [xk, b]
            a = xk
            fa = fxk

    return xk, it, v_xk

def falsa_posizione(fname,a,b,tolx,tolf,maxit):
    """
    Metodo della falsa posizione (regula falsi) per trovare lo zero di una funzione.
    
    Parametri:
    fname: funzione di cui trovare lo zero
    a, b: estremi dell'intervallo iniziale
    tolx: tolleranza sull'errore relativo
    tolf: tolleranza sulla funzione
    maxit: numero massimo di iterazioni
    
    Uso: x, it, v_x = falsa_posizione(lambda x: x**2 - 2, 0, 2, 1e-6, 1e-6, 100)
    """
    fa=fname(a)
    fb=fname(b)
    if fa*fb > 0:  # Controllo segni opposti
        print("Metodo di bisezione non applicabile")
        return None,None,None

    it=0
    v_xk=[]
    fxk=1+tolf
    errore=1+tolx
    xprec=a
    while it < maxit and abs(fxk) >= tolf and errore >= tolx:  # Triple criterio di arresto
        xk= a - fa*(b-a)/(fb-fa)  # Formula della falsa posizione (interpolazione lineare)
        v_xk.append(xk)
        it+=1
        fxk=fname(xk)
        if fxk==0:
            return xk,it,v_xk

        if fa*fxk < 0:  # Lo zero è in [a, xk]
           b= xk
           fb= fxk
        elif fb*fxk < 0:  # Lo zero è in [xk, b]
           a=xk
           fa=fxk
        if xk!=0:
            errore= abs(xk-xprec)/abs(xk)  # Errore relativo
        else:
            errore=abs(xk-xprec)  # Errore assoluto se xk=0
        xprec=xk
    return xk,it,v_xk

def corde(fname,coeff_ang,x0,tolx,tolf,nmax):
    """
    Metodo delle corde per trovare lo zero di una funzione.
    
    Parametri:
    fname: funzione di cui trovare lo zero
    coeff_ang: coefficiente angolare fisso della retta (approssimazione della derivata)
    x0: punto iniziale
    tolx, tolf: tolleranze
    nmax: numero massimo iterazioni
    
    Uso: x, it, v_x = corde(lambda x: x**2 - 2, 2, 1.5, 1e-6, 1e-6, 100)
    """
    # coeff_ang è il coefficiente angolare della retta che rimane fisso per tutte le iterazioni
    xk=[]
    
    it=0
    errorex=1+tolx
    erroref=1+tolf
    while it < nmax and errorex >= tolx and erroref >= tolf:  # Triple criterio di arresto
       
       fx0=fname(x0)  # Valore della funzione nel punto corrente
       d=coeff_ang  # Coefficiente angolare fisso (non si aggiorna come in Newton)
      
       x1=x0 - fx0/d  # Formula delle corde: x_{k+1} = x_k - f(x_k)/m
       fx1=fname(x1)  # Valore della funzione nel nuovo punto
       if x1!=0:
            errorex=abs(x1-x0)/abs(x1)  # Errore relativo
       else:
            errorex=abs(x1-x0)  # Errore assoluto
       
       erroref=abs(fx1)  # Errore sulla funzione
       
       x0=x1
       it=it+1
       xk.append(x1)
      
    if it==nmax:
        print('Corde : raggiunto massimo numero di iterazioni \n')
        
    
    return x1,it,xk
    
def newton(fname,fpname,x0,tolx,tolf,nmax):
    """
    Metodo di Newton-Raphson per trovare lo zero di una funzione.
    
    Parametri:
    fname: funzione di cui trovare lo zero
    fpname: derivata prima della funzione
    x0: punto iniziale
    tolx, tolf: tolleranze
    nmax: numero massimo iterazioni
    
    Uso: x, it, v_x = newton(lambda x: x**2 - 2, lambda x: 2*x, 1.5, 1e-6, 1e-6, 100)
    """
    xk=[]
   
    it=0
    errorex=1+tolx
    erroref=1+tolf
    while it < nmax and errorex >= tolx and erroref >= tolf:  # Triple criterio di arresto
       
       fx0=fname(x0)
       fpx0=fpname(x0)  # Derivata prima nel punto corrente
       if abs(fpx0) < np.finfo(float).eps:  # Controllo se derivata è praticamente zero
            print(" derivata prima nulla in x0")
            return None, None,None
       d=fpx0  # La derivata è il coefficiente angolare della tangente

       x1=x0 - fx0/d  # Formula di Newton: x_{k+1} = x_k - f(x_k)/f'(x_k)
       fx1=fname(x1)
       erroref=np.abs(fx1)
       if x1!=0:
            errorex=abs(x1-x0)/abs(x1)  # Errore relativo
       else:
            errorex=abs(x1-x0)  # Errore assoluto

       it=it+1
       x0=x1
       xk.append(x1)
      
    if it==nmax:
        print('Newton: raggiunto massimo numero di iterazioni \n')
        
    
    return x1,it,xk

def newton_modificato(fname,fpname,m,x0,tolx,tolf,nmax):
    """
    Metodo di Newton modificato per zeri multipli.
    
    Parametri:
    fname: funzione di cui trovare lo zero
    fpname: derivata prima della funzione
    m: molteplicità dello zero
    x0: punto iniziale
    tolx, tolf: tolleranze
    nmax: numero massimo iterazioni
    
    Uso: x, it, v_x = newton_modificato(lambda x: (x-1)**2, lambda x: 2*(x-1), 2, 1.5, 1e-6, 1e-6, 100)
    """
    #m è la molteplicità dello zero

    xk=[]
   
    it=0
    errorex=1+tolx
    erroref=1+tolf
    while it < nmax and errorex >= tolx and erroref >= tolf:  # Triple criterio di arresto
       
       fx0=fname(x0)
       fpx0=fpname(x0)
       if abs(fpx0) < np.finfo(float).eps:  # Controllo derivata nulla
            print(" derivata prima nulla in x0")
            return None, None,None
       d=fpx0  # Derivata prima

       x1=x0 - m*fx0/d  # Formula Newton modificato: x_{k+1} = x_k - m*f(x_k)/f'(x_k)
       fx1=fname(x1)
       erroref=np.abs(fx1)
       if x1!=0:
            errorex=abs(x1-x0)/abs(x1)  # Errore relativo
       else:
            errorex=abs(x1-x0)  # Errore assoluto

       it=it+1
       x0=x1
       xk.append(x1)
      
    if it==nmax:
        print('Newton modificato: raggiunto massimo numero di iterazioni \n')
        
    
    return x1,it,xk
    
def secanti(fname,xm1,x0,tolx,tolf,nmax):
    """
    Metodo delle secanti per trovare lo zero di una funzione.
    
    Parametri:
    fname: funzione di cui trovare lo zero
    xm1: punto x_{k-1}
    x0: punto x_k
    tolx, tolf: tolleranze
    nmax: numero massimo iterazioni
    
    Usa due punti precedenti per approssimare la derivata con differenze finite.
    Uso: x, it, v_x = secanti(lambda x: x**2 - 2, 1.0, 1.5, 1e-6, 1e-6, 100)
    """
    xk=[]
    
    it=0
    errorex=1+tolx
    erroref=1+tolf
    while it < nmax and errorex >= tolx and erroref >= tolf:  # Triple criterio di arresto
        
        fxm1=fname(xm1)  # f(x_{k-1})
        fx0=fname(x0)    # f(x_k)
        d=(fx0-fxm1)/(x0-xm1)  # Approssimazione derivata con differenze finite

        x1=x0 - fx0/d  # Formula secanti: x_{k+1} = x_k - f(x_k)/f'_approx(x_k)
      
        
        fx1=fname(x1)
        xk.append(x1)
        if x1!=0:
            errorex=abs(x1-x0)/abs(x1)  # Errore relativo
        else:
            errorex=abs(x1-x0)  # Errore assoluto
            
        erroref=abs(fx1)  # Errore sulla funzione
        xm1=x0  # Aggiornamento: x_{k-1} diventa x_k
        x0=x1   # x_k diventa x_{k+1}
        
        it=it+1
       
   
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
    """
    Metodo di Newton-Raphson per sistemi di equazioni non lineari.
    
    Parametri:
    initial_guess: vettore iniziale
    F_numerical: funzione che calcola F(x) (vettore delle funzioni)
    J_Numerical: funzione che calcola lo Jacobiano J(x)
    tolX, tolF: tolleranze
    max_iterations: numero massimo iterazioni
    
    Risolve F(x) = 0 dove F: R^n -> R^n
    Uso: x, it, err = newton_raphson([1,1], F_func, J_func, 1e-6, 1e-6, 100)
    """

    X= np.array(initial_guess, dtype=float)
    
   

    it=0
    
    erroreF=1+tolF
    erroreX=1+tolX
    
    errore=[]
    
    while it < max_iterations and erroreX >= tolX and erroreF >= tolF:  # Triple criterio
        jx = J_Numerical(X[0], X[1])  # Calcola matrice Jacobiana nel punto corrente
        if np.linalg.matrix_rank(jx) < len(X):  # Controllo rango massimo
            print("La matrice dello Jacobiano calcolata nell'iterato precedente non è a rango massimo")
            return None, None, None
        fx = F_numerical(X[0], X[1])  # Calcola F(x)
        fx = fx.squeeze()
        s = np.linalg.solve(jx, -fx)  # Risolve J(x_k)·s = -F(x_k)
        Xnew = X + s  # x_{k+1} = x_k + s
        normaXnew = np.linalg.norm(Xnew, 1)
        if normaXnew != 0:
            erroreX = np.linalg.norm(s, 1) / normaXnew  # Errore relativo
        else:
            erroreX = np.linalg.norm(s, 1)  # Errore assoluto
        errore.append(erroreX)
        fxnew = F_numerical(Xnew[0], Xnew[1])  # F(x_{k+1})
        erroreF = np.linalg.norm(fxnew.squeeze(), 1)
        X = Xnew
        it = it + 1
    
    return X,it,errore

def newton_raphson_corde(initial_guess, F_numerical, J_Numerical, tolX, tolF, max_iterations):
    """
    Metodo di Newton-Raphson con variante delle corde per sistemi nonlineari.
    Lo Jacobiano viene aggiornato solo ogni N iterazioni invece di ogni iterazione.
    
    Parametri: come newton_raphson standard
    Uso: x, it, err = newton_raphson_corde([1,1], F_func, J_func, 1e-6, 1e-6, 100)
    """

    X= np.array(initial_guess, dtype=float)
    
   

    it=0
    
    erroreF=1+tolF
    erroreX=1+tolX
    
    errore=[]
    
    while it < max_iterations and erroreX >= tolX and erroreF >= tolF:
        if it == 0:  # Calcola Jacobiano solo alla prima iterazione (o ogni N iterazioni)
            jx = J_Numerical(X[0], X[1])
            if np.linalg.matrix_rank(jx) < len(X):
                print("La matrice dello Jacobiano calcolata nell'iterato precedente non è a rango massimo")
                return None, None, None
        fx = F_numerical(X[0], X[1])
        fx = fx.squeeze()
        s = np.linalg.solve(jx, -fx)  # Usa sempre lo stesso Jacobiano
        Xnew = X + s
        normaXnew = np.linalg.norm(Xnew, 1)
        if normaXnew != 0:
            erroreX = np.linalg.norm(s, 1) / normaXnew
        else:
            erroreX = np.linalg.norm(s, 1)
        errore.append(erroreX)
        fxnew = F_numerical(Xnew[0], Xnew[1])
        erroreF = np.linalg.norm(fxnew.squeeze(), 1)
        X = Xnew
        it = it + 1
    
    return X,it,errore


def newton_raphson_sham(initial_guess, update, F_numerical, J_Numerical, tolX, tolF, max_iterations):
    """
    Metodo di Newton-Raphson con strategia di Shamanskii.
    Lo Jacobiano viene aggiornato ogni 'update' iterazioni.
    
    Parametri:
    update: ogni quante iterazioni aggiornare lo Jacobiano
    Altri parametri: come newton_raphson standard
    
    Uso: x, it, err = newton_raphson_sham([1,1], 3, F_func, J_func, 1e-6, 1e-6, 100)
    """
    

    X= np.array(initial_guess, dtype=float)
    
   

    it=0
    
    erroreF=1+tolF
    erroreX=1+tolX
    
    errore=[]
    
    while it < max_iterations and erroreX >= tolX and erroreF >= tolF:
        if it % update == 0:  # Aggiorna Jacobiano ogni 'update' iterazioni
            jx = J_Numerical(X[0], X[1])
            if np.linalg.matrix_rank(jx) < len(X):
                print("La matrice dello Jacobiano calcolata nell'iterato precedente non è a rango massimo")
                return None, None, None
        fx = F_numerical(X[0], X[1])
        fx = fx.squeeze()
        s = np.linalg.solve(jx, -fx)
        Xnew = X + s
        normaXnew = np.linalg.norm(Xnew, 1)
        if normaXnew != 0:
            erroreX = np.linalg.norm(s, 1) / normaXnew
        else:
            erroreX = np.linalg.norm(s, 1)
        errore.append(erroreX)
        fxnew = F_numerical(Xnew[0], Xnew[1])
        erroreF = np.linalg.norm(fxnew.squeeze(), 1)
        X = Xnew
        it = it + 1
    
    return X,it,errore




#Minimo di una funzion enon lineare

def newton_raphson_minimo(initial_guess, grad_func, Hessian_func, tolX, tolF, max_iterations):
    """
    Metodo di Newton-Raphson per trovare il minimo di una funzione.
    
    Parametri:
    initial_guess: punto iniziale
    grad_func: funzione che calcola il gradiente
    Hessian_func: funzione che calcola la matrice Hessiana
    tolX, tolF: tolleranze
    max_iterations: numero massimo iterazioni
    
    Risolve ∇f(x) = 0 per trovare punti critici (minimi, massimi, selle)
    Uso: x, it, err = newton_raphson_minimo([1,1], grad_f, hess_f, 1e-6, 1e-6, 100)
    """

    X= np.array(initial_guess, dtype=float)
    
    it=0
    
    erroreF=1+tolF
    erroreX=1+tolX
    
    errore=[]
    
    while it < max_iterations and erroreX >= tolX and erroreF >= tolF:
        
        Hx = Hessian_func(X)  # Calcola matrice Hessiana
        
        if np.linalg.matrix_rank(Hx) < len(X):  # Controllo rango massimo
            print("La matrice Hessiana calcolata nell'iterato precedente non è a rango massimo")
            return None, None,None
        
        gfx = grad_func(X)  # Calcola gradiente
        gfx = gfx.squeeze() 
        
        s = np.linalg.solve(Hx, -gfx)  # Risolve H(x_k)·s = -∇f(x_k)
        
        Xnew=X + s  # x_{k+1} = x_k + s
        
        normaXnew=np.linalg.norm(Xnew,1)
        if normaXnew!=0:
            erroreX=np.linalg.norm(s,1)/normaXnew  # Errore relativo
        else:
            erroreX=np.linalg.norm(s,1)  # Errore assoluto
            
        errore.append(erroreX)
        gfxnew=grad_func(Xnew)  # Gradiente nel nuovo punto
        erroreF= np.linalg.norm(gfxnew.squeeze(),1)
        X=Xnew
        it=it+1
    
    return X,it,errore

#Metodi Iterativi basati sullo splitting della matrice: jacobi, gauss-Seidel - Gauss_seidel SOR
def jacobi(A,b,x0,toll,it_max):
    """
    Metodo iterativo di Jacobi per risolvere sistemi lineari Ax = b.
    
    Parametri:
    A: matrice del sistema
    b: vettore termini noti
    x0: vettore iniziale
    toll: tolleranza
    it_max: numero massimo iterazioni
    
    Splitting: A = D - E - F dove D=diag(A), E=-tril(A,-1), F=-triu(A,1)
    Iterazione: x^{(k+1)} = D^{-1}(E+F)x^{(k)} + D^{-1}b
    
    Uso: x, it, err = jacobi(A, b, x0, 1e-6, 1000)
    """
    errore=1000
    d=np.diag(A)  # Elementi diagonali
    n=A.shape[0]
    invM=np.diag(1/d)  # D^{-1}
    E=-np.tril(A,-1)  # Parte triangolare inferiore stretta cambiata di segno
    F=-np.triu(A,1)   # Parte triangolare superiore stretta cambiata di segno
    N=E+F  # N = E + F (matrice non diagonale)
    T=invM @ N  # Matrice di iterazione T = D^{-1}(E+F)
    autovalori=np.linalg.eigvals(T)
    raggiospettrale=max(abs(autovalori))  # Raggio spettrale = max|λ_i|
    print("raggio spettrale jacobi", raggiospettrale)
    it=0
    
    er_vet=[]
    while it<=it_max and errore>=toll:
        x=invM @ (N @ x0 + b)  # x^{(k+1)} = D^{-1}((E+F)x^{(k)} + b)
        errore=np.linalg.norm(x-x0, ord=np.inf)  # Errore tra iterazioni successive
        er_vet.append(errore)
        x0=x.copy()
        it=it+1
    return x,it,er_vet


def gauss_seidel(A,b,x0,toll,it_max):
    """
    Metodo iterativo di Gauss-Seidel per risolvere sistemi lineari Ax = b.
    
    Parametri: come Jacobi
    
    Splitting: A = D - E - F dove D=diag(A), E=-tril(A,-1), F=-triu(A,1)
    M = D - E (triangolare inferiore), N = F (triangolare superiore)
    Iterazione: (D-E)x^{(k+1)} = Fx^{(k)} + b
    
    Uso: x, it, err = gauss_seidel(A, b, x0, 1e-6, 1000)
    """
    errore=1000
    d=np.diag(A)  # Elementi diagonali
    D=np.diag(d)  # Matrice diagonale
    E=-np.tril(A,-1)  # Parte triangolare inferiore stretta cambiata di segno
    F=-np.triu(A,1)   # Parte triangolare superiore stretta cambiata di segno
    M=D-E  # M = D - E (triangolare inferiore)
    N=F    # N = F (triangolare superiore)
    T=np.linalg.inv(M) @ N  # Matrice di iterazione T = M^{-1}N
    autovalori=np.linalg.eigvals(T)
    raggiospettrale=max(abs(autovalori))  # Raggio spettrale
    print("raggio spettrale Gauss-Seidel ",raggiospettrale)
    it=0
    er_vet=[]
    while it<=it_max and errore>=toll:  # Condizioni di arresto
        x=np.linalg.solve(M, N @ x0 + b)  # Risolve (D-E)x^{(k+1)} = Fx^{(k)} + b
        errore=np.linalg.norm(x-x0, ord=np.inf)  # Errore tra iterazioni successive
        er_vet.append(errore)
        x0=x.copy()
        it=it+1
    return x,it,er_vet

def gauss_seidel_sor(A,b,x0,toll,it_max,omega):
    """
    Metodo di Gauss-Seidel con sovrarilassamento (SOR).
    
    Parametri:
    omega: parametro di rilassamento (0 < omega < 2)
           omega = 1: Gauss-Seidel standard
           omega < 1: sottorilassamento  
           omega > 1: sovrarilassamento
    Altri parametri: come Gauss-Seidel
    
    M_ω = D + ωE, N_ω = (1-ω)D - ωF
    Iterazione: (D + ωE)x^{(k+1)} = ((1-ω)D - ωF)x^{(k)} + ωb
    
    Uso: x, it, err = gauss_seidel_sor(A, b, x0, 1e-6, 1000, 1.2)
    """
    errore=1000
    d=np.diag(A)  # Elementi diagonali
    D=np.diag(d)  # Matrice diagonale
    E=-np.tril(A,-1)  # Parte triangolare inferiore stretta cambiata di segno
    F=-np.triu(A,1)   # Parte triangolare superiore stretta cambiata di segno
    Momega=D+omega*E  # M_ω = D + ωE
    Nomega=(1-omega)*D-omega*F  # N_ω = (1-ω)D - ωF
    T=np.linalg.inv(Momega) @ Nomega  # Matrice di iterazione T = M_ω^{-1}N_ω
    autovalori=np.linalg.eigvals(T)
    raggiospettrale=max(abs(autovalori))  # Raggio spettrale
    print("raggio spettrale Gauss-Seidel SOR ", raggiospettrale)
    
    M=Momega  # M = M_ω
    N=Nomega  # N = N_ω
    it=0
    xold=x0.copy()
    xnew=x0.copy()
    er_vet=[]
    while it<=it_max and errore>=toll:
        
        xtilde=np.linalg.solve(M, N @ xold + omega*b)  # Passo di rilassamento
        xnew=xtilde  # Aggiornamento soluzione
        errore=np.linalg.norm(xnew-xold, ord=np.inf)  # Errore tra iterazioni
        er_vet.append(errore)
        xold=xnew.copy()
        it=it+1
    return xnew,it,er_vet


#Metodi di Discesa

def steepestdescent(A,b,x0,itmax,tol):
    """
    Metodo del gradiente (steepest descent) per risolvere Ax = b.
    
    Parametri:
    A: matrice simmetrica definita positiva
    b: vettore termini noti
    x0: vettore iniziale
    itmax: numero massimo iterazioni
    tol: tolleranza
    
    Minimizza f(x) = (1/2)x^T A x - b^T x
    Gradiente: ∇f(x) = Ax - b = -r (residuo cambiato di segno)
    
    Uso: x, err, sol, it = steepestdescent(A, b, x0, 1000, 1e-6)
    """
 
    n,m=A.shape
    if n!=m:
        print("Matrice non quadrata")
        return [],[]
    
    
   # inizializzare le variabili necessarie
    x = x0

     
    r = A @ x - b  # Residuo r = Ax - b
    p = -r  # Direzione di discesa = -gradiente
    it = 0
    nb=np.linalg.norm(b)
    errore=np.linalg.norm(r)/nb
    vec_sol=[]
    vec_sol.append(x.copy())
    vet_r=[]
    vet_r.append(errore)
     
# utilizzare il metodo del gradiente per trovare la soluzione
    while errore >= tol and it < itmax:  # Condizioni di arresto
        it=it+1
        Ap=A @ p  # Prodotto matrice-vettore
       
        alpha = (r.T @ r) / (p.T @ Ap)  # Passo ottimale
                
        x = x + alpha * p  # Aggiornamento soluzione
        
         
        vec_sol.append(x.copy())
        r=A @ x - b  # Nuovo residuo
        errore=np.linalg.norm(r)/nb
        vet_r.append(errore)
        p =-r  # Nuova direzione = -gradiente
        
    iterates_array = np.vstack([arr.T for arr in vec_sol])
    return x,vet_r,iterates_array,it

def conjugate_gradient(A,b,x0,itmax,tol):
    """
    Metodo del gradiente coniugato per risolvere Ax = b.
    
    Parametri: come steepest descent
    
    Genera direzioni coniugate rispetto ad A: p_i^T A p_j = 0 per i≠j
    Convergenza teorica in n passi per sistemi n×n
    
    Uso: x, err, sol, it = conjugate_gradient(A, b, x0, 1000, 1e-6)
    """
    n,m=A.shape
    if n!=m:
        print("Matrice non quadrata")
        return [],[]
    
    
   # inizializzare le variabili necessarie
    x = x0
    
    r = A @ x - b  # Residuo iniziale
    p = -r  # Prima direzione di ricerca = -gradiente
    it = 0
    nb=np.linalg.norm(b)
    errore=np.linalg.norm(r)/nb
    vec_sol=[]
    vec_sol.append(x0.copy())
    vet_r=[]
    vet_r.append(errore)
# utilizzare il metodo del gradiente coniugato per calcolare la soluzione
    while errore >= tol and it < itmax:  # Condizioni di arresto
        it=it+1
        Ap=A @ p  # Prodotto matrice-vettore
        alpha = (r.T @ r) / (p.T @ Ap)  # Passo ottimale
        x = x + alpha * p  # Aggiornamento soluzione
        vec_sol.append(x.copy())
        rtr_old=r.T @ r  # Prodotto scalare del residuo precedente
        r= A @ x - b  # Nuovo residuo
        gamma=(r.T @ r) / rtr_old  # Coefficiente per coniugazione
        errore=np.linalg.norm(r)/nb
        vet_r.append(errore)
        p = -r + gamma * p  # Nuova direzione coniugata
   
    iterates_array = np.vstack([arr.T for arr in vec_sol])
    return x,vet_r,iterates_array,it

#Soluzione di sistemi sovradeterminati

def eqnorm(A,b):
    """
    Risoluzione di sistemi sovradeterminati tramite equazioni normali.
    
    Parametri:
    A: matrice m×n con m > n (più equazioni che incognite)
    b: vettore termini noti
    
    Risolve min ||Ax - b||_2 tramite A^T A x = A^T b
    
    Uso: x = eqnorm(A, b)
    """
 
    G= A.T @ A  # Matrice di Gram G = A^T A (n×n, simmetrica, definita positiva)
    f=A.T @ b   # Vettore A^T b
    
    L=spLin.cholesky(G, lower=True)  # Fattorizzazione di Cholesky G = L L^T
    U=L.T  # U = L^T
        
    y=spLin.solve_triangular(L, f, lower=True)  # Risolve Ly = f
    x=spLin.solve_triangular(U, y, lower=False) # Risolve Ux = y
        
    return x


def qrLS(A,b):
    """
    Risoluzione di sistemi sovradeterminati tramite fattorizzazione QR.
    
    Parametri:
    A: matrice m×n con m > n
    b: vettore termini noti
    
    Usa fattorizzazione A = QR dove Q ortogonale, R triangolare superiore
    Risolve min ||Ax - b||_2 tramite Rx = Q^T b
    
    Uso: x, res = qrLS(A, b)
    """
    n=A.shape[1]  # numero di colonne di A
    Q,R=spLin.qr(A)  # Fattorizzazione QR
    h=Q.T @ b  # h = Q^T b
    x,residuals,rank,s=spLin.lstsq(R[:n,:], h[:n])  # Risolve Rx = h (parte superiore)
    residuo=np.linalg.norm(b - A @ x)  # Calcolo del residuo ||b - Ax||
    return x,residuo



def SVDLS(A,b):
    """
    Risoluzione di sistemi sovradeterminati tramite decomposizione SVD.
    
    Parametri:
    A: matrice m×n con m > n
    b: vettore termini noti
    
    Usa SVD: A = U Σ V^T dove U (m×m), Σ (m×n), V (n×n)
    Soluzione: x = V Σ^+ U^T b (pseudoinversa di Moore-Penrose)
    
    Uso: x, res = SVDLS(A, b)
    """
    m,n=A.shape  #numero di righe e  numero di colonne di A
    U,s,VT=spLin.svd(A)  # Decomposizione SVD
    
    V=VT.T  # V = (V^T)^T
    thresh=np.spacing(1)*m*s[0] ##Calcolo del rango della matrice, numero dei valori singolari maggiori di una soglia
    k=np.sum(s > thresh)  # Rango numerico (conta valori singolari > soglia)
    
    d=U.T @ b  # d = U^T b
    d1=d[:k]   # Primi k elementi di d
    s1=s[:k]   # Primi k valori singolari
    
    c=d1/s1    # c = d1 ./ s1 (divisione elemento per elemento)
    x=V[:,:k] @ c  # x = V_k c (prime k colonne di V)
    residuo=np.linalg.norm(b - A @ x)  # Residuo
    return x,residuo
     

#-----------Interpolazione

def plagr(xnodi,j):
    """
    Calcola il j-esimo polinomio di base di Lagrange.
    
    Parametri:
    xnodi: vettore dei nodi di interpolazione
    j: indice del polinomio di base (0, 1, ..., n)
    
    Restituisce: 
    p: funzione polinomiale L_j(x) = ∏_{k≠j} (x-x_k)/(x_j-x_k)
    
    Uso: L_j = plagr(x_nodes, j); poi valutare con L_j(x_eval)
    """
    
    xzeri=np.zeros_like(xnodi)
    n=xnodi.size
    if j==0:
       xzeri=xnodi[1:]  # Tutti i nodi tranne il primo (j=0)
    else:
       xzeri=np.append(xnodi[:j], xnodi[j+1:])  # Tutti i nodi tranne il j-esimo
    
    num=np.poly(xzeri)   # Numeratore: polinomio con zeri in xzeri
    den=np.polyval(num, xnodi[j])  # Denominatore: valore del numeratore in x_j
    
    p=num/den  # Polinomio di Lagrange normalizzato
    
    return p



def InterpL(x, y, xx):
    """
    Interpolazione di Lagrange per calcolare il polinomio interpolante.
    
    Parametri:
    x: vettore dei nodi di interpolazione (n+1 punti)
    y: valori della funzione nei nodi f(x_i) = y_i
    xx: punti dove valutare il polinomio interpolante
    
    Calcola p(x) = Σ_{i=0}^n y_i L_i(x) dove L_i sono i polinomi di Lagrange
    
    Uso: y_interp = InterpL(x_nodes, y_values, x_eval)
    """
    
    n=len(x)      # Numero di nodi
    m=len(xx)     # Numero di punti di valutazione
    L=np.zeros((m,n))  # Matrice dei polinomi di Lagrange valutati
    for j in range(n):  # Per ogni nodo
        p=plagr(x,j)     # Calcola j-esimo polinomio di Lagrange
        L[:,j]=np.polyval(p, xx)  # Valuta L_j(xx)
    
    
    return L @ y  # Polinomio interpolante: p(xx) = Σ y_j L_j(xx)


# determinante
A = np.array([
    [3.0, 2.0, 1.0, -1.0],
    [4.0, 6.0, 3.0, 2.0],
    [2.0, 1.0, 4.0, 3.0],
    [1.0, 4.0, 2.0, 7.0]
])
P, L, U = scipy.linalg.lu(A)
print(P)
print(L)
print(U)

det_P= -1
det_U = np.prod(np.diag(U))
det_A=det_P*det_U
print(det_A)
print(np.linalg.det(A))

# Calcolo dell'inversa di A tramite la fattorizzazione LU (senza usare np.linalg.inv)
n = A.shape[0]
A_inv_LU = np.zeros_like(A)
I = np.eye(n)

for i in range(n):
    e = I[:, i]
    # Risolvi P @ y = e
    y = np.linalg.solve(P, e)
    # Risolvi L @ z = y
    z = np.linalg.solve(L, y)
    # Risolvi U @ x = z
    x = np.linalg.solve(U, z)
    # x è la colonna i-esima dell'inversa
    A_inv_LU[:, i] = x

print("Inversa calcolata con LU:\n", A_inv_LU)
print("Inversa calcolata con numpy:\n", np.linalg.inv(A))  # solo per confronto


plt.semilogy(errore_nr, label='Newton-Raphson')
plt.semilogy(errore_nrc, label='Corde')
plt.semilogy(errore_nrs, label='Shamanskii')
plt.xlabel('Iterazione')
plt.ylabel('Errore relativo tra iterati')
plt.title('Confronto convergenza metodi')
plt.legend()
plt.grid(True, which='both')
plt.show()