import numpy as np
from scipy.special import legendre

def LegendreFit(x,y,deg=5):
    c = np.polynomial.legendre.Legendre.fit(x,y,deg=deg).convert().coef
    return c

def getLegendreValues(x,c):
    """
    Obtain the legendre values 
        val = \sum_i P_i(x)
    """
    N = len(c)

    y = np.zeros_like(x)
    for i in range(N):
        Pi = legendre(i)
        y += c[i] * Pi(x)
    
    return y

