import matplotlib.pyplot as plt
import scipy
import numpy as np
import math

#
# functions to solve the dispersion relation
#

def solution(a, b, c):
    
    ''' Solve a 2nd order algebric equation    
    '''
    return (-b + np.sqrt(b**2 - 4* a * c))/(2*a)


# terms for the Bertin & Romeo dispersion relation (no drag)

def A(beta):

    return beta/16

def B(lamb, alpha, beta):

    return lamb *((beta+1)*lamb - (alpha+beta))/4

def C(lamb, alpha):

    return lamb**4 -lamb**3 *(alpha+1)

# terms for the Longarini 2022 dispersion relation without backreaction

def B_st(lamb, alpha, beta, st):

    return lamb *((beta + 1 + 1/(st**2))*lamb - (alpha + beta))/4

def C_st(lamb, alpha,st):

    return lamb**4 * (1 + st**(-2)) - lamb**3 * (1 + alpha + (1+alpha)/(st**2))

# terms for the Longarini 2022 dispersion relation with backreaction

def B_st_BR(lamb, alpha, beta, st):

    return lamb *((beta + 1 + 1/(st**2) + alpha*(1+beta) /(st**2) + 
                   alpha**2 * beta /(st**2))*lamb - (alpha+beta))/4

def C_st_BR(lamb, alpha,st):

    return lamb**4 * (1 + st**(-2) + 2*alpha/(st**2) + alpha**2 /(st**2)
                     )-lamb**3 *(alpha+1 + (1+alpha)/(st**2) + (2*alpha*(1+alpha))/(st**2) 
                                +(1+alpha)*alpha**2 /(st**2))


def marg_stab(lamb, alpha, beta, st, Q):

    ''' Marginal stability curve as a function of the dimensionless perturbation wavelength
    lamb = dimensionless perturbation wavelength (usually from 0 to 1+dtg ratio)
    alpha = dtg ratio
    beta = relative temperature
    st = stokes number
    Q = which marginal stability curve: Q=0 Bertin&Romeo (no drag), Q=1 No backreaction, Q=2 backreaction
    '''

    if (Q == 0):
        return solution(A(beta), B(lamb, alpha, beta), C(lamb, alpha))
    
    if (Q == 1):
        return solution(A(beta), B_st(lamb, alpha, beta, st), C_st(lamb, alpha, st))
    
    if (Q == 2):
        return solution(A(beta), B_st_BR(lamb, alpha, beta, st), C_st_BR(lamb, alpha, st))
    
    else:
        print('Error with Q')
        return 0

def const_jeanslength(hr, mstar, radius, sigmatot):

    return 2 * hr**2 * mstar/(radius * sigmatot)


def most_unst_wavelength_dless(lamb, alpha, beta, st, Q):

    ''' Most unstable wavelength in dimensionless unit 
    lamb = dimensionless perturbation wavelength (usually from 0 to 1+dtg ratio)
    alpha = dtg ratio
    beta = relative temperature
    st = stokes number
    Q = which marginal stability curve: Q=0 Bertin&Romeo (no drag), Q=1 No backreaction, Q=2 backreaction
    '''

    return lamb[np.argmax(marg_stab(lamb, alpha, beta, st, Q))]


def most_unst_wavelength_dimensional(lamb, alpha, beta, st, Q, hr, mstar, radius, sigmatot):

    ''' Most unstable wavelength in physics unit 
    lamb = dimensionless perturbation wavelength (usually from 0 to 1+dtg ratio)
    alpha = dtg ratio
    beta = relative temperature
    st = stokes number
    Q = which marginal stability curve: Q=0 Bertin&Romeo (no drag), Q=1 No backreaction, Q=2 backreaction
    hr = aspect ratio at the radius of interest
    radius = radius of interest
    sigmatot = total surface density at the radius of interest
    '''

    return most_unst_wavelength_dless(lamb, alpha, beta, st, Q
                ) * 2 * const_jeanslength(hr, mstar, radius, sigmatot)

def jeans_mass_dless(lamb, alpha, beta, st, Q):

    ''' Jeans mass in dimensionless units 
    lamb = dimensionless perturbation wavelength (usually from 0 to 1+dtg ratio)
    alpha = dtg ratio
    beta = relative temperature
    st = stokes number
    Q = which marginal stability curve: Q=0 Bertin&Romeo (no drag), Q=1 No backreaction, Q=2 backreaction
    '''

    return most_unst_wavelength_dless(lamb, alpha, beta, st, Q) ** 2

def jeans_mass_dimensional(lamb, alpha, beta, st, Q, hr, mstar, radius, sigmatot):

    ''' Jeans mass in physical units
    lamb = dimensionless perturbation wavelength (usually from 0 to 1+dtg ratio)
    alpha = dtg ratio
    beta = relative temperature
    st = stokes number
    Q = which marginal stability curve: Q=0 Bertin&Romeo (no drag), Q=1 No backreaction, Q=2 backreaction
    hr = aspect ratio at the radius of interest
    radius = radius of interest
    sigmatot = total surface density at the radius of interest
    '''
    
    return sigmatot * most_unst_wavelength_dimensional(
        lamb, alpha, beta, st, Q, hr, mstar, radius, sigmatot)**2