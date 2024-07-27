#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import pickle
from scipy.linalg import eigh_tridiagonal

def load_datasets(sets, L_max, a,):
    xx = []
    yy = []
    dydx = []
    for b, x in enumerate(sets):
        with open(x, 'rb') as f:
            data = pickle.load(f)
        # Since we avoid systems with magnetisation, the spin up and down densities should be equal.
        # I enforce this by averaging the two
        # The discrepancy between n_up and n_dn is a numerical artefact from the DMRG
        # Typically, this error is of order 1E-6 to 1E-9 
        n = (data['n_up'] + data['n_dn'])/2
        n = np.pad(n, ((0,0), (2*a, 2*a + L_max-data['L'])))
        n = np.array([np.concatenate([n[:,i:i+2*a+1], n[:,i:i+2*a+1]], axis = 1) for i in range(L_max +2*a)])
        e_hxc = data['e_hxc']
        v_hxc = data['v_hxc']
        v_hxc = v_hxc - np.mean(v_hxc, axis = 1)[:,None]
        v_hxc = np.pad(v_hxc,((0,0),(0,L_max-data['L'])))
        
        xx.append(n)
        yy.append(e_hxc)
        dydx.append(v_hxc)
    return [np.concatenate(xx, axis = 1), np.concatenate(yy), np.concatenate(dydx, axis = 0)]

def solve_nonint(V, n, L):
    vals, vecs = eigh_tridiagonal(V, -1*np.ones(L-1), select='i', select_range=(0,n-1), check_finite=False, tol=0.0, lapack_driver='auto')
    density = np.square(vecs).sum(axis=1)
    return density, vals.sum()

def semilocaldensities(L,a,d):
    d = np.pad(d, ((0,0), (2*a, 2*a)))
    return np.array([np.concatenate([d[:,i:i+2*a+1], d[:,i:i+2*a+1]], axis = 1) for i in range(L+2*a)])

def semilocaldensities_flipped(L,a,d):
    d = np.pad(d, ((0,0), (2*a, 2*a)))
    d = np.array([np.concatenate([d[:,i:i+2*a+1], d[:,i:i+2*a+1]], axis = 1) for i in range(L+2*a)])
    return np.flip(d, axis = 2)