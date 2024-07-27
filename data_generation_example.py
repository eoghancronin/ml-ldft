#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import tenpy
from tenpy.models.fermions_hubbard import FermionicHubbardChain
from tenpy.networks.mps import MPS
from tenpy.algorithms import dmrg
import time
import numpy as np
import pickle
from tools import solve_nonint
from scipy.optimize import minimize as minimize
L=18
N=L//3
x = (L+1)/2

delta = np.sqrt(np.linspace(0,9,5)) # This distribution of the magnitude of disorder gives a good corresponding distribution of the densities
num = len(delta)

model_params = dict(cons_N='N', cons_Sz='Sz', t=1, U=4, mu=0, V=0, lattice="Chain", bc_MPS='finite',
                        order='default', L=L)
product_state = ["up", "down",0]*(L//3)
n_up = np.zeros((num, L))
n_dn = np.zeros((num, L))
v = np.zeros((num, L))
v_hxc = np.zeros((num, L))
norms = np.zeros(num)
d_err = np.zeros(num)
e = np.zeros(num)
emv = np.zeros(num)
e_ni = np.zeros(num)
e_hxc = np.zeros(num)
times = np.zeros(num)

for a, w in enumerate(delta):
    x = FermionicHubbardChain(model_params)
    x.manually_call_init_H = True
    V = w*np.random.random(L); V=V-np.mean(V)
    #V = efield(w, L)
    for b, y in enumerate(V):
        x.add_onsite_term(strength = y , i=b, op = 'Cdd Cd')
        x.add_onsite_term(strength = y , i=b, op = 'Cdu Cu')
    x.init_H_from_terms()
    psi = MPS.from_product_state(x.lat.mps_sites(), product_state, bc=x.lat.bc_MPS)
    dmrg_params = {
            'active_sites': 2,
            'mixer': False,
            'trunc_params': {'chi_max': 350},
            'N_sweeps_check': 10,
            'norm_tol':1E-6,
            'max_hours': 1}
    t1 = time.time()
    info = dmrg.run(psi, x, dmrg_params)
    times[a] = time.time() - t1
    n_dn[a] = psi.expectation_value('Cdd Cd', [i for i in range(L)])
    n_up[a] = psi.expectation_value('Cdu Cu', [i for i in range(L)])
    e[a] = info['E']
    emv[a] = e[a] - np.dot(n_up[a] + n_dn[a], V)
    v[a] = V
    norms[a] = info['sweep_statistics']['norm_err'][-1]
    print(str(a + 1)+'/'+str(num)+' DMRG calculations completed')
    print('DMRG time '+str(time.time() - t1))
    print('DMRG truncation error '+str(norms[a]))
    
def cost_n(n,v, states):
    def cost(vxc):
        diag = solve_nonint(v+vxc, N, L)
        return np.sqrt(np.sum(np.square(diag[0] - n)))
    return cost

n_ = (n_up + n_dn)/2 # They should be roughly equal anyway... The difference between the two is a numerical artefact from the DMRG and is quite small

for j in range(num):
    V = v[j]
    n = n_[j]
    t1 = time.time()
    success = False
    temp_cost = cost_n(n, V, N)
    pred = minimize(temp_cost, np.zeros(L),method = 'BFGS', jac = '2-point', options = {'finite_diff_rel_step':1E-9})
    if pred.fun < 1E-6:
        success = True
    if success == True:
        result = solve_nonint(V + pred.x, N, L)
        e_ni[j] = result[1] - np.dot(V+pred.x, result[0])
        d_err[j] = pred.fun
        v_hxc[j] = pred.x - np.mean(pred.x)
        print(str(j + 1)+'/'+str(num)+' Reverse-engineering calculations completed')
        print('Reverse-engineering time', time.time() - t1)
        print('Reverse engineering error '+str(d_err[j]))

e_hxc = emv - 2*e_ni

data = {'L': L, 
     'num' : num, 
     'n_up': n_up, 
     'n_dn': n_dn, 
     'e_mb': emv, 
     'v': v,
     'e_hxc': e_hxc,
     'v_hxc': v_hxc, 
     'norm_tol': norms, 
     'reverse_engineering_error': d_err}

# Uncomment below to save the data
'''
with open('data/exact_data/L'+str(L)+'_num'+str(num)+'.pkl', 'wb') as f2:
        pickle.dump(data, f2)
'''