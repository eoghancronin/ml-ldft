import numpy as np
import json
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Input
from tools import semilocaldensities, semilocaldensities_flipped, solve_nonint
import time

L=24
a=1 #non-locality

######### Loading the disorder configurations v, and the corresponding exact ground state quantities #############

with open('data_json/exact_data/L'+str(L)+'_num1000.json', 'r') as f:
    data = json.load(f)
data = {'L':data['L'],
         'num':data['num'],
         'n_up':np.array(data['n_up']),
         'n_dn':np.array(data['n_dn']),
         'e_mb':np.array(data['e_mb']),
         'v':np.array(data['v']),
         'e_hxc':np.array(data['e_hxc']),
         'v_hxc':np.array(data['v_hxc']),
         'norm_tol':np.array(data['norm_tol']),
         'reverse_engineering_error':np.array(data['reverse_engineering_error'])
    }
n_dmrg = (data['n_up'] + data['n_dn'])/2
v = data['v']
e_dmrg = data['e_mb']
e_hxc_dmrg = data['e_hxc']
v_hxc_dmrg = data['v_hxc']
num=1000

rm = tf.keras.layers.Lambda(lambda x: x - tf.math.reduce_mean(x, axis = 1)[:,None])    

def construct_network(L, a):
    n = 64
    inputs_list = [Input(shape=((4*a+2))) for i in range(L+2*a)]
    d1 = Dense(n) ;d2 = Dense(n); d3 = Dense(n); d4 = Dense(n); d5 = Dense(n); d6 = Dense(1)
    a1 = tf.keras.layers.ELU()
    elist = [d6(a1(d5(a1(d4(a1(d3(a1(d2(a1(d1(i))))))))))) for i in inputs_list]
    s = tf.keras.layers.add([i for i in elist])
    opt = tf.keras.optimizers.Adam(learning_rate = 3E-4)
    model = Model(inputs = inputs_list, outputs = s)
    model.compile(optimizer = opt, loss = 'MAE')
    model.load_weights('trained_weights/a'+str(a)+'_n64sym.h5')
    return model

############### Mixing function, decays linearly with number of iterations, stops decaying after b iterations and stays at a value c ############

def mix_f(a,b,c,x):
    if x <= b:
        return c + a*(1 - x/b)
    else:
        return c

############## Solves the Kohn-Sham equations for the each of the disorder configuartions ################

def ks_scheme(V, delta, max_its, N,model,L,a):
    rm = tf.keras.layers.Lambda(lambda x: x - tf.math.reduce_mean(x, axis = 1)[:,None])
    num = len(V)
    n0 = np.repeat(N[:,None]/L,L,1)
    n1 = np.repeat(N[:,None]/L,L,1)
    successlist = np.repeat(np.array([False]), num)
    etalist = np.zeros(num)
    itslist = np.zeros(num)
    elist = np.zeros(num)
    n = tf.convert_to_tensor(semilocaldensities(L,a,n0))
    nf = tf.convert_to_tensor(semilocaldensities_flipped(L,a,n0))
    with tf.GradientTape(persistent=True) as t:
        t.watch(n)
        s = model([x for x in n])
    grads = t.gradient(s, n)
    with tf.GradientTape(persistent=True) as t:
        t.watch(nf)
        sf = model([x for x in nf])
    gradsf = t.gradient(sf, nf)
    vxc = tf.transpose(tf.convert_to_tensor([tf.keras.layers.Add()([grads[i+j][:,-1-j] for j in range(2*a+1)]) for i in range(L)]))
    vxc= rm(vxc) ; vxc = vxc.numpy()
    vxcf= tf.transpose(tf.convert_to_tensor([tf.keras.layers.Add()([gradsf[i+j][:,j] for j in range(2*a+1)]) for i in range(L)]))
    vxcf= rm(vxcf) ; vxcf = vxcf.numpy()
    vxc = vxc+vxcf

    veff = V + vxc
    
    for b in np.where(successlist == False)[0]:
        data = solve_nonint(veff[b], N[b], L)
        n1[b] = data[0]
        etalist[b] = np.sqrt(np.dot(n1[b]-n0[b],n1[b]-n0[b]))
    its = 0
    while (its < max_its) & ((successlist.all()) == False):
        t1 = time.time()
        mix = mix_f(0.025, 2000, 0.005, its)
        n0 = (1-mix)*n0 + mix*n1
        t3 = time.time()
        idx = np.where(successlist==False)[0]
        n = tf.convert_to_tensor(semilocaldensities(L,a,n0[idx]))
        nf = tf.convert_to_tensor(semilocaldensities_flipped(L,a,n0[idx]))
        with tf.GradientTape(persistent=True) as t:
            t.watch(n)
            t.watch(nf)
            s = model([x for x in n])
            sf = model([x for x in nf])
        grads = t.gradient(s, n)
        gradsf = t.gradient(sf, nf)
        vxc = tf.transpose(tf.convert_to_tensor([tf.keras.layers.Add()([grads[i+j][:,-1-j] for j in range(2*a+1)]) for i in range(L)]))
        vxc= rm(vxc) ; vxc = vxc.numpy()
        vxcf= tf.transpose(tf.convert_to_tensor([tf.keras.layers.Add()([gradsf[i+j][:,j] for j in range(2*a+1)]) for i in range(L)]))
        vxcf= rm(vxcf) ; vxcf = vxcf.numpy()
        vxc = vxc+vxcf
        exc = (s + sf).numpy()[:,0]
        veff = V[idx] + vxc
        #print('t_vxc ', time.time() - t3)
        t1 = time.time()
        for c, b in enumerate(idx):
            data = solve_nonint(veff[c], N[b], L) # This eigensolver only uses one thread, some improvement could be made here
            n1[b] = data[0]
            etalist[b] = np.sqrt(np.dot(n1[b]-n0[b],n1[b]-n0[b]))
            itslist[b] +=1
            elist[b] = 2*(data[1] - np.dot(data[0], veff[c])) + exc[c]
            if etalist[b] < delta:
                successlist[b] =True
        #print('t_diag', time.time()-t1)
        its +=1
        if its%50 == 0:
            print('iteration '+str(its)+'/'+str(max_its))
            print('converged points '+str(sum(successlist))+'/'+str(num))
            print('least converged datapoint', np.max(etalist))
            #print('min ', np.min(etalist))
            print(' ')
    n = tf.convert_to_tensor(semilocaldensities(L,a,n0))
    nf = tf.convert_to_tensor(semilocaldensities_flipped(L,a,n0))
    with tf.GradientTape(persistent=True) as t:
        t.watch(n)
        t.watch(nf)
        s = model([x for x in n])
        sf = model([x for x in nf])
    grads = t.gradient(s, n)
    gradsf = t.gradient(sf, nf)
    vxc = tf.transpose(tf.convert_to_tensor([tf.keras.layers.Add()([grads[i+j][:,-1-j] for j in range(2*a+1)]) for i in range(L)]))
    vxc= rm(vxc) ; vxc = vxc.numpy()
    vxcf= tf.transpose(tf.convert_to_tensor([tf.keras.layers.Add()([gradsf[i+j][:,j] for j in range(2*a+1)]) for i in range(L)]))
    vxcf= rm(vxcf) ; vxcf = vxcf.numpy()
    vxc = (vxc+vxcf)
    exc = (s + sf).numpy()[:,0]
    return (n0, elist, successlist, etalist, itslist, vxc, exc)

model = construct_network(L,a)
delta = 1E-5 
max_its = 1000 # I usually let it run for longer, not necessary here
t1 = time.time() # Should take 673 iterations, about 7 minutes (using the a=1 functional)
ksdata = ks_scheme(v, delta, max_its,(L//3)*np.ones(num, dtype = np.int32), model,L,a)
print(time.time()-t1)

e_mlks = ksdata[1]
n_mlks = ksdata[0]
v_hxc_mlks = ksdata[-2]
e_hxc_mlks = ksdata[-1]
e_mae = np.mean(np.abs(e_mlks - e_dmrg))/L # MAE per site
n_mae = np.mean(np.abs(n_mlks - n_dmrg))
e_hxc_mae = np.mean(np.abs(e_hxc_mlks - e_hxc_dmrg))/L
v_hxc_mae = np.mean(np.abs(v_hxc_mlks - v_hxc_dmrg))

print('e_mae = ', e_mae)
print('e_hxc_mae = ', e_hxc_mae)
print('n_mae = ', n_mae)
print('v_hxc_mae = ', v_hxc_mae)