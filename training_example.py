# -*- coding: utf-8 -*-
import os 
os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # Using CPU, seems to be faster
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Input
tf.keras.backend.set_floatx('float32')
from data_org import dexc_dmrg_pkl as dexc_dmrg_pkl
from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()
L=24 # Largest system size in the training data
a=4

f = dexc_dmrg_pkl(sets = ['data/pickled_data/L18_num15000.pkl', 'data/pickled_data/L21_num15000.pkl', 'data/pickled_data/L24_num15000.pkl',
                     'data/pickled_data/L18_num1000_electric_field.pkl', 'data/pickled_data/L21_num1000_electric_field.pkl', 'data/pickled_data/L24_num1000_electric_field.pkl',
                     'data/pickled_data/L18_num1866_high_disorder.pkl', 'data/pickled_data/L18_num1866_high_disorder.pkl', 'data/pickled_data/L18_num1866_high_disorder.pkl',
                     'data/pickled_data/L24_num64_zeros.pkl'],
             L_max = L, a=a)  


rm = tf.keras.layers.Lambda(lambda x: x - tf.math.reduce_mean(x, axis = 1)[:,None])
@tf.function
def reverse2(x):
    return(tf.reverse(x, [1]))


n = 64
inputs_list = [Input(shape=((4*a+2))) for i in range(L+2*a)]
d1 = Dense(n) ;d2 = Dense(n); d3 = Dense(n); d4 = Dense(n); d5 = Dense(n);d6 = Dense(n);d7 = Dense(n);d8 = Dense(n); d10 = Dense(1)
a1 = tf.keras.layers.ELU()
with tf.GradientTape() as t:
    t.watch(inputs_list)
    elist = [d10(a1(d5(a1(d4(a1(d3(a1(d2(a1(d1(i))))))))))) for i in inputs_list]
    s = tf.keras.layers.add([i for i in elist])

grads = t.gradient(s, inputs_list)
v1 = tf.transpose(tf.convert_to_tensor([tf.keras.layers.Add()([grads[i+j][:,-1-j] for j in range(2*a+1)]) for i in range(L)]))
v2 = tf.transpose(tf.convert_to_tensor([tf.keras.layers.Add()([grads[i+j][:,-1-j-2*a-1] for j in range(2*a+1)]) for i in range(L)]))
v10 = rm(v1)/10
v20 = rm(v2)/10

i2 = [reverse2(x) for x in inputs_list] # For symmetry
with tf.GradientTape() as t2:
    t2.watch(i2)
    elist2 = [d10(a1(d5(a1(d4(a1(d3(a1(d2(a1(d1(i))))))))))) for i in i2]
    s2 = tf.keras.layers.add([i for i in elist2])

gradsf = t2.gradient(s2, i2)
v1f = tf.transpose(tf.convert_to_tensor([tf.keras.layers.Add()([gradsf[i+j][:,j] for j in range(2*a+1)]) for i in range(L)]))
v2f = tf.transpose(tf.convert_to_tensor([tf.keras.layers.Add()([gradsf[i+j][:,j+2*a+1] for j in range(2*a+1)]) for i in range(L)]))
v10f = rm(v1f)/10
v20f = rm(v2f)/10
vtot1 = (v10 + v10f)
vtot2 = (v20 + v20f)
stot = s + s2

model = Model(inputs = inputs_list, outputs = [stot, vtot1, vtot2])
opt = tf.keras.optimizers.legacy.Adam(learning_rate = 3E-4)
epochs = 5000
def scheduler(epoch, lr):
    func = 3E-4*(1/30)**(epoch/epochs)
    print(func)
    return func
callback = tf.keras.callbacks.LearningRateScheduler(scheduler)
model.compile(optimizer = opt, loss = 'MSE')
history = model.fit([x for x in f[0]], [f[1], f[2]/10, f[2]/10], epochs = epochs, batch_size = 32, shuffle = True, callbacks = [callback])
# Takes about 26s per epoch


model.save_weights('a'+str(a)+'_n'+str(n)+'test.h5')

