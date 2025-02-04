import os 
os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # Disabling GPU, CPU seems to be faster
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Suppressing warnings
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Input
tf.keras.backend.set_floatx('float32')
from tools import load_datasets
from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()

########## Loading Datasets ##########
L=24 # Largest system size in the training data
a=1 # non-locality
'''
f = load_datasets(sets = ['data/exact_data/L18_num15000.json', 'data/exact_data/L21_num15000.json', 'data/exact_data/L24_num15000.json',
                     'data/exact_data/L18_num1000_electric_field.json', 'data/exact_data/L21_num1000_electric_field.json', 'data/exact_data/L24_num1000_electric_field.json',
                     'data/exact_data/L18_num1866_high_disorder.json', 'data/exact_data/L18_num1866_high_disorder.json', 'data/exact_data/L18_num1866_high_disorder.json',
                     'data/exact_data/L24_num64_zeros.json'],
             L_max = L, a=a)  '''
f = load_datasets(sets = ['data_json/exact_data/L18_num15000.json', 'data_json/exact_data/L21_num15000.json', 'data_json/exact_data/L24_num15000.json',
                         'data_json/exact_data/L18_num1000_electric_field.json', 'data_json/exact_data/L21_num1000_electric_field.json', 
                          'data_json/exact_data/L24_num1000_electric_field.json',
                         'data_json/exact_data/L18_num1866_high_disorder.json', 'data_json/exact_data/L18_num1866_high_disorder.json', 
                          'data_json/exact_data/L18_num1866_high_disorder.json',
                         'data_json/exact_data/L24_num64_zeros.json'],
             L_max = L, a=a)  

########## Defining the model, should take about a minute ##########
rm = tf.keras.layers.Lambda(lambda x: x - tf.math.reduce_mean(x, axis = 1)[:,None])
@tf.function
def reverse2(x):
    return(tf.reverse(x, [1])) # We flip the semilocal densities, input both to the neural network and sum the two outputs to force a spatial symmetry. Basically, g(x) = f(x) + f(-x)

n = 64 # number of nodes per layer
inputs_list = [Input(shape=((4*a+2))) for i in range(L+2*a)]
d1 = Dense(n) ;d2 = Dense(n); d3 = Dense(n); d4 = Dense(n); d5 = Dense(n); d10 = Dense(1)
a1 = tf.keras.layers.ELU()
with tf.GradientTape() as t:
    t.watch(inputs_list)
    elist = [d10(a1(d5(a1(d4(a1(d3(a1(d2(a1(d1(i))))))))))) for i in inputs_list]
    s = tf.keras.layers.add([i for i in elist])

grads = t.gradient(s, inputs_list)
v_dn = tf.transpose(tf.convert_to_tensor([tf.keras.layers.Add()([grads[i+j][:,-1-j] for j in range(2*a+1)]) for i in range(L)]))
v_up = tf.transpose(tf.convert_to_tensor([tf.keras.layers.Add()([grads[i+j][:,-1-j-2*a-1] for j in range(2*a+1)]) for i in range(L)]))
v_up = rm(v_up)
v_dn = rm(v_dn)

i2 = [reverse2(x) for x in inputs_list] # For symmetry
with tf.GradientTape() as t2:
    t2.watch(i2)
    elist2 = [d10(a1(d5(a1(d4(a1(d3(a1(d2(a1(d1(i))))))))))) for i in i2]
    s2 = tf.keras.layers.add([i for i in elist2])

gradsf = t2.gradient(s2, i2)
v_dn_f = tf.transpose(tf.convert_to_tensor([tf.keras.layers.Add()([gradsf[i+j][:,j] for j in range(2*a+1)]) for i in range(L)]))
v_up_f = tf.transpose(tf.convert_to_tensor([tf.keras.layers.Add()([gradsf[i+j][:,j+2*a+1] for j in range(2*a+1)]) for i in range(L)]))
v_up_f = rm(v_up_f)
v_dn_f = rm(v_dn_f)
v_up_tot = (v_up+ v_up_f)
v_dn_tot= (v_dn + v_dn_f)
s_tot = s + s2               

model = Model(inputs = inputs_list, outputs = [s_tot, v_up_tot, v_dn_tot])

########## Training the model ##########
opt = tf.keras.optimizers.legacy.Adam(learning_rate = 3E-4)
epochs = 20 # I trained my models for 5000 epochs
def scheduler(epoch, lr):
    func = 3E-4*(1/30)**(epoch/epochs)
    return func
callback = tf.keras.callbacks.LearningRateScheduler(scheduler)
model.compile(optimizer = opt, loss = 'MSE')
history = model.fit([x for x in f[0]], [f[1], f[2], f[2]], epochs = epochs, batch_size = 32, shuffle = True, callbacks = [callback])
# Takes about 26s per epoch

#model.save_weights('a'+str(a)+'_n'+str(n)+'test.h5')