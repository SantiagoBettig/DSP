#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# %matplotlib qt        -> Genera gráfico en ventana nueva
# %matplotlib inline    -> Gnera gráfico en el mismo spyder

"""
Created on Wed Sep  6 16:54:34 2023
@author: Santiago Bettig

Descripción:
El objetivo del script es modelar el comportamiento de un ADC, por lo que el 
input será una señal (en este caso, senoidal), se agregará ruido con
una distribución uniforme y luego se lo muestreara y cuantificará

"""

import numpy as np
import math
import matplotlib.pyplot as plt

def sin_gen(vmax = 1, dc = 0, ff = 1, ph = 0, nn = 1000, fs = 1000):
    # Lo primero que hay que hacer es crear un vector temporal que determinará
    # el inicio y el fin de la señal
    ts = 1/fs   # Tiempos de muestreo, intervalo de tiempo en el que se  
                # tomarán las muestras
    tf = nn * ts # Tiempo final de la señal
    tt = np.arange(0, tf, ts) # Creo el vector temporal
    ##
    # Ahora tengo que generar mi señal senoidal, para eso voy a aprovechar
    # la función np.sin de la biblioteca numpy. Para ello, debemo evaluar
    # dicha función en cada uno de los puntos del vector temporal creado
    # previamente
    xx = vmax * np.sin(2 * np.pi * ff * tt + ph) + dc
    ##
    # Devuelvo el vector temporal y el valor de la señal en cada uno de esos
    # puntos
    return tt, xx

# Sabemos que la varianza de una función aleatoria con distribución
# uniforme puede ser calculada como Var(X) = q²/12
def rnd_nmbrs_gen(var, N):
    q = math.sqrt(var*12);
    rnd_numbers = q * np.random.random(N) - (q/2);
    return rnd_numbers;

# El objetivo de la función es obtener una señal aleatoria a partir de una
# relación señal a ruido definida por parámetro, considerando como señal
# una senoidal de amplitud normalizada ( sqrt(2) ) 
def noise_gen(snr, N):
    # Lo primero que tenemos que hacer es obtener la varianza de la señal
    # aleatoria, tengamos en cuenta que esto está normalizado, por lo que
    # luego habría que desnormalizar
    var = np.power(10.0,snr/(-10.0));
    return rnd_nmbrs_gen(var, N);

# Defino los datos del ADC:

Vr = 2;                         # Tensión de referencia del ADC
n_bits = 16;                     # Cantidad de bits del ADC
quantum = Vr / (2**(n_bits-1)); # Calculo el valor del cuanto (Cada paso del ADC)

# Con esta frecuencia representaré la señal sobremuestreada 

f_cont  = 500       # Frecuencia de muestreo
N       = f_cont    # Cantidad de muestras

# Establezco la frecuencia de mi señal senoidal, la relación señal a ruido
# que tendrá y la amplitud de la misma (En este caso, unitaria)

f_sin   = 10; # Frecuencia de mi senoidal
snr     = 30; # Relación señal a ruido
A_sin   = 1;  # AMplitud de Senoidal

fig1, (ax1, ax2, ax3) = plt.subplots(3);

# Lo primero que voy a hacer es generarme la señal senoidal 
# sin_gen(Amp, DC, Frec, Phase, Muestras, Fs)
t, y_sin = sin_gen(A_sin, 0, f_sin, 0, N, f_cont)
ax1.plot(y_sin, color = 'green', label = 'Función Senoidal')
ax1.legend(loc = 'upper right');

# Luego voy a generar la señal aleatoria, donde el primero parámetro
# será la relación señal a ruido deseada y el segundo parámetro la 
# cantidad de muestras
y_noise = (A_sin/(math.sqrt(2))) * noise_gen(snr, N);
ax2.plot(y_noise, color = 'blue', label = 'Función Aleatoria')
ax2.legend(loc = 'upper right');

# Por último, voy a desnormalizar y realizar la suma de ambas
noisy_sin = y_sin + y_noise;
ax3.plot(noisy_sin, color = 'red', label = 'Senoidal Ruidosa')
ax3.legend(loc = 'upper right');

# Lo que voy a hacer ahora, para simular un muestreador, es quedarme con
# una cantidad menor de muestras, 1 cada ovs_rate muestras.
ovs_rate = 4;
noisy_sin_sampled = noisy_sin[::ovs_rate]
# Luego voy a realizar la cuantificación, para ello voy a generar el redondeo
# de cada una de las muestras y luego multiplicarlas por el cuanto
noisy_sin_quantif = np.round(noisy_sin_sampled/quantum, decimals = 0) * quantum;
# Por útlimo, voy a obtener el ruido de cuantificación haciendo la resta entre
# la señal original y la señal cuantizada
quantif_noise = noisy_sin_sampled - noisy_sin_quantif;

fig2, (ax1, ax2) = plt.subplots(2);

ax1.plot(noisy_sin_sampled, color = 'green', label = 'Señal Muestreada')
ax1.legend(loc = 'upper right');

ax1.plot(noisy_sin_quantif, color = 'blue', label = 'Señal Cuantificada')
ax1.legend(loc = 'upper right');

ax2.plot(quantif_noise/quantum, color = 'red', label = 'Ruido de Cuantificación')
ax2.legend(loc = 'upper right');

fig3, (ax1, ax2) = plt.subplots(2);

ax1.plot(quantif_noise/quantum, color = 'red', label = 'Ruido de Cuantificación')
ax1.legend(loc = 'upper right');

ax2.hist(quantif_noise/quantum, color = 'blue', label = 'Distribución');
ax2.legend(loc = 'upper right');

var_quantif_noise = np.var(quantif_noise);
var_quantif_noise_v2 = (quantum**2)/12;
correlate_quantif_noise = np.correlate(quantif_noise, quantif_noise)/N;
mean_quantig_noise = np.mean(quantif_noise);













