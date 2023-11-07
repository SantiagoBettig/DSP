#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# %matplotlib qt        -> Genera gráfico en ventana nueva
# %matplotlib inline    -> Genera gráfico en el mismo spyder

# Ventanas: Boxcar (Lo mismo que no ventanear), Flattop, etc

"""
Created on Wed Oct 4 18:43:23 2023
@author: Santiago Bettig

Descripción:
    El objetivo de la siguiente tarea es estimar la frecuencia de
    la señal a través de dos estimadores distintos utilizando por un lado
    el método de Welch y por otro el de Blackman-Tukey

"""

import math
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sg
import spectrum as sp

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

# Con esta frecuencia representaré la señal analógica 
# (voy a trabajar normalizado)

f_cont  = 1000      # Frecuencia continua
N       = f_cont    # Cantidad de muestras de señal continua

# Establezco  la relación señal a ruido que tendrá y la amplitud 
# de la misma (Potencia unitaria)

snr     = 20                 # Relación señal a ruido
A_sin   = math.sqrt(2);     # Amplitud de Senoidal para potencia normalizada

# Por otro lado, la experiencia tendrá "n_run" realizaciones, por lo que me voy
# a crear una matriz vacía que contemple todas las muestras

n_run = 200;
y_sin = np.empty((n_run, N), float);
y_noise = np.empty((n_run, N), float);
noisy_sin = np.empty((n_run, N), float);

# Me creo un eje temporal con el que voy a plotear mis señales
# t_axis = np.arange(0, N*1/f_cont, 1/f_cont);

for i in range(n_run):
    # Primero voy a generar la secuencia de números aleatorios que va representar
    # mi variación en frecuencia
    q = 0.5;
    ohmega_0 = f_cont/4;
    fr = np.random.uniform(-q,q);
    ohmega = ohmega_0 + fr
    
    # Ahora voy a generar mi senoidal con la frecuencia obtenida anteriormente
    t, y_sin[i] = sin_gen(A_sin, 0, ohmega, 0, N, f_cont);
    
    # Luego, voy a calcular el ruido que debo adicionar a la senoidal, con
    # la SNR deseada:
    y_noise[i] = (A_sin/(math.sqrt(2))) * noise_gen(snr, N);
    
    # La señal final será la suma de ambas:
    noisy_sin[i] = y_sin[i] + y_noise[i];

# Este margen lo voy a usar para plotear solamente el area de interes
margin  = 10;
M       = 100;
 
# Armo unos vectores de frecuencia para poder plotear los resultados
frec_axis_cont      = np.arange(0, f_cont, f_cont/N);
frec_axis_cont_filt = (frec_axis_cont > (f_cont/4 - margin)) & (frec_axis_cont <= (f_cont/4 + margin));

fft_noisy_sin       = np.empty((n_run, N), float);

fft_welch_sin       = np.empty((n_run, N), float);
f_welch_sin         = np.empty((n_run, N), int);

frec_axis_blackman  = np.arange(0, N, 1);
fft_bt_sin          = np.empty((n_run, N), float)

fig1, (ax1)         = plt.subplots(1);

# Ahora voy a hacer una fft para conocer el espectro de cada una de esas 
# señales que obtuve
for i in range(n_run):
    fft_noisy_sin[i]                    = 20*np.log10(np.abs(np.fft.fft(noisy_sin[i])/N));
    f_welch_sin[i], fft_welch_sin[i]    = 20*np.log10(np.abs(sg.welch(noisy_sin[i], f_cont, 'bartlett', N, return_onesided = False)));
    fft_bt_sin[i]                       = 20*np.log10(np.abs(sp.cshift(sp.CORRELOGRAMPSD(X = noisy_sin[i], Y = noisy_sin[i], lag = M, window = 'bartlett', NFFT = N), N/2)/(N/2)));
    
    ax1.plot(frec_axis_cont[frec_axis_cont_filt], fft_noisy_sin[i][frec_axis_cont_filt], 'x:g')    
    ax1.plot(frec_axis_cont[frec_axis_cont_filt], fft_welch_sin[i][frec_axis_cont_filt], '--b')
    ax1.plot(frec_axis_cont[frec_axis_cont_filt], fft_bt_sin[i][frec_axis_cont_filt], 'x:r')    
    
# Ahora voy a obtener el estimador de frecuencia, para ello voy abuscar el valor 
# mas alto de amplitud y me voy a quedar con el argumento de dicho valor
# (el cual representaría el valor de frecuencia)
frec_axis_cont_filt     = frec_axis_cont <= f_cont/2;
frec_axis_blackman_filt = frec_axis_blackman <= f_cont/2;
est_frec_noisy          = np.empty((n_run, 1), float);
est_frec_welch          = np.empty((n_run, 1), float);
est_frec_bt             = np.empty((n_run, 1), float);

for i in range(n_run):
    est_frec_noisy[i]   = np.argmax(fft_noisy_sin[i][frec_axis_cont_filt]);
    est_frec_welch[i]   = np.argmax(fft_welch_sin[i][frec_axis_cont_filt]);
    est_frec_bt[i]      = np.argmax(fft_bt_sin[i][frec_axis_blackman_filt]);

# Voy a imprimir los resultados:
fig4, (ax1) = plt.subplots(1);
plt.title('Estimación Frecuencia')
ax1.hist(est_frec_noisy, color = 'blue', label = 'BoxCar', bins = 10);
ax1.hist(est_frec_welch, color = 'orange', label = 'Welch', bins = 10);
ax1.hist(est_frec_bt, color = 'red', label = 'Blackman', bins = 10);
ax1.legend(loc = 'upper right');

print("\nEstimación de Frecuencia\n")
# Obtengo los estadísticos de cada estimador
boxcar_frec_mean    =  np.mean(est_frec_noisy);
boxcar_frec_var     =  np.var(est_frec_noisy);
print("BoxCar -> La media es de:", boxcar_frec_mean, "y su varianza:", boxcar_frec_var)
welch_frec_mean     =  np.mean(est_frec_welch);
welch_frec_var      =  np.var(est_frec_welch);
print("Welch -> La media es de:", welch_frec_mean, "y su varianza:", welch_frec_var)
blackman_frec_mean  =  np.mean(est_frec_bt);
blackman_frec_var   =  np.var(est_frec_bt);
print("Blackman -> La media es de:", blackman_frec_mean, "y su varianza:", blackman_frec_var)