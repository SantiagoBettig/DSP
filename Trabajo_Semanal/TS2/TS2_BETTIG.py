#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# %matplotlib qt        -> Genera gráfico en ventana nueva
# %matplotlib inline    -> Gnera gráfico en el mismo spyder

"""
Created on Spyder3
@author: Santiago Bettig
Descripción:
    
    El objetivo de la tarea es desarrollar un algoritmo que calcule la 
    transformada discreta de fourier (DFT), cuyo parámetro de entrada será
    una matriz de Nx1 números reales, su salida un matriz de Nx1 números
    complejos
    
"""

# Se importan las bibliotecas para realizar operaciones matemáticas 
# no standard (numpy) y la biblioteca que permite imprimir gráficos en 
# pantalla (matplotlib)
import numpy as np
import math 
import matplotlib.pyplot as plt

# Creo la función que me permitirá obtener la DFT.
def my_dft(xx):
    # Lo primero que hago es obtener la cantidad de muestras que 
    # tiene mi función
    N = np.size(xx)
    # Creo el vector donde terminará almacenandose el resultado final
    dft_output = np.zeros(N,dtype=np.complex128)
    # Hago el doble lazo que me permite obtener la DFT. Hay que tener 
    # en cuenta que por cada BIN debo hacer la sumatoria de todas las
    # muestras afectadas por la base ortonormal
    for k in range(0, N):
        for n in range(0, N):
            # np.complex(real, imaginario)
            dft_output[k] += (xx[n] * np.exp(-2j*np.pi*k*n/(len(xx))))
    # Un dato importante es que el resultado está fuera de escala
    # Si se quisiera escalar al mismo, se debe devidir a cada uno de los 
    # elementos del vector por la cantidad de muestras (N en este caso)
    return dft_output

def sin_gen(vmax = 1, dc = 0, ff = 1, ph = 0, nn = 1000, fs = 1000):
    # Lo primero que hay que hacer es crear un vector temporal que determinará
    # el inicio y el fin de la señal
    ts = 1/fs   # Tiempos de muestreo, intervalo de tiempo en el que se  
                # tomarán las muestras
    tf = N * ts # Tiempo final de la señal
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

# Se definen la frecuencia de muestro y la cantidad de muestras. 
# En este caso tendremos estaremos trabajando de forma normalizada, por
# lo que N = fs

fs = 500     # Frecuencia de muestreo
N = fs      # Cantidad de muestras

fig1, (ax1, ax2, ax3) = plt.subplots(3);

# Lo primero que voy a hacer es generarme la señal senoidal 
# que utilizaré para probar mi DFT
# sin_gen(Amp, DC, Frec, Phase, Muestras, Fs)
t, y = sin_gen(10, 0, 10, 0, N, fs)
ax1.plot(y, color = 'green', label = 'Función Senoidal')
ax1.legend(loc = 'upper right');

# Calculo la DFT utilizando mi función, luego voy a plotearla en 
# recuadro superior del gráfico
XX = my_dft(y)
ax2.plot(np.abs(XX), color = 'blue', label = 'DFT')
ax2.legend(loc = 'upper right');

# A modo de verificación, voy a realizar la FFT que posee la biblioteca
# numpy e imprimir el resultado en el recuadro inferior
XX = np.fft.fft(y);
ax3.plot(np.abs(XX), color = 'red', label = 'FFT')
ax3.legend(loc = 'upper right');

# Por último, voy a hacer exactamente la misma prueba pero con una señal
# aleatoria de varianza = 4
# Para ello, debemos obtener una función aleatoria que cumpla dicha
# condición

# Defino la varizanza que tendrá mi señal aleatoria
var = 4;
rnd_numbers = rnd_nmbrs_gen(var, N);

fig2, (ax1, ax2, ax3) = plt.subplots(3);
fig2.suptitle('Señal Aleatoria');
# Verifico haber obtenido la secuencia de números aleatorios
ax1.plot(rnd_numbers, label = 'Función Aleatoria');
ax1.legend(loc = 'upper right');

# Calculo la DFT utilizando mi función, luego voy a plotearla en 
# recuadro superior del gráfico
XX = my_dft(rnd_numbers)
ax2.plot(np.abs(XX), color = 'blue', label = 'DFT');
ax2.legend(loc = 'upper right')

# A modo de verificación, voy a realizar la FFT que posee la biblioteca
# numpy e imprimir el resultado en el recuadro inferior
XX = np.fft.fft(rnd_numbers);
ax3.plot(np.abs(XX), color = 'red', label = 'FFT')
ax3.legend(loc = 'upper right')
