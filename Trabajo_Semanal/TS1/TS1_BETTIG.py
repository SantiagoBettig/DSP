#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# %matplotlib qt        -> Genera gráfico en ventana nueva
# %matplotlib inline    -> Gnera gráfico en el mismo spyder

"""
Created on Spyder3
@author: Santiago Bettig
Descripción:
    
    El objetivo de la tarea es realizar un generador de funciones senoidales
    que permite la parametrización de distintos parámetros, entre ellos: 
    la amplitud máxima de la senoidal, valor medio, frecuencia,  fase,
    cantidad de muestras digitalizada por el ADC y la
    frecuencia de muestreo del ADC.
    Una vez realizada esta primera etapa, la idea es proponer una nueva
    función que permita la generación de otro tipo de señal, en mi caso, 
    una señal del tipo triangular.
    
"""
# Se importan las bibliotecas para realizar operaciones matemáticas 
# no standard (numpy) y la biblioteca que permite imprimir gráficos en 
# pantalla (matplotlib)
import numpy as np
import matplotlib.pyplot as plt
# Para la segunda parte del trabajo, importaré la biblioteca signal, que me
# permitirá utilizar la función sawtooth para generar mi señal triangular
from scipy import signal as sg

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
    
def triangle_gen(vmax = 1, dc = 0, ff = 1, ph = 0, nn = 1000, fs = 1000, sl = 0.5):
    # Lo primero que hay que hacer es crear un vector temporal que determinará
    # el inicio y el fin de la señal
    ts = 1/fs   # Tiempos de muestreo, intervalo de tiempo en el que se  
                # tomarán las muestras
    tf = N * ts # Tiempo final de la señal
    tt = np.arange(0, tf, ts) # Creo el vector temporal
    # Ahora tengo que generar mi señal triangular, para eso voy a aprovechar
    # la función sg.sawtooth de la biblioteca signal. Para ello, debemo evaluar
    # dicha función en cada uno de los puntos del vector temporal creado
    # previamente
    xx = vmax * sg.sawtooth(2 * np.pi * ff * tt + ph, sl) + dc
    ##
    # Devuelvo el vector temporal y el valor de la señal en cada uno de esos
    # puntos
    return tt, xx

# Se definen la frecuencia de muestro y la cantidad de muestras. 
# En este caso tendremos estaremos trabajando de forma normalizada, por
# lo que N = fs

fs = 10000   # Frecuencia de muestreo
N = fs      # Cantidad de muestras

# Utilizo subplot para poder imprimir ambas señales en un mismo gráfico
# Posiciono la señal senoidal en el recuadro superior
plt.subplot(2, 1, 1)
t, y = sin_gen(10, 2, 10, 0, N, fs)
plt.plot(t, y, color = 'blue')
plt.ylabel("Amplitud[V]")

# Posiciono la señal triangular en el recuadro inferior
plt.subplot(2, 1, 2)
t, y = triangle_gen(10, 2, 10, 0, N, fs)
plt.plot(t, y, color = 'green')
plt.ylabel("Amplitud[V]")
plt.xlabel("Tiempo[N*ts]")