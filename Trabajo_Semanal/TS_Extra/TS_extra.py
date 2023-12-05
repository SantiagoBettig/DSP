# El obejtivo de la siguiente tarea será poner en práctica los conocimientos
# obtenidos durante la cursada, aprovechando una señal de PPG que obtuve 
# para realizar mi proyecto final. 
# Lo que haré será muy similar a la realizado en la TS5 y TS6, permitiendome
# sacar algunas conclusiones

import csv,os
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as sig
from scipy import interpolate as interp

with open('MAX30102_data_full.csv', 'r') as csv_file:
    reader = csv.reader(csv_file);
    ppg_vec = np.array([]) 
    for row in reader:
        ppg_vec = np.append(ppg_vec, float(row[0]))
plt.figure(1)
plt.plot(ppg_vec[19000:20000])
plt.title('Señal PPG')
plt.xlabel('Tiempo [t]')  
plt.ylabel('Amplitud')

fs = 400;

# Obtengo la PSD del vector a través de Welch
len_ppg    = len(ppg_vec);
welch_ppg  = np.empty((len_ppg, 1), float);
f_norm, welch_ppg = sig.welch(ppg_vec, fs, 'blackman', nfft = 8192, return_onesided=True)
f_filt = f_norm < 10;
plt.figure(2)
plt.plot(f_norm[f_filt], welch_ppg[f_filt], 'x-g', alpha=0.5)
plt.title('PSD')
plt.xlabel('Tiempo [t]')  
plt.ylabel('Amplitud')  

# Ahora voy a buscar el ancho de banda útil de mi señal.
# Primero que nada, tengo que saber cual es la potencia total:
total_energy = np.sum(welch_ppg);
cumsum_energy = np.cumsum(welch_ppg);
# Voy a recorrer el vector cumsum hasta obtener un porcentaje de energía
bw_threshold = 0.98;
for i in range(len(cumsum_energy)):
    if (cumsum_energy[i] >= bw_threshold*total_energy):
        bw_norm_i = i;
        break;
bw_norm = f_norm[bw_norm_i];
print("Pulsos Normales      -> BW: ", bw_norm, "Hz");

# Voy a empezar calculando el filtro pasa altos. Para ellos, establezco 
# que la frecuencia de corte sea de 0.15Hz y que sea de orden 4
f_cut = 0.15
N = 4
Wn = f_cut
sos_matrix_hpf = sig.butter(N, Wn, 'highpass', False, 'sos', fs)

# Voy a seguir filtro pasa bajos. Para ellos, establezco 
# que la frecuencia de corte sea de 7.5Hz y que sea de orden 4
f_cut = 7.5
N = 4
Wn = f_cut
sos_matrix_lpf = sig.butter(N, Wn, 'lowpass', False, 'sos', fs)

ppg_filt_1_lpf = sig.sosfilt(sos_matrix_lpf, ppg_vec);
ppg_filt_1     = sig.sosfilt(sos_matrix_hpf, ppg_filt_1_lpf);

ppg_t = np.arange(0, len(ppg_vec), 1);
time_filt_norm  = (ppg_t > 12600) & (ppg_t < 15000);
time_filt_lp    = (ppg_t > 21000) & (ppg_t < 24000);

fig3, (ax1, ax2) = plt.subplots(2);
ax1.set_title('Filtro LPF y HPF IIR')
ax1.plot(ppg_vec[time_filt_norm]-np.mean(ppg_vec[time_filt_norm]),      'b', label = 'Original')
ax1.plot(ppg_filt_1[time_filt_norm],        'r', label = 'FIR')
ax1.set(xlabel = 'Tiempo [t]', ylabel = 'Amplitud')
ax1.legend(loc = 'upper right')
ax2.plot(ppg_vec[time_filt_lp]-np.mean(ppg_vec[time_filt_lp]),      'b', label = 'Original')
ax2.plot(ppg_filt_1[time_filt_lp],        'r', label = 'FIR')
ax2.set(xlabel = 'Tiempo [t]', ylabel = 'Amplitud')

ws1         = 0.03 #Hz
wp1         = 0.15  #Hz
wp2         = 7.5 #Hz
ws2         = 20.0 #Hz
ripple      = 0.5 # dB
atenuacion  = 40  # dB

frecs    = np.array([0, ws1, wp1, wp2, ws2, (fs//2)])
gains    = np.array([-atenuacion, -atenuacion, -ripple, -ripple, -atenuacion, 0])
gains    = 10**(gains/20)
gains[5] = 0

num_win    = sig.firwin2(10001, frecs, gains , window=('kaiser', 14), fs = fs)
ppg_filt_3 = sig.filtfilt(num_win, 1, ppg_vec, axis = 0)

fig4, (ax1, ax2) = plt.subplots(2);
ax1.set_title('Filtro LPF y HPF IIR')
ax1.plot(ppg_vec[time_filt_norm]-np.mean(ppg_vec[time_filt_norm]),      'b', label = 'Original')
ax1.plot(ppg_filt_3[time_filt_norm],        'r', label = 'LPF y HPF')
ax1.set(xlabel = 'Tiempo [t]', ylabel = 'Amplitud')
ax1.legend(loc = 'upper right')
ax2.plot(ppg_vec[time_filt_lp]-np.mean(ppg_vec[time_filt_lp]),      'b', label = 'Original')
ax2.plot(ppg_filt_3[time_filt_lp],        'r', label = 'LPF y HPF')
ax2.set(xlabel = 'Tiempo [t]', ylabel = 'Amplitud')
        
max_peaks = sig.find_peaks(ppg_vec, distance = 200)
min_peaks = sig.find_peaks(ppg_vec**2, distance = 200)
isoelectric_points = np.empty((1, len(max_peaks[0])), float);
isoelectric_grid = np.empty((1, len(max_peaks[0])), float);

for i in range(len(max_peaks[0])):
    max_value = max_peaks[0][i]
    min_value = min_peaks[0][i]
    isoelectric_points[0][i] = (ppg_vec[max_value] + ppg_vec[min_value])/2
    isoelectric_grid[0][i] = (max_value + min_value)/2

time_grid = np.arange(0, len(isoelectric_points[0]), 1);
ppg_grid= np.arange(0, len(ppg_vec), 1);

isoelectric_interp = interp.CubicSpline(isoelectric_grid[0], isoelectric_points[0]);
interp_ppg = ppg_vec-isoelectric_interp(ppg_grid);

plt.figure(5)
plt.plot(ppg_vec,      'b',    label = 'Original');
plt.plot(isoelectric_interp(ppg_grid), label = 'Interpolante');
plt.xlabel('Tiempo [t]');
plt.ylabel('Amplitud');
plt.legend();

fig6, (ax1, ax2) = plt.subplots(2);
ax1.set_title('Filtro Alineal');
ax1.plot(ppg_vec[time_filt_norm]-np.mean(ppg_vec[time_filt_norm]),      'b', label = 'Original');
ax1.plot(interp_ppg[time_filt_norm],        'r', label = 'Filtrado Alineal');
ax1.set(xlabel = 'Tiempo [t]', ylabel = 'Amplitud');
ax1.legend(loc = 'upper right');
ax2.plot(ppg_vec[time_filt_lp]-np.mean(ppg_vec[time_filt_lp]),      'b', label = 'Original');
ax2.plot(interp_ppg[time_filt_lp],        'r', label = 'Filtrado Alineal');
ax2.set(xlabel = 'Tiempo [t]', ylabel = 'Amplitud');

# plt.figure(8)
# plt.plot(ppg_vec,      'b',    label = 'Original')
# plt.plot(ppg_filt_3,   'g',    label = 'Firwin Bidir')
# plt.plot(ppg_filt_1,    'r',    label = 'IIR')
# plt.plot(isoelectric_interp(ppg_grid), label = 'Interpolate')
# plt.plot(ppg_vec-isoelectric_interp(ppg_grid), label = 'Isoelectric Null')
# plt.title('Post Filtrado')
# plt.xlabel('Tiempo [t]')  
# plt.ylabel('Amplitud')  
# plt.legend()