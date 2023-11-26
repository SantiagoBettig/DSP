import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as sig
from scipy import interpolate as interp

def vertical_flaten(a):
  return a.reshape(a.shape[0],1)

def group_delay( freq, phase):    
    groupDelay = -np.diff(phase)/np.diff(freq)
    return(np.append(groupDelay, groupDelay[-1]))

def filter_design(ripple = 0, att = 40, ws1 = 0.1, wp1 = 1.0, wp2 = 40.0, ws2 = 50.0, fs = 1000):
    return sig.iirdesign(wp=np.array([wp1, wp2]) / (fs/2), ws=np.array([ws1, ws2]) / (fs/2), gpass=ripple, gstop=att, analog=False, ftype='cheby1', output='sos')

# Para listar las variables que hay en el archivo
mat_struct = sio.loadmat('./ECG_TP4.mat')
fs = 1000

# Me quedo con todas las muestras de ECG
ecg_one_lead = vertical_flaten(mat_struct['ecg_lead'])

# Profundidad que tendrá cada una de las muestras por separado. Desde el punto
# máximo me moveré N1 muestras para atrás y N2 muestras hacia adelante
N = 600;
N1 = 270;
N2 = N - N1

# Obtengo todos los puntos donde tengo los máximos relativos y la longitud
# de dicho vector (Cantidad de picos)
qrs_peak = vertical_flaten(mat_struct['qrs_detections'])
n_run = len(qrs_peak)

# Creo unos arrays vacios donde voy a guardar todas las muestras
ecg_samples         = np.empty((n_run, N), float);
ecg_samples_detrend = np.empty((n_run, N), float);
ecg_norm            = np.empty((n_run, N), float);

# Empiezo a recorrer el vector hasta encontrar uno de estos picos, luego
# guardo N1 muestras hacia atras y N2 muestras hacia adelante
# Con detrend elimino la tendencia que tenga mi señal
for i in range(n_run):
    index = qrs_peak[i][0];
    ecg_samples[i] = np.reshape(ecg_one_lead[index-N1:index+N2], [N,])
    ecg_samples_detrend[i] = sig.detrend(ecg_samples[i])

# Ploteo los resultados obtenidos
plt.figure(2)
plt.plot(np.transpose(ecg_samples_detrend[0]), 'b', alpha=0.5)
plt.plot(np.transpose(ecg_samples_detrend), 'b', alpha=0.5)
plt.title('Pulsos')
plt.xlabel('Tiempo [t]')  
plt.ylabel('Amplitud')  

# Voy a separar entre latidos normales y ventriculares. Para ello, primero
# voy a buscar el valor máximo para normalizar las señales:
max_value = 0.0;
max_aux = 0.0;
for i in range(n_run):
    max_aux = np.max(ecg_samples_detrend[i])
    if(max_value < max_aux):
        max_value = max_aux;
for i in range(n_run):
    ecg_norm[i] = ecg_samples_detrend[i]/max_value;

# Para separar entre normales y ventriculares lo que voy a hacer es 
# determinar cuales de los pulsos superan cierto umbral(ventriculares).
# Los que no lo superan, seran normales
peak_threshold = 0.5;
pulse_norm = np.array([]);
pulse_vent = np.array([]);
for i in range(n_run):
    if (np.max(ecg_norm[i]) < peak_threshold):
        pulse_norm = np.append(pulse_norm, ecg_norm[i], axis = 0);
    else:
        pulse_vent = np.append(pulse_vent, ecg_norm[i], axis = 0);
pulse_norm = np.reshape(pulse_norm, ((len(pulse_norm)//N), N));
pulse_vent = np.reshape(pulse_vent, ((len(pulse_vent)//N), N));
# Ploteo los resultados obtenidos
plt.figure(3)
plt.plot(np.transpose(pulse_norm[0]), 'g', alpha=0.5, label = 'Normales')
plt.plot(np.transpose(pulse_vent[0]), 'b', alpha=0.5, label = 'Ventriculares')
plt.plot(np.transpose(pulse_norm), 'g', alpha=0.5)
plt.plot(np.transpose(pulse_vent), 'b', alpha=0.5)
plt.title('Superposición de Pulsos')
plt.xlabel('Tiempo [t]')  
plt.ylabel('Amplitud Normalizada')  
plt.legend()

# Luego, voy a promediar todos los pulsos normales y los ventriculares entre
# si para obtener un pulso promedio, con el que trabajaré a partir de este 
# momento. El mismo me servirá para poder conocer el contenido espectral de
# la señal
pulse_norm_avg = np.average(np.transpose(pulse_norm), axis = 1)
pulse_vent_avg = np.average(np.transpose(pulse_vent), axis = 1)
plt.figure(4)
plt.plot(np.transpose(pulse_norm_avg), 'g', alpha=0.5, label = 'Normal')
plt.plot(np.transpose(pulse_vent_avg), 'b', alpha=0.5, label = 'Ventricul')
plt.title('Promedio de Señales')
plt.xlabel('Tiempo [t]')  
plt.ylabel('Amplitud Normalizada')  
plt.legend()

# Obtengo la PSD de ambos vectores a través de Welch
len_norm    = len(pulse_norm);
len_vent    = len(pulse_vent);
welch_norm  = np.empty((len_norm, 2049), float);
welch_vent  = np.empty((len_vent, 2049), float);
for i in range(len_norm):
    f_norm, welch_norm[i] = sig.welch(pulse_norm[i], fs, 'blackman', N, nfft = 4096, return_onesided=True)
for i in range(len_vent):
    f_vent, welch_vent[i] = sig.welch(pulse_vent[i], fs, 'blackman', N, nfft = 4096, return_onesided=True)
welch_norm_avg = np.average(np.transpose(welch_norm), axis = 1)
welch_vent_avg = np.average(np.transpose(welch_vent), axis = 1)
frec_filter = f_norm < 45
plt.figure(5)
plt.plot(f_norm[frec_filter], welch_norm_avg[frec_filter]/np.max(welch_norm_avg[frec_filter]), 'x-g', alpha=0.5, label = 'Normal')
plt.plot(f_vent[frec_filter], welch_vent_avg[frec_filter]/np.max(welch_vent_avg[frec_filter]), 'x-b', alpha=0.5, label = 'Ventricul')
plt.title('PSD')
plt.xlabel('Frecuencia [Hz]')  
plt.ylabel('Amplitud Normalizada')  
plt.legend()

# Ahora voy a buscar el ancho de banda útil de mi señal.
# Primero que nada, tengo que saber cual es la potencia total:
total_energy_norm = np.sum(welch_norm_avg);
total_energy_vent = np.sum(welch_vent_avg);
cumsum_energy_norm = np.cumsum(welch_norm_avg);
cumsum_energy_vent = np.cumsum(welch_vent_avg);
# Voy a recorrer el vector cumsum hasta obtener un porcentaje de energía
bw_threshold = 0.98;
for i in range(len(cumsum_energy_norm)):
    if (cumsum_energy_norm[i] >= bw_threshold*total_energy_norm):
        bw_norm_i = i;
        break;
bw_norm = f_norm[bw_norm_i];
for i in range(len(cumsum_energy_vent)):
    if (cumsum_energy_vent[i] >= bw_threshold*total_energy_vent):
        bw_vent_i = i;
        break;
bw_vent = f_vent[bw_vent_i];
print("Pulsos Normales      -> BW: ", bw_norm, "Hz");
print("Pulsos Ventriculares -> BW: ", bw_vent, "Hz")

# Frecuencias de muestreo
f_cut_hpf   = 0.3
f_cut_lpf   = 35
N           = 4
   
sos_matrix_hpf = sig.butter(N = N, Wn = f_cut_hpf, btype = 'highpass', 
                            analog = False, fs = fs, output = 'sos')
   
sos_matrix_lpf = sig.butter(N = N, Wn = f_cut_lpf, btype = 'lowpass', 
                            analog = False, fs = fs, output = 'sos')

ecg_filt_hpf_1      = sig.sosfilt(sos_matrix_hpf, ecg_one_lead, axis = 0);
ecg_filt_filt_hpf_1 = sig.sosfiltfilt(sos_matrix_hpf, ecg_one_lead, axis = 0);
ecg_filt_1          = sig.sosfilt(sos_matrix_lpf, ecg_filt_hpf_1, axis = 0);
ecg_filt_filt_1     = sig.sosfiltfilt(sos_matrix_lpf, ecg_filt_filt_hpf_1, axis = 0);

ecg_time        = np.arange(0, len(ecg_one_lead), 1);
time_filt_norm  = (ecg_time > 4234) & (ecg_time < 5867);
time_filt_lp    = (ecg_time > 100000) & (ecg_time < 120000);

fig6, (ax1, ax2) = plt.subplots(2);
ax1.set_title('Filtro LPF y HPF IIR')
ax1.plot(ecg_one_lead[time_filt_norm],      'b', label = 'Original')
ax1.plot(ecg_filt_1[time_filt_norm],        'r', label = 'LPF y HPF')
ax1.plot(ecg_filt_filt_1[time_filt_norm],   'g', label = 'LPF y HPF Bidir')
ax1.set(xlabel = 'Tiempo [t]', ylabel = 'Amplitud')
ax1.legend(loc = 'upper right')
ax2.plot(ecg_one_lead[time_filt_lp],      'b', label = 'Original')
ax2.plot(ecg_filt_1[time_filt_lp],        'r', label = 'LPF y HPF')
ax2.plot(ecg_filt_filt_1[time_filt_lp],   'g', label = 'LPF y HPF Bidir')
ax2.set(xlabel = 'Tiempo [t]', ylabel = 'Amplitud')

ripple      = 0.5 # dB
atenuacion  = 40  # dB

ws1 = 0.03 #Hz
wp1 = 0.3  #Hz
wp2 = 35.0 #Hz
ws2 = 50.0 #Hz
    
frecs    = np.array([0, ws1, wp1, wp2, ws2, (fs//2)])
gains    = np.array([-atenuacion, -atenuacion, -ripple, -ripple, -atenuacion, 0])
gains    = 10**(gains/20)
gains[5] = 0

sos_bpf         = filter_design(ripple, atenuacion, ws1, wp1, wp2, ws2, fs);
ecg_filt_2      = sig.sosfilt(sos_bpf, ecg_one_lead, axis = 0);
ecg_filt_filt_2 = sig.sosfiltfilt(sos_bpf, ecg_one_lead, axis = 0);

fig7, (ax1, ax2) = plt.subplots(2);
ax1.set_title('Filtro BPF IIR')
ax1.plot(ecg_one_lead[time_filt_norm],      'b', label = 'Original')
ax1.plot(ecg_filt_2[time_filt_norm],        'r', label = 'LPF y HPF')
ax1.plot(ecg_filt_filt_2[time_filt_norm],   'g', label = 'LPF y HPF Bidir')
ax1.set(xlabel = 'Tiempo [t]', ylabel = 'Amplitud')
ax1.legend(loc = 'upper right')
ax2.plot(ecg_one_lead[time_filt_lp],      'b', label = 'Original')
ax2.plot(ecg_filt_2[time_filt_lp],        'r', label = 'LPF y HPF')
ax2.plot(ecg_filt_filt_2[time_filt_lp],   'g', label = 'LPF y HPF Bidir')
ax2.set(xlabel = 'Tiempo [t]', ylabel = 'Amplitud')

num_win         = sig.firwin2(10001, frecs, gains , window=('kaiser', 14), fs = fs)
ecg_filt_filt_3 = sig.filtfilt(num_win, 1, ecg_one_lead, axis = 0)

fig8, (ax1, ax2) = plt.subplots(2);
ax1.set_title('Filtro FIR')
ax1.plot(ecg_one_lead[time_filt_norm],      'b', label = 'Original')
ax1.plot(ecg_filt_filt_3[time_filt_norm],   'g', label = 'LPF y HPF Bidir')
ax1.set(xlabel = 'Tiempo [t]', ylabel = 'Amplitud')
ax1.legend(loc = 'upper right')
ax2.plot(ecg_one_lead[time_filt_lp],      'b', label = 'Original')
ax2.plot(ecg_filt_filt_3[time_filt_lp],   'g', label = 'LPF y HPF Bidir')
ax2.set(xlabel = 'Tiempo [t]', ylabel = 'Amplitud')

# # num_firls_lp = sig.firls(cant_coef, np.append( [0.0], frecs[3:]), gains[2:], weight = np.array([5, 10]), fs=2)
# # num_remez_lp = sig.remez(cant_coef, np.append( [0.0], frecs[3:]), gains[3:5], weight = np.array([1, 5]), grid_density = 64, fs=2)

## Filtrado Alineal ##

# Filtrado por interpolación
ms_to_sample = 120.0; # Cantidad de ms que voy a retroceder desde el pico
n_samples_to_backward = (int)(ms_to_sample/1000.0 * fs); # Cantidad de muestras que retrocedo
isoelectric_sample = np.empty((n_run, 1), float);
isoelectric_sample_t = np.empty((n_run, 1), int);
for i in range(n_run):
    index = qrs_peak[i][0];
    isoelectric_sample_t[i] = index - n_samples_to_backward;
    isoelectric_sample[i] = ecg_one_lead[isoelectric_sample_t[i]]
time_grid = np.arange(0, len(ecg_one_lead), 1);
isoelectric_interp = interp.CubicSpline(isoelectric_sample_t.flatten(), isoelectric_sample);
isolectric_frame = isoelectric_interp(time_grid);

fig8, (ax1, ax2) = plt.subplots(2);
ax1.set_title('Interpolación')
ax1.plot(ecg_one_lead[time_filt_norm],      'b', label = 'Original')
ax1.plot(isolectric_frame[time_filt_norm],   'g', label = 'Interpolante')
ax1.plot(ecg_one_lead[time_filt_norm]-isolectric_frame[time_filt_norm],   'r', label = 'Resta')
ax1.set(xlabel = 'Tiempo [t]', ylabel = 'Amplitud')
ax1.legend(loc = 'upper right')
ax2.plot(ecg_one_lead[time_filt_lp],      'b', label = 'Original')
ax2.plot(isolectric_frame[time_filt_lp],   'g', label = 'Interpolante')
ax2.plot(ecg_one_lead[time_filt_lp]-isolectric_frame[time_filt_lp],   'r', label = 'Resta')
ax2.set(xlabel = 'Tiempo [t]', ylabel = 'Amplitud')

# Filtrado por mediana
kernel_200  = (int)(200/1000*fs)+1;
kernel_600  = (int)(600/1000*fs)+1;
ecg_frame = ecg_one_lead[:];
med_200     = sig.medfilt(ecg_frame.flatten(), kernel_200)
med_600     = sig.medfilt(med_200, kernel_600)
plt.plot(med_600, label = 'Mediana')
plt.plot(ecg_one_lead, label = 'Original')
median_ecg = np.transpose(ecg_one_lead)-med_600
plt.plot(np.transpose(median_ecg), label = 'Resta')
plt.legend()

fig9, (ax1, ax2) = plt.subplots(2);
ax1.set_title('Interpolación')
ax1.plot(ecg_one_lead[time_filt_norm],      'b', label = 'Original')
ax1.plot(med_600[time_filt_norm],   'g', label = 'Mediana')
ax1.plot(np.transpose(median_ecg)[time_filt_norm],   'r', label = 'Resta')
ax1.set(xlabel = 'Tiempo [t]', ylabel = 'Amplitud')
ax1.legend(loc = 'upper right')
ax2.plot(ecg_one_lead[time_filt_lp],      'b', label = 'Original')
ax2.plot(med_600[time_filt_lp],   'g', label = 'Interpolante')
ax2.plot(np.transpose(median_ecg)[time_filt_lp],   'r', label = 'Resta')
ax2.set(xlabel = 'Tiempo [t]', ylabel = 'Amplitud')
    
    
