#Zadanie 1A
import numpy as np
import matplotlib.pyplot as plt


A = 230
f = 50
f1 = 10000
f2 = 500
f3 = 200
t_g = 0.1
time1 = np.linspace(0,t_g, int(t_g * f1), endpoint=False)
time2 = np.linspace(0,t_g, int(t_g * f2), endpoint=False)
time3 = np.linspace(0,t_g, int(t_g * f3), endpoint=False)

y1 = A * np.sin(2*np.pi*f*time1)
y2 = A * np.sin(2*np.pi*f*time2)
y3 = A * np.sin(2*np.pi*f*time3)

plt.plot(time1,y1,'-b')
plt.plot(time2,y2,'r-o')
plt.plot(time3,y3,'k-x')

# zadanie 1B
A= 230
f=50
f1 =10000
f2 = 26
f3 = 25
f4 = 24

t_g = 1
time1 = np.linspace(0,t_g, int(t_g * f1), endpoint=False)
time2 = np.linspace(0,t_g, int(t_g * f2), endpoint=False)
time3 = np.linspace(0,t_g, int(t_g * f3), endpoint=False)
time4 = np.linspace(0,t_g, int(t_g * f4), endpoint=False)


y1 = A * np.sin(2*np.pi*f*time1)
y2 = A * np.sin(2*np.pi*f*time2)
y3 = A * np.sin(2*np.pi*f*time3)
y4 = A * np.sin(2*np.pi*f*time4)

plt.plot(time1,y1,'-b')
plt.plot(time2,y2,'g-o')
plt.plot(time3,y3,'r-o')
plt.plot(time4,y4,'k-o')

# zadanie 1C
import numpy as np
import matplotlib.pyplot as plt

fs = 100
T = 1
A = 1

t = np.arange(0, T, 1/fs)

for f in range(0, 301, 5):
    y = A * np.sin(2 * np.pi * f * t)
    plt.figure(figsize=(10, 2))
    plt.plot(t, y)
    plt.title(f'Sinusoida o częstotliwości {f} Hz (Obieg {f//5 + 1})')
    plt.xlabel('Czas [s]')
    plt.ylabel('Amplituda')
    plt.grid(True)
    plt.show()

def compare_frequencies(frequencies, signal_type='sin'):
    t = np.arange(0, T, 1/fs)
    plt.figure(figsize=(10, 6))
    for f in frequencies:
        if signal_type == 'sin':
            y = A * np.sin(2 * np.pi * f * t)
        else:
            y = A * np.cos(2 * np.pi * f * t)
        plt.plot(t, y, label=f'{f} Hz')
    plt.title(f'Porównanie {signal_type}usoid dla częstotliwości: {", ".join(map(str, frequencies))}')
    plt.xlabel('Czas [s]')
    plt.ylabel('Amplituda')
    plt.legend()
    plt.grid(True)
    plt.show()

frequencies_to_compare = [[5, 105, 205], [95, 195, 295], [95, 105]]

for frequencies in frequencies_to_compare:
    compare_frequencies(frequencies, 'sin')

for frequencies in frequencies_to_compare:
    compare_frequencies(frequencies, 'cos')

#Zadanie 2
import numpy as np
import matplotlib.pyplot as plt

fs2 = 200
fs1 = 10000
A = 230

dt2 = 1/fs2
dt1 = 1/fs1

time_fs2 = np.arange(0, 0.1, dt2)
time_fs1 = np.arange(0, 0.1, dt1)

y_fs2 = A * np.sin(2 * np.pi * 50 * time_fs2)
y_fs1 = A * np.sin(2 * np.pi * 50 * time_fs1)

y_out = np.zeros(len(time_fs1))

#Używamy wzoru z zadania
for i in range(len(time_fs1)):
    val = 0
    for n in range(len(time_fs2)):
        T1 = 1/fs2
        T2 = 1/fs1
        t = i * T2
        nT = n * T1
        y = np.pi/T1 * (t - nT)
        sinc = 1
        if y != 0:
            sinc = np.sin(y)/y
        val = val + y_fs2[n] * sinc
    y_out[i] = val

plt.figure()
plt.grid(True)
plt.plot(time_fs1, y_fs1, 'b-')
plt.plot(time_fs1, y_out, 'r-')
plt.xlabel('Czas (s)')
plt.ylabel('Amplituda (V)')
plt.title('Rekonstrukcja sygnału')
plt.legend(['Sygnał oryginalny', 'Sygnał zrekonstruowany'])

errors = np.abs(y_fs1 - y_out)
plt.figure()
plt.grid(True)
plt.plot(time_fs1, errors, 'k-')
plt.xlabel('Czas [s]')
plt.ylabel('Amplituda [V]')
plt.title('Błędy rekonstrukcji sygnału sin(x)/x')
plt.legend(['Błędy rekonstrukcji'])

plt.show()

#Zadanie 3
import numpy as np
import matplotlib.pyplot as plt
import scipy.io


def mycorr(x, y):
    # Obliczanie korelacji wzajemnej dwóch wektorów
    # Korelację oblicza się jako sumę wartości obu sygnałów przemnożonych przez siebie
    # x, y - wektory wejściowe
    # r - wektor wynikowy korelacji wzajemnej
    # lags - opóźnienia dla każdej wartości w r
    n = len(x)
    m = len(y)
    # obliczenie średnich wartości obu wektorów
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    # obliczenie odchyleń standardowych obu wektorów
    x_std = np.std(x)
    y_std = np.std(y)
    r = np.zeros(n + m - 1)
    lags = np.arange(-n + 1, m)

    # wyliczenie korelacji wzajemnej
    for i in range(len(r)):
        if lags[i] < 0:
            r[i] = np.sum((x[0:n + lags[i]] - x_mean) * (y[-lags[i]:m] - y_mean))
        elif lags[i] == 0:
            r[i] = np.sum((x - x_mean) * (y - y_mean))
        else:
            r[i] = np.sum((x[lags[i]:n] - x_mean) * (y[0:m - lags[i]] - y_mean))

        r[i] = r[i] / (x_std * y_std * (n - abs(lags[i])))
    return r

data = scipy.io.loadmat('adsl_x.mat')
x = np.array(data['x'])
prefix_len = 32  # długość prefiksu
frame_len = 512  # długość ramki
package_len = prefix_len + frame_len # długość ramki i prefiksu


max_corr = 0  # początkowa wartość maksymalnej korelacji
st_prefix_probe = np.zeros((3, 1))

for i in range(len(x) // 3):  # pętla po całym sygnale
    if (i + 3 * package_len) > len(x):
        break
        #

    max_corr_grup = 0
    tmp_prefix_probe = np.zeros((3, 1))

    for j in range(3):
      # według wykładu pakiety mają występować idealnie po sobie dlatego:
      # trzeba znaleść takie miejsca dla których suma 3 korelancji będzie największa
        prefix = x[i + j * package_len: i + j * package_len + prefix_len]  # prefiks
        tmp_prefix_probe[j, 0] = i + j * package_len

        copy_probe_block = x[i + j * package_len + frame_len: i + j * package_len + frame_len + prefix_len]
        # korelacja między pierwszymi 32 próbkami a ostatnimi 32 próbkami w oknie
        corr = mycorr(prefix, copy_probe_block)

        max_corr_grup += np.mean(corr)

    if max_corr_grup > max_corr:  # jeśli korelacja jest większa niż dotychczasowa maksymalna
        max_corr = max_corr_grup  # aktualizacja maksymalnej korelacji
        st_prefix_probe = tmp_prefix_probe  # początek prefiksu

plt.plot(x)
for i in range(3):
    plt.plot(st_prefix_probe[i, 0], 0, 'rx')
plt.legend(['signal'])
plt.show()
