import numpy as np
import matplotlib.pyplot as plt

p12 = -0.5 + 9.5j
p34 = -1 + 10j
p56 = -0.5 + 10.5j
z12 = 5j
z34 = 15j

# Duplikowanie biegunow i zer
p = [p12, p34, p56, np.conj(p12), np.conj(p34), np.conj(p56)]
z = [z12, z34, np.conj(z12), np.conj(z34)]

# Wspolczynnik wzmocnienia
wzm = 0.426

# Plotowanie biegunow oraz zer
plt.figure(figsize=(8, 6))
plt.plot(np.real(p), np.imag(p), "o", label="Poles")
plt.plot(np.real(z), np.imag(z), "x", label="Zeros")
plt.grid(True)
plt.axis('equal')
plt.xlabel("Re(z)")
plt.ylabel("Im(z)")
plt.title("Poles and Zeros")
plt.legend()

# Obliczenie wspolczynnikow wielomianowych
a = np.poly(p)
b = np.poly(z) * wzm

# Odpowiedz liniowa czestotliwościowa
w = np.arange(4, 16.1, 0.1)
s = w * 1j
Hlinear = np.abs(np.polyval(b, s) / np.polyval(a, s))

# Plotowanie wykresu liniowej odpowiedzi czestotliwosciowej
plt.figure()
plt.plot(w, Hlinear)
plt.xlabel("Frequency [rad/s]")
plt.ylabel("|H(jw)|")
plt.title("Linear Frequency Response")

# Odpowiedz w skali logarytmicznej
Hlog = 20 * np.log10(Hlinear)
plt.figure()
plt.semilogx(w, Hlog, 'r')
plt.xlabel("Frequency [rad/s]")
plt.ylabel("20log10(|H(jw)|)")
plt.title("Logarithmic Frequency Response")

# odpowiedz fazowa
H_phase = np.angle(np.polyval(b, s) / np.polyval(a, s))

#Plotowanie odpowiedzi fazowej
plt.figure()
plt.plot(w, H_phase, 'g')
plt.xlabel('Frequency [rad/s]')
plt.ylabel('Angle [rad]')
plt.title('Phase-Frequency Characteristic')
plt.grid(True)

plt.show()