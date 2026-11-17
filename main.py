import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt

dset = np.loadtxt('decay_data.csv', delimiter=',', skiprows=1, dtype=int)
base = dset[0:49, 0]
caesium = dset[0:49, 1]
barium = dset[:, 2]
btime = np.zeros(len(barium))
for i in range(len(barium)):
    btime[i] = 10*i

# Part 1
bsm = np.mean(base)
bss = np.std(base)
print(bsm,bss)

base_edges = np.arange(base.min() - 0.5, base.max() + 1.5, 1)
plt.figure(figsize=(10, 6))
plt.hist(base, bins=base_edges, alpha=0.7, label='Base', color='blue')
plt.xlabel('Number of decay events')
plt.ylabel('Number of occurrences')
plt.xticks(np.arange(base.min(), base.max() + 1, 1))
plt.title('Baseline Radiation Level')
plt.savefig('base.png', dpi=300)

# Part 2
csm = np.mean(caesium)
css = np.std(caesium)
print(csm,css)

caesium_edges = np.arange(caesium.min() - 0.5, caesium.max() + 1.5, 1)
plt.figure(figsize=(10, 6))
plt.hist(caesium, bins=caesium_edges, alpha=0.7, label='Caesium-137', color='green')
plt.xlabel('Number of decay events')
plt.xticks(np.arange(caesium.min(), caesium.max() + 1, 1))
plt.ylabel('Number of occurrences')
plt.title('Caesium-137 Decay')
plt.savefig('caesium.png', dpi=300)

# Part 3
def decay(t, N0, dp):
    return N0 * np.exp(-dp * t)

popt, pcov = opt.curve_fit(decay, btime, barium, p0=[np.max(barium), 0.1])
dp = popt[1]
delp = np.sqrt(np.diag(pcov))[1]
print(dp, delp)

thalf = np.log(2)/dp
dhalf = (np.log(2)/dp**2)*delp
print(thalf, dhalf)

trand = np.linspace(0, np.max(btime), 1000)

plt.figure(figsize=(10, 6))
plt.scatter(btime, barium, label='Barium-137', color='red')
plt.plot(trand, decay(trand, *popt), label='Fitted Curve', color='black')
plt.xlabel('Time (s)')
plt.ylabel('Decay events')
plt.title('Barium-137 Decay')
plt.savefig('barium.png', dpi=300)