import numpy as np
import scipy.stats as stats
import pandas as pd
import matplotlib.pyplot as plt 

green = '#57cc99'
red = '#e56b6f'
blue = '#22577a'
yellow = '#ffca3a'

bar_width = 0.5

TARBIAT_CSV_FILE_PATH = "../given-files/Tarbiat.csv"

data = pd.read_csv(TARBIAT_CSV_FILE_PATH)

fig, (ax1, ax2) = plt.subplots(1, 2)

x = data["metro"]
y = data["BRT"]

ax1.hist(x, bins='auto', alpha=0.7, label='Metro', color=red, edgecolor='black', width=bar_width)
ax1.set_title('Metro')
ax1.grid(True)
ax1.legend()
ax1.set_xlabel('Number of arrives')
ax1.set_ylabel('Freq')

ax2.hist(y, bins='auto', alpha=0.7, label='BRT', color=yellow, edgecolor='black', width=bar_width)
ax2.set_title('BRT')
ax2.grid(True)
ax2.legend()
ax2.set_xlabel('Number of arrives')
ax2.set_ylabel('Freq')

plt.show()

fig, (ax1, ax2) = plt.subplots(1, 2)

counts1, bins1, _ = plt.hist(x, bins='auto')
counts2, bins2, _ = plt.hist(y, bins='auto')

plt.cla()

x_values = np.arange(0, 10)

x_mean = np.mean(x)
y_mean = np.mean(y)

poisson_x = stats.poisson(x_mean)
pmf_x = poisson_x.pmf(x_values)
poisson_y = stats.poisson(y_mean)
pmf_y = poisson_y.pmf(x_values)

ax1.bar(x_values + bar_width, pmf_x, color=blue, label='X ~ Pois({:.2f})'.format(x_mean), alpha=0.7, width=bar_width)
ax1.bar(bins1[:-1], counts1 / len(x), width=bar_width, alpha=0.7, label='Metro', color=red, edgecolor='black')
ax1.set_title('Metro')
ax1.grid(True)
ax1.legend()

ax2.bar(x_values + bar_width, pmf_y, color=blue, label='Y ~ Pois({:.2f})'.format(y_mean), alpha=0.7, width=bar_width)
ax2.bar(bins2[:-1], counts2 / len(y), width=bar_width, alpha=0.7, label='BRT', color=yellow, edgecolor='black')
ax2.set_title('BRT')
ax2.grid(True)
ax2.legend()

plt.show()

fig, ax = plt.subplots()

z = x + y

counts3, bins3, _ = plt.hist(z, bins='auto')

plt.cla()

poisson_z = stats.poisson(x_mean + y_mean)
pmf_z = poisson_z.pmf(x_values)

ax.bar(x_values + bar_width, pmf_z, color=blue, label='Z ~ Pois({:.2f})'.format(x_mean + y_mean), width=bar_width, alpha=0.7)
ax.bar(bins3[:-1], counts3 / len(z), width=bar_width, alpha=0.7, label='BRT + Metro', color=green, edgecolor='black')
ax.set_title('BRT + Metro')
ax.grid(True)
ax.legend()

plt.show()

fig, ax = plt.subplots()

p = x_mean / (x_mean + y_mean)
n = 8

w_counts = [0] * (9)
x_values = np.arange(len(w_counts))
pmf_w = stats.binom.pmf(x_values, n, p)

for _, row in data.iterrows():
    if row['metro'] + row['BRT'] == n:
        w_counts[row['metro']] += 1

count = np.count_nonzero((data['metro'] + data['BRT'] == n))
w_counts = [x / count for x in w_counts]

ax.bar(x_values + bar_width, w_counts, color=red, label='Conditional probability in action', alpha=0.7, edgecolor='black', width=bar_width)
ax.bar(x_values, pmf_w, color=blue, label='W ~ Bin(8, {:.2f})'.format(p), alpha=0.7, width=bar_width, edgecolor='black')

ax.set_title('Conditional probability')
ax.grid(True)
ax.legend()

plt.show()
