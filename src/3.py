import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from time import sleep
from IPython import display

t = 1000
p = np.linspace(0,1,t)
fy = stats.beta.pdf(p, a=1, b=1)

DIGITS_CSV_FILE_PATH = "../given-files/digits.csv"

df = pd.read_csv(DIGITS_CSV_FILE_PATH)

r201 = df.iloc[200]
r202 = df.iloc[201]
df.drop(200, inplace=True)
df.drop(201, inplace=True)

df.loc[:, df.columns != 'label'] = df.loc[:, df.columns != 'label'].map(lambda x: 0 if x < 128 else 1)

chosen_row = df.iloc[87].values[1:].reshape(28, 28)

plt.imshow(chosen_row)
plt.show()

def update(fy: np.array, n: bool) -> np.array:
    p = np.linspace(0,1,t)
    pny = p if n else 1 - p
    integral = 0
    for i in range(t):
        integral += fy[i] * pny[i] / 1000
    post = fy * pny / integral
    return post

plt.figure(figsize=(10,8))
for i in range(100):
    n =  df[df['label'] == 8].iloc[i, df.columns.get_loc('pixel404')]
    fy = update(fy, n)

    plt.plot(p, fy, 'r', label='1')
    plt.ylim(-1, 10)
    plt.xlim(0, 1)
    plt.text(0.1,9,f'number of seen data : {i + 1}, p = {fy.argmax() / t :.2f}', color='purple')
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    sleep(0.05)
