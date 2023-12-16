import random
import sympy as sp
import matplotlib.pyplot as plt 
import numpy as np

green = '#57cc99'
red = '#e56b6f'
blue = '#22577a'
yellow = '#ffca3a'

line_thickness = 2

def monte_carlo(n):
    k = 1000
    ans = 0

    for _ in range(k):
        seen = 0
        trial = 0
        while seen != n:
            trial += 1
            if random.random() <= (n - seen) / n:
                seen += 1

        ans += trial

    return ans / k

def theorical_solution(n):
    n = int(n)
    expectation = 0

    s = sp.symbols('s')
    x = sp.symbols('x')

    for i in range(1, n + 1):
        p = (n - i + 1) / n
        gen_func = p * sp.exp(s) / (1 - (1 - p) * sp.exp(s))
        deriv = sp.diff(gen_func, s)
        expectation += deriv.subs({s:0})

    return expectation

dom = 20
x = np.linspace(1, dom, dom)

mc = [monte_carlo(n) for n in x]
ts = [theorical_solution(n) for n in x]

plt.plot(x, mc, 'o', label='Monte Carlo Approach', color=blue, alpha=0.5, markersize=10)
plt.plot(x, ts, 'o', label='Theorical Approach', color=red, alpha=0.5, markersize=10)

plt.xlabel('n')
plt.ylabel('Solution for n')
plt.title('Coupon collector’s problem')
plt.legend()
plt.grid(True)

plt.show()

dom = 60
x = np.linspace(1, dom, dom)
mc = [monte_carlo(n) for n in x]

plt.plot(x, mc, 'o', label='Monte Carlo Approach', color=blue, alpha=0.7)
plt.plot(x, x * np.log(x), label='nlog(n)', color=green, linewidth=line_thickness)

plt.xlabel('n')
plt.ylabel('Solution for n')
plt.title('Coupon collector’s problem')
plt.legend()
plt.grid(True)

plt.show()
