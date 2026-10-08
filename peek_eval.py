import numpy as np
x = np.load('models/checkpoints/evaluations.npz')
t = x['timesteps']
r = x['results'].mean(axis=1)
peak_i = r.argmax()
print('peak:', int(t[peak_i]), 'value:', round(float(r[peak_i]), 2))
print('final:', int(t[-1]), 'value:', round(float(r[-1]), 2))
print('last 10 evals:')
for i in range(max(0, len(t)-10), len(t)):
    print(' ', int(t[i]), round(float(r[i]), 2))