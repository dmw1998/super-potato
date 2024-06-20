from multilevel_estimator import *
from sus_sr import *

import time

# Set the number of samples
N = 100
M = 150
p0 = 0.1


start_time = time.time()
p_f = sus_sr(p0, N, M)
sr_time = time.time() - start_time
print('p_f =', p_f)
print('Time:', sr_time)

start_time = time.time()
p_f = mle(p0, M, N, 10)
ml_time = time.time() - start_time
print('p_f =', p_f)
print('Time:', ml_time)