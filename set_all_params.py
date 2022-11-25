from scipy.stats import qmc
import pickle
import os
# Press the green button in the gutter to run the script.

F_STAR10 = [-3.0, -0.0]
ALPHA_STAR = [-0.5, 1]
F_ESC10 = [-3.0, -0.0]
ALPHA_ESC = [-1 , 0.5]
M_TURN = [8.0, 10.0]
t_STAR = [0.0, 1.0]
L_X = [30, 42]
E_0 = [100, 1500]
X_RAY_SPEC_INDEX = [-1,3]


l_bounds = [F_STAR10[0],ALPHA_STAR[0], F_ESC10[0],ALPHA_ESC[0], M_TURN[0],t_STAR[0], L_X[0] ,E_0[0], X_RAY_SPEC_INDEX[0]] 
u_bounds = [F_STAR10[1],ALPHA_STAR[1], F_ESC10[1],ALPHA_ESC[1], M_TURN[1],t_STAR[1], L_X[1] ,E_0[1], X_RAY_SPEC_INDEX[1]] 
sampler = qmc.LatinHypercube(d=9)
sample = sampler.random(n=100000)
paramspace = qmc.scale(sample, l_bounds, u_bounds)
n = len(paramspace)/2500
print(n)
for i in range(int(n)):
    outputName = os.getcwd() +'/new_samples/21cmfastData_batch' + str(i)+'.pk'
    
    for params in paramspace[int(n*(i-1)):int(n*(i-1))+2500]:
        print(params.shape)
        exit(0)
        pickle.dump(params, open(outputName, 'ab'))
