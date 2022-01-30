import os
import numpy as np

# change directory to prism
print(os.getcwd())
os.chdir("C:/Programme/prism-4.5/bin")


# define number of simulated paths
num_paths = 500        

# reasonable p's from 0.2 to 0.4 and 0.6 to 0.8
#p1 = np.arange(0.2, 0.41, 0.01).round(decimals=2)
#p2 = np.arange(0.6, 0.81, 0.01).round(decimals=2)
#p_range = np.concatenate([p1, p2])

p = [0.34,0.25,0.2,0.4,0.21,0.64,0.26,0.77,0.34,0.65]

p2 = [0.32,0.39]
p5 = [0.37,0.44,0.51,0.58,0.65]
#p10 = [0.27, 0.34, 0.41, 0.48, 0.55, 0.62, 0.69, 0.76, 0.83, 0.90]
#p10 = [0.22, 0.29, 0.36, 0.43, 0.50, 0.57, 0.64, 0.71, 0.78, 0.85]
p10 = [0.17, 0.24, 0.31, 0.38, 0.45, 0.52, 0.59, 0.66, 0.73, 0.80]


"""

### first case: random p

# a) group size n = 2
n = 2
#p = np.random.choice(p_range, n)
#np.savetxt('../../../Users/klein/Documents/Uni/Master_Project/experiment/values_p_' + str(n) +'.txt', (p))
# simulate chain and save paths
for i in range(0, num_paths):
                os.system('prism ../../../Users/klein/Documents/Uni/Master_Project/casestudy_' + str(n) + '.pm -const r_0=' + str(p[0]) + ',r_1=' + str(p[1]) + ' -simpath vars=(b),time=15 \
                          ../../../Users/klein/Documents/Uni/Master_Project/experiment/random_' + str(n) + '/path_' + str(i) + '.txt -simpathlen 2000')


# b) group size n = 5
n = 5
#p = np.random.choice(p_range, n)
#np.savetxt('../../../Users/klein/Documents/Uni/Master_Project/experiment/values_p_' + str(n) +'.txt', (p))
# simulate chain and save paths
for i in range(0, num_paths):
                os.system('prism ../../../Users/klein/Documents/Uni/Master_Project/casestudy_' + str(n) + '.pm -const r_0=' + str(p[0]) + ',r_1=' + str(p[1]) + 
                          ',r_2=' + str(p[2]) + ',r_3=' + str(p[3]) + ',r_4=' + str(p[4]) + ' -simpath vars=(b),time=15 \
                          ../../../Users/klein/Documents/Uni/Master_Project/experiment/random_' + str(n) + '/path_' + str(i) + '.txt -simpathlen 2000') 
                          


# c) group size n = 10
n = 10
#p = np.random.choice(p_range, n)
p = [0.34,0.25,0.2,0.4,0.21,0.64,0.26,0.77,0.34,0.65]
np.savetxt('../../../Users/klein/Documents/Uni/Master_Project/experiment/values_p_' + str(n) +'.txt', (p))
# simulate chain and save paths
for i in range(0, 2*num_paths):
                os.system('prism ../../../Users/klein/Documents/Uni/Master_Project/casestudy_' + str(n) + '.pm -const r_0=' + str(p[0]) + ',r_1=' + str(p[1]) + 
                          ',r_2=' + str(p[2]) + ',r_3=' + str(p[3]) + ',r_4=' + str(p[4]) + ',r_5=' + str(p[5]) + ',r_6=' + str(p[6]) + ',r_7=' + str(p[7]) +
                          ',r_8=' + str(p[8]) + ',r_9=' + str(p[9]) + ' -simpath vars=(b),time=15 \
                          ../../../Users/klein/Documents/Uni/Master_Project/experiment/random_' + str(n) + '/path_' + str(i) + '.txt -simpathlen 2000')



### second case a: linear p, same slope

# a) group size n = 2
n = 2
#p = np.random.choice(p_range, n)
#np.savetxt('../../../Users/klein/Documents/Uni/Master_Project/experiment/values_p_' + str(n) +'.txt', (p))
# simulate chain and save paths
for i in range(0, num_paths):
                os.system('prism ../../../Users/klein/Documents/Uni/Master_Project/casestudy_' + str(n) + '.pm -const r_0=' + str(p2[0]) + ',r_1=' + str(p2[1]) + ' -simpath vars=(b),time=15 \
                          ../../../Users/klein/Documents/Uni/Master_Project/experiment/linear_a_' + str(n) + '/path_' + str(i) + '.txt -simpathlen 2000')


# b) group size n = 5
n = 5
#p = np.random.choice(p_range, n)
#np.savetxt('../../../Users/klein/Documents/Uni/Master_Project/experiment/values_p_' + str(n) +'.txt', (p))
# simulate chain and save paths
for i in range(0, num_paths):
                os.system('prism ../../../Users/klein/Documents/Uni/Master_Project/casestudy_' + str(n) + '.pm -const r_0=' + str(p5[0]) + ',r_1=' + str(p5[1]) + 
                          ',r_2=' + str(p5[2]) + ',r_3=' + str(p5[3]) + ',r_4=' + str(p5[4]) + ' -simpath vars=(b),time=15 \
                          ../../../Users/klein/Documents/Uni/Master_Project/experiment/linear_a_' + str(n) + '/path_' + str(i) + '.txt -simpathlen 2000') 
                          

"""

# c) group size n = 10
n = 10
#p = np.random.choice(p_range, n)
#p = [0.34,0.25,0.2,0.4,0.21,0.64,0.26,0.77,0.34,0.65]
#np.savetxt('../../../Users/klein/Documents/Uni/Master_Project/experiment/values_p_' + str(n) +'.txt', (p))
# simulate chain and save paths
for i in range(0, 2*num_paths):
                os.system('prism ../../../Users/klein/Documents/Uni/Master_Project/casestudy_' + str(n) + '.pm -const r_0=' + str(p10[0]) + ',r_1=' + str(p10[1]) + 
                          ',r_2=' + str(p10[2]) + ',r_3=' + str(p10[3]) + ',r_4=' + str(p10[4]) + ',r_5=' + str(p10[5]) + ',r_6=' + str(p10[6]) + ',r_7=' + str(p10[7]) +
                          ',r_8=' + str(p10[8]) + ',r_9=' + str(p10[9]) + ' -simpath vars=(b),time=15 \
                          ../../../Users/klein/Documents/Uni/Master_Project/experiment/linear_a3_' + str(n) + '/path_' + str(i) + '.txt -simpathlen 2000')
