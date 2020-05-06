'''''
Part 1. In Matlab, or the programming language of your choice, do the following:
• Generate N observations from a normal distribution: data = randn(N, 1); This will generate N 1-D samples with mean equal to 0 and variance equal to 1.
• Estimate the mean and variance of the data for N = 10, 100, 1000 etc.
• Modify the code so that the generated data have mean and variance equal to user-specified
parameters mean and var.
Submit a function that accepts mean, var and N and generates data as in the last step above.
No explanations required.
'''''

# randn和standard_normal都只能返回标准正态分布，对于更一般的正态分布Ν(μ, σ2), 需要使用
#
# σ * np.random.randn(...) + μ
import numpy as np
from pip._vendor.distlib.compat import raw_input
import math

def run(average, var, k):


    var=math.sqrt(var)

    data = average + var*np.random.randn(k)
    print(data)

    return data







if __name__ == '__main__':
    ave=raw_input("mean is : ")
    var=raw_input("variance is : ")
    k=raw_input("K is : ")
    var=float(var)
    ave=float(ave)
    k=int(k)
    run(ave,var,k)




