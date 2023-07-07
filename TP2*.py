"""
@Topic: TP2* 
@Version: 1.0
"""
import matplotlib.pyplot as plt
from enum import auto
from secrets import randbelow
import numpy as np
import numpy.random as npr
import math


def Read_Query_feature_vector(T, n):
    X = np.zeros((T, n), float)
    fX = open(".simualtion/X_T_%d_n_%d"%(T,n))
    for t in range(T):
        line = fX.readline()
        if not line:
            break
        linesp = line.split()
        for i in range(n):
            X[t, i] = float(linesp[i])
    fX.close()
    return X

def Read_Query_market_value_thetastar(T, n):
    thetaStar = np.zeros((n, 1), float)
    fTheta = open(".simulation/theta_T_%d_n_%d"%(T,n))
    line = fTheta.readline()
    linesp = line.split()
    for i in range(n):
        thetaStar[i,0] = float(linesp[i])
    fTheta.close()
    return thetaStar
    
"""
Some Global/Inital Variables
"""   
n = 10 #dimension of features vector
T_e = 2000 #number of rounds
R = 1 #upper bound of 2-norm of ThetaStar
epsilon = 1 / (T_e) #threshold
alphat = 1.0 /(n**2) #切割处位置参数
p_0 = 0.1 
"""
Theta Star
"""
ThetaStar = Read_Query_market_value_thetastar(T_e, n)

"""
Item Vector
"""
X = Read_Query_feature_vector(T_e, n)

"""
TP2*
"""
#shape matrix and center of the current ellipsoid
R2 = R**2 * n
A = np.identity(n)
for i in range(n):
     A[i, i] = R2
c = np.zeros((1,n),float)
"""
Resetting Some Counting/Recording Variables
"""
regretVec = np.zeros(T_e, float) #存储每一轮的regret
totalregretVec = np.zeros(T_e, float) #总计regret
totalMarketValue = np.zeros(T_e, float) 
regretRatio = np.zeros(T_e, float)

#main alg.
for t in range(T_e):
    #Judge whether theta* is in the current ellipsoid
    if (np.dot(np.dot((ThetaStar.transpose() - c.transpose()).transpose(), np.linalg.inv(A)), (ThetaStar.transpose() - c.transpose())) <= 1):
        pass
        #print("Round %d: Yes! Theta* is within the current ellipsoid."%t)
    else:
        #print("Round %d: No! Theta* is outside the current ellipsoid."%t)
        x = input("错误") #发现问题，停止程序
     
    xt = X[t,0:n+1] 
    xt = np.array([xt]) #xt行向量
    xt_T = xt.transpose() #xt列向量
    vt = np.dot(xt, ThetaStar.transpose()) #the market value
    bt_T = np.dot(A, xt_T)/(1.0 * math.sqrt(np.dot(np.dot(xt, A), xt_T))) #intermediate vector列向量
    bt = bt_T.transpose() #行向量
    
    #lower bound and upper bound of current round
    pt = 0.0 #inital post price
    pt_upper = np.dot(xt[0] , c[0] + bt[0])
    pt_lower = np.dot(xt[0] , c[0] - bt[0])
    
    #posted price pt1 or pt2
    if((pt_upper - pt_lower) > epsilon):
        pt = (pt_lower + pt_upper) / 2.0
        p = np.random.binomial(1,p_0,1)
        if(p == 0):
            pt1 = alphat * (pt_upper - pt) + pt
            #print("Posted price1: %f"%pt1)
            if pt1 <= vt:
                nextA = (n**2 * (1 - alphat**2) *1.0 / (n**2 - 1)) * (A - (2.0 * (1 + n * alphat) * np.dot(bt_T,bt)) / ((n + 1) * (1 + alphat))) 
                nextc =  c + (1  + n*alphat) * bt / (n + 1) 
                A = nextA
                c = nextc
                regret = pt_upper - pt1
            else: 
                pt2 = pt - alphat * (pt - pt_lower)
                #print("Posted price2: %f"%pt2)
                if pt2 <= vt:
                    nextA = n * (1 - alphat**2) / (n - 1) * (A - (1 - n * alphat**2) / (1 - alphat**2) * np.dot(bt_T,bt))
                    nextc = c 
                    A = nextA
                    c = nextc
                    regret = pt1 - pt2
                else:
                    nextA = (n**2 * (1 - alphat**2) / (n**2 - 1)) * (A - (2 * (1 + n * alphat) * np.dot(bt_T,bt)) / ((n + 1) * (1 + alphat))) 
                    nextc = c - (1 + n*alphat) * bt / (n+1) 
                    A = nextA
                    c = nextc
                    regret = vt
        else:
            pt = pt 
            if pt <= vt:
                nextA = (n**2 * 1.0 / (n**2-1)) * (A - (2.0 * np.dot(bt_T,bt)) / ((n + 1))) 
                nextc =  c + bt / (n + 1) 
                A = nextA
                c = nextc
                regret = pt_upper - pt
            else: 
                nextA = (n**2 * 1.0 / (n**2 - 1)) * (A - (2 * 1.0 * np.dot(bt_T,bt)) / ((n + 1) * 1.0)) 
                nextc = c -  bt / (n + 1) 
                A = nextA
                c = nextc
                regret = vt
    else:
        pt = pt_lower # A = A; c = c
        #print("Posted price3: %f"%pt)
        regret = pt_upper - pt_lower
    
    regretVec[t] = regret
    totalregretVec[t] = totalregretVec[t - 1] + regret 
    totalMarketValue[t] = totalMarketValue[t - 1] + vt
    #print("Market value: %f; Pt_low: %f; Pt_up: %f" % (vt, pt_lower, pt_upper))
    #print("Regret: %f\n"%(regretVec[t]))


#save regret vector to file
np.savetxt(".simulation/2Regrets_n%d_T_%d*.txt" % (n,T_e), regretVec, fmt='%.10f')
np.savetxt(".simulation/2TotalRegrets_n%d_T_%d*.txt" % (n, T_e), totalregretVec, fmt='%.10f')

#save regret ratio to file
for t in range(T_e):
    if(totalMarketValue[t] != 0):
        regretRatio[t] = totalregretVec[t] * 1.0 / totalMarketValue[t]
np.savetxt(".simulation/2RegretRatio_n%d_T_%d*.txt" % (n, T_e), regretRatio, fmt='%.10f')

#print("Market value: %f; Pt_low: %f; Pt_up: %f" % (vt, pt_lower, pt_upper))
#print("Posted price: %f"%pt)
#print("Regret: %f\n"%(regretVec[t]))