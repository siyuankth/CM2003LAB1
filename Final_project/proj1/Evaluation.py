import numpy as np

def LSD(data1_value,data2_value):
    if data1_value.shape != data2_value.shape:
        return False
    else:
        n = int(data1_value.shape[1]/3) # nr of points
        m = data1_value.shape[0]  # nr of sample
        lsd = np.zeros(shape=(m,1))
        for j in range(m):
            i_sample1 = data1_value[j, :]
            X1 = np.zeros((n))
            Y1 = np.zeros((n))
            Z1 = np.zeros((n))
            i_sample2 = data2_value[j, :]
            X2 = np.zeros((n))
            Y2 = np.zeros((n))
            Z2 = np.zeros((n))
            sum = 0
            for i in range(n):
                X1[i] = i_sample1[i * 3 + 0]
                Y1[i] = i_sample1[i * 3 + 1]
                Z1[i] = i_sample1[i * 3 + 2]
                X2[i] = i_sample2[i * 3 + 0]
                Y2[i] = i_sample2[i * 3 + 1]
                Z2[i] = i_sample2[i * 3 + 2]
                sum = np.square(X1[i] - X2[i]) + np.square(Y1[i] - Y2[i]) + np.square(Z1[i] - Z2[i]) + sum
            lsd[j,0] = sum/n
        return lsd

def LMD(data1_value,data2_value):
    if data1_value.shape != data2_value.shape:
        return False
    else:
        n = int(data1_value.shape[1]/3) # nr of points
        m = data1_value.shape[0]  # nr of sample
        lmd = np.zeros(shape=(m,1))
        for j in range(m):
            i_sample1 = data1_value[j, :]
            X1 = np.zeros((n))
            Y1 = np.zeros((n))
            Z1 = np.zeros((n))
            i_sample2 = data2_value[j, :]
            X2 = np.zeros((n))
            Y2 = np.zeros((n))
            Z2 = np.zeros((n))
            sum = 0
            for i in range(n):
                X1[i] = i_sample1[i * 3 + 0]
                Y1[i] = i_sample1[i * 3 + 1]
                Z1[i] = i_sample1[i * 3 + 2]
                X2[i] = i_sample2[i * 3 + 0]
                Y2[i] = i_sample2[i * 3 + 1]
                Z2[i] = i_sample2[i * 3 + 2]
                sum = abs(X1[i] - X2[i]) + abs(Y1[i] - Y2[i]) + abs(Z1[i] - Z2[i]) + sum
            lmd[j,0] = sum/n
        return lmd