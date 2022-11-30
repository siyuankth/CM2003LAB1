# Read the csv file
import pandas as pd
from tensorflow.keras import losses
import Evaluation
import relevant
import numpy as np
import matplotlib.pyplot as plt
import Visualization
from tensorflow.keras import backend as K
from mpl_toolkits.mplot3d import Axes3D
import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
df = pd.read_csv('X:/Siyuanch/Project2/DATA.csv')
# Relevance analysis (features engineering process)
data = relevant.analysis(df,'Record_ID', False)  # discard the column of Record_ID
# print(data.values[0,0])
# print(int(data.values.shape[1]/3) )

## Random the data
per = np.random.permutation(data.shape[0]) #打乱后的行号
new_data = relevant.random(data.values,per)		#获取打乱后的训练数据

## Data standardization
# from sklearn.preprocessing import scale
# data_s = scale(data)
# data_s = data
from sklearn.preprocessing import StandardScaler
standardizer = StandardScaler()# Same with scale
data_s = standardizer.fit_transform(new_data)

# for i in range(10):
#     print('The current index is %d' %i)
#     [train, test] = relevant.ten_fold(data_s, i)
#     print()
from sklearn.decomposition import PCA
# Initialize the error
Error_LSD_train = []
Error_LMD_train = []
Error_LSD_test = []
Error_LMD_test = []

E_LSD_train_List = []
E_LMD_train_List = []
E_LSD_test_List = []
E_LMD_test_List = []

for i in range(10):
    [train, test] = relevant.ten_fold(data_s, i)

    ## PCA modelling
    # With one certain PCs_nr
    PCs_nr = 5
    pca = PCA(n_components = PCs_nr)
    # pca = PCA(n_components=5)  # 2400 is the number of variables

    X = pca.fit_transform(train)

    # P = pca.components_
    EVR = pca.explained_variance_ratio_
    # num_PC = 5
    # Ratio_sum = sum(EVR[:num_PC]) * 100
    # print('The ratio sum of EVR of %f number of the principle components is %.2f%% '
    #       %(num_PC,Ratio_sum))
    #
    # ## visualization_PCA_explained_variance_ratio
    # plt.plot(np.cumsum(EVR), linewidth = 3)
    # plt.grid()
    # plt.xlabel("number of the PC")
    # plt.ylabel("explained_variance_ratio")
    # plt.title("PCA_explained_variance_ratio with the current index %d" %i)
    # plt.show()

    ## evaluate in train data

    # Back the real data from the PC value in the format of standardization
    Y_train = pca.transform(train)
    train_pred = pca.inverse_transform(Y_train)
    train_diff = train - train_pred
    # LOSS
    LOSS_train = losses.mse(K.reshape(train, (-1,)), K.reshape(train_pred, (-1,))).numpy()
    # LSD and LMD
    real_train = standardizer.inverse_transform(train)
    rebuild_train = standardizer.inverse_transform(train_pred)
    LSD_train = Evaluation.LSD(real_train, rebuild_train)
    LMD_train = Evaluation.LMD(real_train, rebuild_train)

    ## Evaluate in test data
    Y_test = pca.transform(test)
    test_pred = pca.inverse_transform(Y_test)
    test_diff = test - test_pred
    # LOSS
    LOSS_test = losses.mse(K.reshape(test, (-1,)), K.reshape(test_pred, (-1,))).numpy()
    # LSD and LMD
    real_test = standardizer.inverse_transform(test)
    rebuild_test = standardizer.inverse_transform(test_pred)
    LSD_test = Evaluation.LSD(real_test, rebuild_test)
    LMD_test = Evaluation.LMD(real_test, rebuild_test)

    ## Get the mean of error
    Mean_LSD_train = np.mean(LSD_train)
    Mean_LMD_train = np.mean(LMD_train)
    Mean_LSD_test = np.mean(LSD_test)
    Mean_LMD_test = np.mean(LMD_test)
    # Save the mean error

    Error_LSD_train = np.append(Error_LSD_train, Mean_LSD_train)
    Error_LMD_train = np.append(Error_LMD_train, Mean_LMD_train)
    Error_LSD_test = np.append(Error_LSD_test, Mean_LSD_test)
    Error_LMD_test = np.append(Error_LMD_test, Mean_LMD_test)

    # With various PCs_nr and see the PCs_nr impact
    # Initialize the Error in each folder


    # locals()["E_LSD_train_" + str(i)] = []
    # locals()["E_LMD_train_" + str(i)] = []
    # locals()["E_LSD_test_" + str(i)] = []
    # locals()["E_LMD_test_" + str(i)] = []
## Different PCs
#     for PCs_nr in range(1,63):
#         E_LSD_train, E_LMD_train, E_LSD_test, E_LMD_test = relevant.Impact_PCs(train, test, PCs_nr, standardizer)
#         locals()["E_LSD_train_" + str(i)] = np.append(locals()["E_LSD_train_" + str(i)], E_LSD_train)
#         locals()["E_LMD_train_" + str(i)] = np.append(locals()["E_LMD_train_" + str(i)], E_LMD_train)
#         locals()["E_LSD_test_" + str(i)] = np.append(locals()["E_LSD_test_" + str(i)], E_LSD_test)
#         locals()["E_LMD_test_" + str(i)] = np.append(locals()["E_LMD_test_" + str(i)], E_LMD_test)
#     E_LSD_train_List.append(locals()["E_LSD_train_" + str(i)])
#     E_LMD_train_List.append(locals()["E_LMD_train_" + str(i)])
#     E_LSD_test_List.append(locals()["E_LSD_test_" + str(i)])
#     E_LMD_test_List.append(locals()["E_LMD_test_" + str(i)])
#
# # plot the error figure with the different PCs nr
# E_LSD_train_mean_PCs = np.mean(np.array(E_LSD_train_List),0)
# E_LMD_train_mean_PCs = np.mean(np.array(E_LMD_train_List),0)
# E_LSD_test_mean_PCs = np.mean(np.array(E_LSD_test_List),0)
# E_LMD_test_mean_PCs = np.mean(np.array(E_LMD_test_List),0)
# X = range(1,63)
# plt.plot(X,E_LSD_train_mean_PCs, label = "LSD in train data")
# plt.plot(X,E_LMD_train_mean_PCs, label = "LMD in train data")
# plt.plot(X,E_LSD_test_mean_PCs, label = "LSD in test data")
# plt.plot(X,E_LMD_test_mean_PCs, label = "LMD in test data")
# plt.xlabel('The number of PCs')
# plt.ylabel('Mean Error')
# plt.xlim(0,65)
# plt.ylim(0,80)
# plt.legend()
# plt.show()

# Plot

x_data = [f"{i}-fold" for i in range(10)]
width = 0.4
x = len(x_data)
x = np.arange(x)
x1 = x - width/2  # the start location of the first bar
x2 = x1 + width   # The start location of the second bar
p1 = plt.bar(x1, Error_LSD_train, width = 0.4, color = '#d62728')
p2 = plt.bar(x2, Error_LSD_test, width = 0.4)

plt.title('The error for PCA')
plt.xlabel('The sequence number')
plt.ylabel('LSD error')
plt.legend((p1[0],p2[0]),('Train','Test'))
plt.xticks(x, x_data)
plt.show()

p3 = plt.bar(x1, Error_LMD_train, width = 0.4, color = '#d62728')
p4 = plt.bar(x2, Error_LMD_test, width = 0.4)

plt.title('The error for PCA')
plt.xlabel('The sequence number')
plt.ylabel('LMD error')
plt.legend((p3[0],p4[0]),('Train','Test'))
plt.xticks(x, x_data)
plt.show()








#######################################################################################
# X_mean value
X_mean = np.mean(X,axis=0)

# What is the mean value looks like
PCA_mean = pca.inverse_transform(X_mean)
PCA_mean_rebuild = standardizer.inverse_transform(PCA_mean)
Visualization.line_([new_data,PCA_mean_rebuild],0,800,'0th suture compared with the mean suture')
# What is the standard value looks like
X_std = np.std(X,axis = 0)
X_std2 = -1*X_std
PCA_std = pca.inverse_transform(X_std)
PCA_std_rebuild = standardizer.inverse_transform(PCA_std)
Visualization.line_([new_data,PCA_std_rebuild],0,800,'0th suture compared with the + std suture')


PCA_std2 = pca.inverse_transform(X_std2)
PCA_std2_rebuild = standardizer.inverse_transform(PCA_std2)
Visualization.line_([new_data,PCA_std2_rebuild],0,800,'0th suture compared with the - std suture')

Visualization.single_line(PCA_std_rebuild,0,800,'+std suture')
Visualization.single_line(PCA_std2_rebuild,0,800,'-std suture')

## Visulaize the data
# Visualization.scatter_(data.values,0,800,'0th Original scatter')
# Visualization.line_([data.values,data.values],0,800,'0th suture')

# evaluate
# G = S P   data_S = X  pca.component   ## WRONG
## !!!!!!!!!!
# guess = X @ pca.components_
# diff2 = data_s - guess
## !!!!!!!!! DO NOT TRUST DIFF2
# Back the real data from the PC value
# pred = pca.inverse_transform(X)
# diff3 = data_s - pred
#
# ## LOSS
#
# TEMP = losses.mse(K.reshape(data_s, (-1,)), K.reshape(pred, (-1,))).numpy()
#
# ## Diff_RATIO: The ratio of the errors
# Diff3_RATIO = diff3 / data_s
# # Recover to the original data
# Data_rebuild = standardizer.inverse_transform(pred)
# ### Evaluate
# LSD = Evaluation.LSD(data.values, Data_rebuild)
# LMD = Evaluation.LMD(data.values, Data_rebuild)
# diff_rebuild_original = data - Data_rebuild
# Diff_RATIO = diff_rebuild_original / data
# Percent = Diff_RATIO * 100
# ABS = abs(Percent)
#
# ## Find the location of the max value
# aa = np.argmax(ABS)
# r, c = divmod(aa, ABS.shape[1])
# print(r, c)
# print(ABS.stack().idxmax())
# ## Find the max value for dataframe
# bb = ABS.s#找到最大值和最大值所对应的位置
# text.stack().max()
# Out[2]: 89
# text.stack().idxmax()
# Out[3]: (1, 2)
# print('The max error ratio is %f' %(ABS.stack().max()))
# print('The min error ratio is %f' %(ABS.stack().min()))


# mean_1PC = np.mean(X[:,0])
# min_1PC = np.min(X[:,0])
# max_1PC = np.max(X[:,0])
# print('The first PC mean value is %f with the min value of %f and max value of %f'
#       %(np.mean(X[:,0]),np.min(X[:,0]),np.max(X[:,0])))
# print('The first PC mean value is %f with the min value of %f and max value of %f'
#       %(np.mean(X[:,1]),np.min(X[:,1]),np.max(X[:,1])))
# print('The first PC mean value is %f with the min value of %f and max value of %f'
#       %(np.mean(X[:,2]),np.min(X[:,2]),np.max(X[:,2])))

# Visualization.line_([data.values,Data_rebuild],0,800,'0th suture by PCA')
# Visualization.line_([data.values,Data_rebuild],1,800,'1st suture by PCA')
# Visualization.line_([data.values,Data_rebuild],2,800,'2nd suture by PCA')
# Visualization.line_([data.values,Data_rebuild],3,800,'3rd suture by PCA')
# Visualization.line_([data.values,Data_rebuild],4,800,'4th suture by PCA')
# Visualization.line_([data.values,Data_rebuild],5,800,'5th suture by PCA')
# Visualization.line_([data.values,Data_rebuild],6,800,'6th suture by PCA')
# Visualization.line_([data.values,Data_rebuild],7,800,'7th suture by PCA')
# Visualization.line_([data.values,Data_rebuild],8,800,'8th suture by PCA')
# Visualization.line_([data.values,Data_rebuild],9,800,'9th suture by PCA')
# Visualization.line_([data.values,Data_rebuild],10,800,'10th suture by PCA')
# Visualization.line_([data.values,Data_rebuild],11,800,'11th suture by PCA')
# Visualization.line_([data.values,Data_rebuild],12,800,'12th suture by PCA')


print(data)



