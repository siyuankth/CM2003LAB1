import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def scatter_(data, i ,j,CONTEXT):
    i_sample = data[i, :]
    X = np.zeros((j))
    Y = np.zeros((j))
    Z = np.zeros((j))

    for i in range(j):
        X[i] = i_sample[i * 3 + 0]
        Y[i] = i_sample[i * 3 + 1]
        Z[i] = i_sample[i * 3 + 2]

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.scatter(X, Y, Z, marker='o', color='c', alpha=0.5, linewidths = 0.1)
    ax.set_title(CONTEXT)
    plt.show()



def single_line(data, i, j, CONTEXT):
    if data.shape[0] == 2400:
        i_sample = data
    else:
        i_sample = data[i, :]
    X = np.zeros((j))
    Y = np.zeros((j))
    Z = np.zeros((j))

    for iii in range(j):
        X[iii] = i_sample[iii * 3 + 0]
        Y[iii] = i_sample[iii * 3 + 1]
        Z[iii] = i_sample[iii * 3 + 2]
    X1_new = X[0:100]
    X2_new = X[100:200]
    X3_new = X[200:400]
    X4_new = X[400:600]
    X5_new = X[600:650]
    X6_new = X[650:700]
    X7_new = X[700:800]
    X_new = np.hstack((X2_new, X5_new, X7_new, X6_new[::-1], X1_new[::-1], X2_new[0]))  #
    Y1_new = Y[0:100]
    Y2_new = Y[100:200]
    Y3_new = Y[200:400]
    Y4_new = Y[400:600]
    Y5_new = Y[600:650]
    Y6_new = Y[650:700]
    Y7_new = Y[700:800]

    Y_new = np.hstack((Y2_new, Y5_new, Y7_new, Y6_new[::-1], Y1_new[::-1], Y2_new[0]))
    Z1_new = Z[0:100]
    Z2_new = Z[100:200]
    Z3_new = Z[200:400]
    Z4_new = Z[400:600]
    Z5_new = Z[600:650]
    Z6_new = Z[650:700]
    Z7_new = Z[700:800]
    Z_new = np.hstack((Z2_new, Z5_new, Z7_new, Z6_new[::-1], Z1_new[::-1], Z2_new[0]))
    fig = plt.figure(figsize=(8, 8))
    ax = fig.gca(projection='3d')
            ## Axes3D doesn't have the title
            # ax = Axes3D(fig)

    ax.plot(X_new, Y_new, Z_new, color="deeppink")
    ax.plot(X3_new, Y3_new, Z3_new, color="deeppink")
    ax.plot(X4_new, Y4_new, Z4_new, color="deeppink")

    ax.set_title(CONTEXT)
    plt.show()


def line_(data1, i, j, CONTEXT):
    count = 0
    for ii in range(len(data1)):
        data = data1[ii]
        if data.shape[0] == 2400:
            i_sample = data
        else:
            i_sample = data[i, :]
        X = np.zeros((j))
        Y = np.zeros((j))
        Z = np.zeros((j))

        for iii in range(j):
            X[iii] = i_sample[iii * 3 + 0]
            Y[iii] = i_sample[iii * 3 + 1]
            Z[iii] = i_sample[iii * 3 + 2]
        X1_new = X[0:100]
        X2_new = X[100:200]
        X3_new = X[200:400]
        X4_new = X[400:600]
        X5_new = X[600:650]
        X6_new = X[650:700]
        X7_new = X[700:800]
        X_new = np.hstack((X2_new, X5_new, X7_new, X6_new[::-1], X1_new[::-1], X2_new[0]))  #
        Y1_new = Y[0:100]
        Y2_new = Y[100:200]
        Y3_new = Y[200:400]
        Y4_new = Y[400:600]
        Y5_new = Y[600:650]
        Y6_new = Y[650:700]
        Y7_new = Y[700:800]

        Y_new = np.hstack((Y2_new, Y5_new, Y7_new, Y6_new[::-1], Y1_new[::-1], Y2_new[0]))
        Z1_new = Z[0:100]
        Z2_new = Z[100:200]
        Z3_new = Z[200:400]
        Z4_new = Z[400:600]
        Z5_new = Z[600:650]
        Z6_new = Z[650:700]
        Z7_new = Z[700:800]
        Z_new = np.hstack((Z2_new, Z5_new, Z7_new, Z6_new[::-1], Z1_new[::-1], Z2_new[0]))
        if count == 0:
            fig = plt.figure(figsize=(8, 8))
            ax = fig.gca(projection='3d')
            ## Axes3D doesn't have the title
            # ax = Axes3D(fig)

            ax.plot(X_new, Y_new, Z_new, color="deeppink")
            ax.plot(X3_new, Y3_new, Z3_new, color="deeppink")
            ax.plot(X4_new, Y4_new, Z4_new, color="deeppink")
            count = count + 1
            # ax.set(xlabel="X", ylabel="Y", zlabel="Z")
            # ax.set_title(CONTEXT)
            # ax.set_title(CONTEXT[count])
        else:
            ax.plot(X_new, Y_new, Z_new, color="blue")
            ax.plot(X3_new, Y3_new, Z3_new, color="blue")
            ax.plot(X4_new, Y4_new, Z4_new, color="blue")
            count = count + 1
            # ax.set(xlabel="X", ylabel="Y", zlabel="Z")
            # ax.set_title(CONTEXT)
    ax.set_title(CONTEXT)
    plt.show()



# axes = fig.subplot(nrows = 4, ncols = 4, subplot_kw = dict(fc = 'whitesmoke',
#                                                            projection = '3d'))
# elevs = [0,10,20,30]
# azims = [0,30,45,60]
# for i, theta1 in enumerate(elevs):
#     for j, theta2 in enumerate(azims):
#         ax = axes[i,j]
#         ax.scatter(X, Y, Z, marker='o', color='c', alpha=0.5)
#         ax.view_init(elev=theta1, azim = theta2)
#         ax.set_title(f'elevs:{theta1} azims:{theta2}')
#
# plt.show()

############################### DATA  VISIULAZITION
# i = 0  # i means the number of the row   ith sample
# j = 800  # j means the number of the points
# i_sample = data.values[i,:]
# X = np.zeros((j))
# Y = np.zeros((j))
# Z = np.zeros((j))
#
#
# for i in range(j):
#     X[i] = i_sample[i * 3 + 0]
#     Y[i] = i_sample[i * 3 + 1]
#     Z[i] = i_sample[i * 3 + 2]
# X1_new = X[0:100]
# X2_new = X[100:200]
# X3_new = X[200:400]
# X4_new = X[400:600]
# X5_new = X[600:650]
# X6_new = X[650:700]
# X7_new = X[700:800]
# X_new = np.hstack((X2_new,X5_new,X7_new,X6_new[::-1],X1_new[::-1],X2_new[0]))  #
# Y1_new = Y[0:100]
# Y2_new = Y[100:200]
# Y3_new = Y[200:400]
# Y4_new = Y[400:600]
# Y5_new = Y[600:650]
# Y6_new = Y[650:700]
# Y7_new = Y[700:800]
#
# Y_new = np.hstack((Y2_new,Y5_new,Y7_new,Y6_new[::-1],Y1_new[::-1],Y2_new[0]))
# Z1_new = Z[0:100]
# Z2_new = Z[100:200]
# Z3_new = Z[200:400]
# Z4_new = Z[400:600]
# Z5_new = Z[600:650]
# Z6_new = Z[650:700]
# Z7_new = Z[700:800]
# Z_new = np.hstack((Z2_new,Z5_new,Z7_new, Z6_new[::-1],Z1_new[::-1],Z2_new[0]))
#
# fig = plt.figure()
# ax = fig.gca(projection='3d')
#
# ax.plot(X_new, Y_new, Z_new, color="deeppink")
# ax.plot(X3_new,Y3_new,Z3_new, color="deeppink")
# ax.plot(X4_new,Y4_new,Z4_new,color="deeppink")
# ax.set(xlabel="X", ylabel="Y", zlabel="Z")
# ax.set_title('0th Original suture', fontsize='12')
# plt.pause(0)
