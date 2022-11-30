# Read the csv file
import pandas as pd
import relevant
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses
from tensorflow.keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import savemat
import Visualization
import Evaluation
import VAE_function
import os


#  Read the data
df = pd.read_csv('X:/Siyuanch/Project2/DATA.csv')
# Relevance analysis (features engineering process)
data = relevant.analysis(df, 'Record_ID', False)  # discard the column of Record_ID

## Random the data
# Use the same order with PCA
per = np.array([15,51,48,39,7,2,13,30,64,65,20,1,52,54,34,67,60,0,25,45,24,23,53,19,37,59,10,22,4,68,27,42,40,41,63,57,11,35,55,14,5,26,31,62,49,18,33,43,66,61,44,6,12,32,16,38,28,36,17,21,50,47,9,3,8,29,58,46,56])
# Use the random order
# per = np.random.permutation(data.shape[0]) #打乱后的行号
new_data = relevant.random(data.values,per)

## Data standardization
from sklearn.preprocessing import StandardScaler
standardizer = StandardScaler()# Same with scale
data_s = standardizer.fit_transform(new_data)
################################
Error_LSD_train = []
Error_LMD_train = []
Error_LSD_test = []
Error_LMD_test = []

E_LSD_train_List = []
E_LMD_train_List = []
E_LSD_test_List = []
E_LMD_test_List = []

## Ten floder



n = 2400

latent_dim = 5
beta = 1e-3
base = 16
conv_size = 3
padding_size = 2
st = 1
epoch = 1000
act = 'tanh'
lr = 0.00001
name = 'VAE'
rank = 'TRY'
model_name = f'_{name}.h5'
act = 'tanh'

Epochs_LSD_train_List = []
Epochs_LMD_train_List = []
Epochs_LSD_test_List = []
Epochs_LMD_test_List = []
#The impact of epochs
# for j in range(1,6):
#     epoch = 2000 * j
#     Sum_1 = 0
#     Sum_2 = 0
#     Sum_3 = 0
#     Sum_4 = 0
#     for i in range(10):
#         [train, test] = relevant.ten_fold(data_s, i)
#         folder_name = f'VAE_{i}'
#         wdir = f'./{folder_name}/'
#
#         Epochs_LSD_train, Epochs_LMD_train, Epochs_LSD_test, Epochs_LMD_test = relevant.Impact_VAE(train,test,standardizer,latent_dim,
#                                                                                           beta,base,conv_size,padding_size,st,
#                                                                                           epoch,act,lr,n)
#         Sum_1 = Sum_1 + Epochs_LSD_train
#         Sum_2 = Sum_2 + Epochs_LMD_train
#         Sum_3 = Sum_3 + Epochs_LSD_test
#         Sum_4 = Sum_4 + Epochs_LMD_test
#         # List
#     Sum_1 = Sum_1 / 10
#     Sum_2 = Sum_2 / 10
#     Sum_3 = Sum_3 / 10
#     Sum_4 = Sum_4 / 10
#
#     Epochs_LSD_train_List.append(Sum_1)
#     Epochs_LMD_train_List.append(Sum_2)
#     Epochs_LSD_test_List.append(Sum_3)
#     Epochs_LMD_test_List.append(Sum_4)



    # Epochs_LSD_train_List = np.append(Epochs_LSD_train_List, Epochs_LSD_train)
    # Epochs_LMD_train_List = np.append(Epochs_LMD_train_List, Epochs_LMD_train)
    # Epochs_LSD_test_List = np.append(Epochs_LSD_test_List, Epochs_LSD_test)
    # Epochs_LMD_test_List = np.append(Epochs_LMD_test_List, Epochs_LMD_test)
#
# # Save the results
# columns = ['Epochs_LSD_train_Error','Epochs_LMD_train_Error','Epochs_LSD_test_Error','Epochs_LMD_test_Error']
# df = pd.DataFrame(list(zip(Epochs_LSD_train_List,Epochs_LMD_train_List,Epochs_LSD_test_List,Epochs_LMD_test_List)),
#                   columns = columns)
# print(df)
# df.to_excel('Epochs_impact.xlsx')

# df.to_excel('Courses.xlsx', sheet_name='Technologies')


## Visualization
# train_encode = encoder.predict(train_data)
# train_predict = decoder.predict(train_encode)
# train_predict = np.squeeze(train_predict)
# rebuild_train = standardizer.inverse_transform(train_predict)
#
# test_encode = encoder.predict(test_data)
# test_predict = decoder.predict(test_encode)
# test_predict = np.squeeze(test_predict)
# rebuild_test = standardizer.inverse_transform(test_predict)
#
# real_test = standardizer.inverse_transform(test)
# real_train = standardizer.inverse_transform(train)
#
#
# Visualization.line_([real_train,rebuild_train],0,800,'0th suture by VAE')
# Visualization.line_([real_test,rebuild_test],0,800,'0th suture by VAE')





for i in range(10):
    [train, test] = relevant.ten_fold(data_s, i)
    folder_name = f'VAE_{i}'
    wdir = f'./{folder_name}/'
    # encoder = models.load_model('en_try_VAE.h5', custom_objects={'sampling': sampling})
    # print(encoder.summary())

    inp = layers.Input(shape=(n, 1))
    x = layers.Conv1D(base, conv_size, activation=act, strides=st, padding='same')(inp)
    x = layers.MaxPooling1D(padding_size, padding='same')(x)
    x = layers.Conv1D(base * 2, conv_size, activation=act, strides=st, padding='same')(x)
    x = layers.MaxPooling1D(padding_size, padding='same')(x)
    x = layers.Conv1D(base * 4, conv_size, activation=act, strides=st, padding='same')(x)
    x = layers.MaxPooling1D(padding_size, padding='same')(x)
    x = layers.Conv1D(base * 8, conv_size, activation=act, strides=st, padding='same')(x)
    x = layers.MaxPooling1D(padding_size, padding='same')(x)
    x = layers.Conv1D(base * 16, conv_size, activation=act, strides=st, padding='same')(x)
    x = layers.MaxPooling1D(padding_size, padding='same')(x)
    x = layers.Flatten()(x)
    x = layers.Dense(base * 8, activation=act)(x)  # 128
    z_mean = layers.Dense(latent_dim)(x)
    z_log_sigma = layers.Dense(latent_dim)(x)
    z = layers.Lambda(VAE_function.sampling)([z_mean, z_log_sigma])

    ##########Encoder
    encoder = models.Model(inp, z)
    print(encoder.summary())
    ############

    code_i = layers.Input(shape=(latent_dim,))
    x = layers.Dense(base * 8, activation=act)(code_i)
    x = layers.Dense(int(base * 16 * n / (2 ** 5)), activation=act)(x)  # 5 means the number of the conv1D
    x = layers.Reshape((int(n / (2 ** 5)), base * 16))(x)
    x = layers.UpSampling1D(padding_size)(x)
    x = layers.Conv1D(base * 16, conv_size, activation=act, strides=st, padding='same')(x)
    x = layers.UpSampling1D(padding_size)(x)
    x = layers.Conv1D(base * 8, conv_size, activation=act, strides=st, padding='same')(x)
    x = layers.UpSampling1D(padding_size)(x)
    x = layers.Conv1D(base * 4, conv_size, activation=act, strides=st, padding='same')(x)
    x = layers.UpSampling1D(padding_size)(x)
    x = layers.Conv1D(base * 2, conv_size, activation=act, strides=st, padding='same')(x)
    x = layers.UpSampling1D(padding_size)(x)
    x = layers.Conv1D(base * 1, conv_size, activation=act, strides=st, padding='same')(x)
    out = layers.Conv1D(1, conv_size, strides=st, padding='same')(x)
    decoder = models.Model(code_i, out)
    print(decoder.summary())

    # print(K.reshape(inp, (-1,)))

    out_d = decoder(encoder(inp))

    model = models.Model(inp, out_d)
    print(model.summary())

    rec_loss = losses.mse(K.reshape(inp, (-1,)), K.reshape(out_d, (-1,)))
    # rec_loss = losses.MeanSquaredError(inp,out_d)
    kl_loss = 1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma)
    kl_loss = K.sum(kl_loss, axis=-1)  ############ sum or mean
    kl_loss *= -0.5
    vae_loss = K.mean(rec_loss + beta * kl_loss)
    model.add_loss(vae_loss)
    model.add_metric(rec_loss, name='rec_loss',
                     aggregation='mean')  ## metric is just the print metric not involved in the training
    model.add_metric(kl_loss, name='kl_loss', aggregation='mean')

    opt = optimizers.Adam(learning_rate=lr)
    # model.compile(optimizer = opt)
    model.compile(optimizer=opt)

    train_data = train[:, :, np.newaxis]
    test_data = test[:,:, np.newaxis]
    hist = model.fit(x=train_data, y=None, epochs=epoch)  # train_data

    os.makedirs(wdir, exist_ok=True)
    savemat(wdir + f'CNNVAEloss_{rank}.mat', hist.history)

    encoder.save(wdir + 'en' + model_name)
    decoder.save(wdir + 'de' + model_name)

## -------------------------------------After training -------------------------------------------------------------------
    # Loss
    plt.figure(figsize=(4, 4))
    plt.title("Learning curve")
    plt.plot(hist.history["loss"], label="loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss Value")
    plt.legend()
    plt.show()

    train_encode = encoder.predict(train_data)
    train_predict = decoder.predict(train_encode)
    train_predict = np.squeeze(train_predict)
    rebuild_train = standardizer.inverse_transform(train_predict)

    test_encode = encoder.predict(test_data)
    test_predict = decoder.predict(test_encode)
    test_predict = np.squeeze(test_predict)
    rebuild_test = standardizer.inverse_transform(test_predict)

    real_test = standardizer.inverse_transform(test)
    real_train = standardizer.inverse_transform(train)
    ### Evaluate
    LSD_test = Evaluation.LSD(real_test, rebuild_test)
    LMD_test = Evaluation.LMD(real_test, rebuild_test)
    LSD_train = Evaluation.LSD(real_train, rebuild_train)
    LMD_train = Evaluation.LMD(real_train, rebuild_train)

    Mean_LSD_train = np.mean(LSD_train)
    Mean_LMD_train = np.mean(LMD_train)
    Mean_LSD_test = np.mean(LSD_test)
    Mean_LMD_test = np.mean(LMD_test)

    Error_LSD_train = np.append(Error_LSD_train, Mean_LSD_train)
    Error_LMD_train = np.append(Error_LMD_train, Mean_LMD_train)
    Error_LSD_test = np.append(Error_LSD_test, Mean_LSD_test)
    Error_LMD_test = np.append(Error_LMD_test, Mean_LMD_test)

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










x_encode = encoder.predict(train_data)
x_predict = decoder.predict(x_encode)