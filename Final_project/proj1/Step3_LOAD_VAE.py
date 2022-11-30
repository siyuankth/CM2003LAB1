import numpy as np
from tensorflow.keras import models
from tensorflow.keras import losses
import Visualization
import pandas as pd
import Evaluation
import relevant
from tensorflow.keras import backend as K
import VAE_function

################################ LOAD   THE MODEL #######################
wdir = './'

name = 'VAE'
model_name = f'_{name}.h5'

latent_dim = 5

encoder = models.load_model('en_VAE.h5')  #, custom_objects={'sampling': sampling}
print(encoder.summary())

decoder = models.load_model(wdir + 'de' + model_name)
print(decoder.summary())

inp = encoder.layers[0].input
out_d = decoder(encoder(inp))

model = models.Model(inp, out_d)
print(model.summary())


#######################LOAD THE DATA ################################

# Read the data
df = pd.read_csv('X:/Siyuanch/Project2/DATA.csv')
# Relevance analysis (features engineering process)
data = relevant.analysis(df,'Record_ID', False)  # discard the column of Record_ID

from sklearn.preprocessing import StandardScaler
standardizer = StandardScaler()  # Same with scale
data_s = standardizer.fit_transform(data)

######################################## Use the model #################
train_data = data_s[:, :, np.newaxis]
x_encode = encoder.predict(train_data)
x_predict = decoder.predict(x_encode)
x_predict = np.squeeze(x_predict)
Data_rebuild = standardizer.inverse_transform(x_predict)

### Evaluate
LSD = Evaluation.LSD(data.values, Data_rebuild)
LMD = Evaluation.LMD(data.values, Data_rebuild)

Visualization.line_([data.values,Data_rebuild],0,800,'0th suture by VAE')
Visualization.line_([data.values,Data_rebuild],1,800,'1st suture by VAE')
Visualization.line_([data.values,Data_rebuild],2,800,'2nd suture by VAE')
Visualization.line_([data.values,Data_rebuild],3,800,'3rd suture by VAE')
Visualization.line_([data.values,Data_rebuild],4,800,'4th suture by VAE')
Visualization.line_([data.values,Data_rebuild],5,800,'5th suture by VAE')
Visualization.line_([data.values,Data_rebuild],6,800,'6th suture by VAE')
Visualization.line_([data.values,Data_rebuild],7,800,'7th suture by VAE')
Visualization.line_([data.values,Data_rebuild],8,800,'8th suture by VAE')
Visualization.line_([data.values,Data_rebuild],9,800,'9th suture by VAE')
Visualization.line_([data.values,Data_rebuild],10,800,'10th suture by VAE')
Visualization.line_([data.values,Data_rebuild],11,800,'11th suture by VAE')
Visualization.line_([data.values,Data_rebuild],12,800,'12th suture by VAE')

# What is the mean value looks like
X_mean = np.mean(x_encode, axis=0)
X_mean = X_mean.reshape(1,5)
VAE_mean = decoder.predict(X_mean)
VAE_mean = np.squeeze(VAE_mean)
VAE_mean_rebuild = standardizer.inverse_transform(VAE_mean)
Visualization.line_([data.values,VAE_mean_rebuild],0,800,'0th suture compared with the mean suture')
# What is the standard value looks like
X_std = np.std(x_encode, axis=0)
X_std = X_std.reshape(1,5)
X_std = X_std + X_mean
VAE_std = decoder.predict(X_std)
VAE_std = np.squeeze(VAE_std)
VAE_std_rebuild = standardizer.inverse_transform(VAE_std)
Visualization.line_([data.values,VAE_std_rebuild],0,800,'0th suture compared with the + std suture')

X_std2 = -1*X_std
X_std2 = X_std2 + X_mean
VAE_std2 = decoder.predict(X_std2)
VAE_std2 = np.squeeze(VAE_std2)
VAE_std2_rebuild = standardizer.inverse_transform(VAE_std2)
Visualization.line_([data.values,VAE_std2_rebuild],0,800,'0th suture compared with the - std suture')

Visualization.single_line(VAE_std_rebuild,0,800,'+std suture')
Visualization.single_line(VAE_std2_rebuild,0,800,'-std suture')

## LOSS

TEMP = losses.mse(K.reshape(data_s, (-1,)), K.reshape(x_predict, (-1,))).numpy()

print(data)