import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

train=pd.read_json("C:/Users/NISARG/Downloads/SHIP_ICE/data/processed/train.json")
test=pd.read_json("C:/Users/NISARG/Downloads/SHIP_ICE/data/processed/test.json")

print(train.head())

x_band1=np.array([np.array(band).astype(np.float32).reshape(75,75) for band in train["band_1"]])
x_band2=np.array([np.array(band).astype(np.float32).reshape(75,75) for band in train["band_2"]])
X_train=np.concatenate([x_band1[:,:,:,np.newaxis],x_band2[:,:,:,np.newaxis]],axis=-1)
y_train=np.array(train["is_iceberg"])
print("Xtrain:",X_train.shape)


x_band1=np.array([np.array(band).astype(np.float32).reshape(75,75) for band in test["band_1"]])
x_band2=np.array([np.array(band).astype(np.float32).reshape(75,75) for band in test["band_2"]])
X_test=np.concatenate([x_band1[:,:,:,np.newaxis],x_band2[:,:,:,np.newaxis]],axis=-1)
print("X_test: ",X_test.shape)

from keras.models import Sequential
from keras.layers import Dense,Convolution2D,Flatten,Dropout
from keras.models import MaxPooling2D

model=Sequential()

model.add(Convolution2D(32,3, activation="relu",input_shape=(75,75,2)))

model.add(Convolution2D(64,3,activation="relu",input_shape=(75,75,2)))

model.add(GlobalAveragePooling2D())

#model.add(Flatten())

model.add(Dropout(0.3))

model.add(Dense(1,activation="sigmoid"))

model.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])

print(model.summary())

model.fit(X_train,y_train,validation_split=0.2,epochs=25)

prediction=model.predict(X_test,verbose=1)

submit_df = pd.DataFrame({'id': test["id"], 'is_iceberg': prediction.flatten()})
submit_df.to_csv("./naive_new_submission.csv", index=False)
