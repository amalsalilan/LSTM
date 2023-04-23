from keras.datasets import mnist
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

(X_train,y_train),(X_test,y_test)=mnist.load_data()

# Normalise pixel values to between 0 and 1
X_train = X_train /255.0
X_test = X_test /255.0

# Convert labels to onehot encoding
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

X_train,X_val,y_train,y_val =train_test_split(X_train,y_train,test_size=0.2,random_state=42)

# Lets Define the architeture of the model

from keras.models import Sequential
from keras.layers import Flatten,Dense

model=Sequential([
    Flatten(input_shape=(28,28)),
    Dense(128,activation='relu'),
    Dense(10,activation='softmax')
])
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

from keras.callbacks import EarlyStopping

# Define early stoping callback
early_stop=EarlyStopping(monitor='val_loss',patience=5)

history=model.fit(X_train,y_train,epochs=100,validation_data=(X_val,y_val),callbacks=[early_stop])


import matplotlib.pyplot as plt

plt.plot(history.history['loss'],label="training Loss")
plt.plot(history.history['val_loss'],label="validation Loss")
plt.ylabel("Epoch")
plt.legend()
plt.show()