from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Activation,Dropout
from tensorflow.keras.optimizers import Adam
from data_preprocessing import preprocess_train_data

def test_bot_model(test_x,test_y):
    model = Sequential()
    model.add(Dense(128,input_shape = (len(test_x[0]),),activation = 'relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64,activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(len(test_y[0]),activation='softmax'))

    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

    history = model.fit(test_x,test_y,batch_size=5,verbose=True)
    model.save('test.h5',history),
print("model file created and saved")

test_x,test_y = preprocess_train_data()
test_bot_model(test_x,test_y)
