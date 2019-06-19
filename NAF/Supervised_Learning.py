from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy as np
import torch


def create_SL_model(state_shape, num_actions, mode):
    model = Sequential()
    model.add(Dense(32, input_shape=(state_shape,), activation='relu'))
    if mode == 'attacker':
        model.add(Dense(num_actions, activation='tanh'))
    else:
        model.add(Dense(num_actions, activation='softmax'))
    model.compile('Rmsprop', 'mse')
    return model

if __name__ == '__main__':
    model = create_SL_model(4, 4, 'vehicle')
    a = model.predict(np.array([20,20,20,20]).reshape(-1,4))
    print(torch.Tensor(a).numpy())

