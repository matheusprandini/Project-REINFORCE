from __future__ import division, print_function
from keras.models import *
from keras.layers.core import Activation, Dense, Flatten, Reshape
from keras.layers.convolutional import Conv2D
from keras.optimizers import Adam
from scipy.misc import imresize
import collections
import numpy as np
import os

IMAGE_SIZE = 84
LEARNING_RATE = 1e-3
NUM_EPOCHS = 1

class ReinforceAgent:

    def __init__(self, num_actions):
        self.model = self.build_model(num_actions)

    # build the model
    def build_model(self, num_actions):
	
		# Sequential Model
        model = Sequential()

        # 1st mlp layer
        model.add(Dense(512, kernel_initializer="normal", input_shape=(IMAGE_SIZE**2,)))
        model.add(Activation("relu"))
		
		# 2st mlp layer
        model.add(Dense(128, kernel_initializer="normal"))
        model.add(Activation("relu"))
		
		# 3st (last) cnn layer -> Classification layer
        model.add(Dense(num_actions, kernel_initializer="normal"))
        model.add(Activation("softmax"))

		
		# Compiling Model
        model.compile(optimizer=Adam(lr=LEARNING_RATE), loss="categorical_crossentropy")

		# Show model details
        model.summary()
		
        return model

    # Preprocess original image (400,400) to (84,84)
    def preprocess_image(self, image):
        
        # single image
        x_t = image
        x_t = imresize(x_t, (IMAGE_SIZE, IMAGE_SIZE, 1)) 
        x_t = x_t.astype("float")
        x_t /= 255.0

        x_t = np.expand_dims(x_t, axis=0)

        return np.reshape(x_t, (1, IMAGE_SIZE*IMAGE_SIZE*1))

    # Pick action stochastically
    def get_action(self, state):
        print(state.shape)
        self.model.predict(state)
        #print(action_probabilities)
        return 0

    def train(self, environment):

        for e in range(NUM_EPOCHS):
            print("Epoch: {:d}".format(e))
            environment.reset()
            game_over = False

            input_t = environment.get_current_frame()
            input_t = self.preprocess_image(input_t)

            while not game_over:
                action = self.get_action(input_t)
                input_tp1, reward, game_over = environment.step(action)
                print(action, reward, game_over)
