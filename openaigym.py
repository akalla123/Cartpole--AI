import gym 
import random 
#import tensorflow as tf
import numpy as np
#import keras
from statistics import mean,median
from collections import Counter

learning_rate = 1e-3

env = gym.make('CartPole-v0')
env.reset()

goal_steps = 500
score_requirements = 50
initial_requirements = 10000

def initial_population():
	training_data = []
	score = []
	accepted_scores = []
	for _ in range(initial_requirements):
		score = 0
		scores = []
		game_memory = []
		previous_observation = []
		for _ in range(goal_steps):
			action = env.action_space.sample()
			observation,reward,done,info = env.step(action)
			if len(previous_observation) > 0:
				game_memory.append([previous_observation,action])
			previous_observation = observation
			score = score + reward
			if done:
				break
		if score >= score_requirements:
			accepted_scores.append(score)
			for data in game_memory:
				if data[1] == 1:
					output = 1
				elif data[1] == 0:
					output = 0
				training_data.append([data[0],output])
		env.reset()
		scores.append(score)
	training_data_save = np.array(training_data)
	np.save('training_data.npy', training_data_save)
	print('Average accepted scores: ', mean(accepted_scores))
	print('Median accepted scores: ', median(accepted_scores))
	print('Total accepted scores: ', len(accepted_scores))
	print('Counter: ', Counter(accepted_scores))
	
	return training_data_save	

a = initial_population()

X = []
Y = []
for i in range(0,len(a)):
  X.append(a[i][0])
  Y.append(a[i][1])

X = np.array(X).reshape(-1,len(a[0][0]),1)
Y = np.array(Y)

Y = Y.reshape(-1)

import tensorflow as tf
from keras.models import Sequential
import numpy as np
from keras.layers import Dense, Dropout, LSTM


model = Sequential() 
model.add(LSTM(256, input_shape=(X.shape[1:]), activation='relu',return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(256,activation='relu'))


model.add(Dense(256, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, Y, epochs=3,validation_split=0.1)


from keras.models import load_model

model.save('model.h5') 
