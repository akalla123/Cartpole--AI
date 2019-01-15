import gym
env = gym.make('CartPole-v0')
import keras
from keras.models import load_model
from keras.utils import CustomObjectScope
from keras.initializers import glorot_uniform
import os
import random
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
with CustomObjectScope({'GlorotUniform': glorot_uniform()}):
  new_model = load_model('model.h5')
scores = []
choices = []
for each_game in range(10):
	print("This is Game:", each_game)
	score = 0
	game_memory = []
	prev_obs = []
	env.reset()
	if each_game % 100 == 0:
		print (each_game)

	for _ in range(500):
		env.render()
		if len(prev_obs) == 0:
			action = random.randrange(0,2)
		else:
			action = np.argmax(new_model.predict(prev_obs.reshape(-1,len(prev_obs),1)))
		choices.append(action)
		
		new_observation,reward,done,info = env.step(action)
		prev_obs = new_observation
		game_memory.append([new_observation,action])
		score = score + reward
		if done:
			break
	print("Your score was: ",score)
