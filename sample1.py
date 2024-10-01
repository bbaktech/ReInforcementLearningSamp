from operator import gt, le
import random
import gym 
import keras
import numpy as np
import tensorflow as tf
from collections import deque
from keras.src import Sequential 
from keras.src.layers import Dense
from keras.src.optimizers import Adam
import os

env = gym.make("CartPole-v0")
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
batch_size = 32
n_episodes = 3

print(state_size)
print(action_size)
output_dir = "model_output/cartpole/"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

#A deep Q-learning agent
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.learning_rate = 0.001
        self.model = self._build_model()
 
    def _build_model(self):
        model = Sequential() 
        model.add(Dense(32, activation="relu",
                        input_dim=self.state_size))
        model.add(Dense(32, activation="relu"))
        model.add(Dense(self.action_size, activation="linear"))
        model.compile(loss="mse",
                     optimizer=Adam(learning_rate=self.learning_rate))
        return model
 
    def remember(self, state, action, reward, next_state, done): 
        self.memory.append((state, action,
                            reward, next_state, done))

    def train(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            print("action:"+ str(action) + "   reward:"+ str(reward))
            target = reward # if done 
            if not done:    #reward is -10 we need to caliculate
                target = (reward +    
                          self.gamma *
                          np.amax(self.model.predict(next_state,verbose=0)[0]))
            #100% correct        
            target_f = self.model.predict(state,verbose=0)
            target_f[0][action] = target
            print('target_f')
            print(target_f.shape)
            print(target_f)
            # print(target_f.shape)
            print(len(state))
            #target_f shape is (1,2) not a single entry
            self.model.fit(state, target_f, epochs=1) 
        
        if gt (self.epsilon , self.epsilon_min):
                self.epsilon *= self.epsilon_decay

    def act(self, state):
        if le(np.random.rand() , self.epsilon):
            return random.randrange(self.action_size) 
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def save(self, name): 
        self.model.save_weights(name)

#learning agent
agent = DQNAgent(state_size, action_size)

#DQN agent interacting with an OpenAI Gym environment
#n_episodes indecates number of time to play to learn
for e in range(n_episodes):
    state = env.reset()
    state = state[0]
    state = np.reshape(state, [1, state_size]) 
    done = False 
    time = 0
    while not done:
        env.render()
        action = agent.act(state)
#        next_state, reward, done, _ = env.step(action)
        next_state, reward, done, truncated,info = env.step(action)
#        print("next_state")
#        print(next_state)
        reward = reward if not done else -10
        next_state = np.reshape(next_state, [1, state_size]) 
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        if done:
            print("episode: {}/{}, score: {}, e: {:.2}"
                .format(e, n_episodes-1, time, agent.epsilon))
        time += 1
    if len(agent.memory) > batch_size:
        agent.train(batch_size) 
    if e % 50 == 0:
        agent.save(output_dir + "weights_"
                + "{:04d}".format(e) + ".weights.h5")
