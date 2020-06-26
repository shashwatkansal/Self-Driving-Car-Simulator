# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import random
import os




import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable

#Time for neural network

class Network(nn.Module):
    
    def __init__(self, input_size, nb_action):
        super(Network, self).__init__()
        self.input_size = input_size
        self.nb_action = nb_action
        
        #First fully connected layer
        self.fc1 = nn.Linear(input_size, 30)
        #Second Fully connected layer
        self.fc2 = nn.Linear(30, nb_action)
        
    def forward(self, state):
        #rectifier function
        x = F.relu(self.fc1(state))
        q_values = self.fc2(x)
        return q_values
    
# Implementing Experience Replay
        
class ReplayMemory(object):
    def __init__(self, capacity):
        #Max no of transitions in memory of events
        self.capacity = capacity
        self.memory = []
    
    #To append new events into memory
    def push(self, event):
        self.memory.append(event)
        if len(self.memory) > self.capacity:
            del self.memory[0]
        
    #Get samples for our memory before AI car starts
    def sample(self, batch_size):
        samples = zip(*random.sample(self.memory, batch_size))
        #Takes samples, concatenate with respect to first dimension (the states)
        #and makes a pytorch variable with tensor(state, action, rewards) and the
        #gradient together. Necessary for gradient descent so can be differentiated.
        return map(lambda x: Variable(torch.cat(x, 0)))    
        
#Implementing Deep Q Learning
        
class Dqn():
    
    def __init__(self, input_size, nb_action, gamma):
        self.gamma = gamma
        self.reward_window = []
        self.model = Network(input_size, nb_action)
        self.memory = ReplayMemory(100000)
        self.optimizer = optim.Adam(self.model.parameters(), lr = 0.001)
        #3 signals, orientation, -orientation (5dim vector)
        self.last_state = torch.Tensor(input_size).unsqueeze(0)
        #Can be 0, 20,-20
        self.last_action = 0
        self.last_reward = 0
        
    def select_action(self, state):
        #Taking Q values into softmax to activate output neuron
        #No need to include gradient, so do 'volatile = True'
        probs = F.softmax(self.model(Variable(state, volatile=True))*7) 
        #Temperature parameter makes softmax values very skewed, helpful to
        #determine more surely the correct output neuron
        
        #Use binomial/multinomial distribution. Most of the time will select
        #option with highest probability. Then return the resulting neuron output (action)
        action = probs.multinomial()
        return action.data[0,0]
    
    def learn(self, batch_state, batch_next_state, batch_reward, batch_action):
        #We get chosen action into output. Use index 1 as we have an extra column for batch processing
        ##Prediction
        outputs = self.model(batch_state).gather(1, batch_action.unsqueeze(1).squeeze(1))
        ##Getting Target
        next_outputs = self.model(batch_next_state).detach().max(1)[0]
        target = self.gamma*next_outputs + batch_reward
        ##Getting Loss
        td_loss = F.smooth_l1_loss(outputs, target)
        #Make gradient manually 0 so it does not mix gradients between minibatches
        self.optimizer.zero_grad()
        
        #Free memory when 'True' for backpropagation
        td_loss.backward(retain_variables=True)
        #Update weights
        self.optimizer.step()
        
   #Update when reaching new state
    def update(self, reward, new_signal):
        new_state = torch.Tensor(new_signal).float().unsqueeze(0)
        #Last state, new state, last action, last reward
        self.memory.push((self.last_state, new_state,torch.LongTensor[int( self.last_action)], torch.Tensor([ self.last_reward])))
        action = self.select_action(new_state)
        if len(self.memory.memory) > 100:
           batch_state, batch_next_state, batch_reward, batch_action = self.memory.sample(100)
           self.learn(batch_state, batch_next_state, batch_reward, batch_action)
        self.last_action = action
        self.last_state = new_state
        self.last_reward = reward
        self.reward_window.append(reward)
        if len(self.reward_window) > 1000:
            del self.reward_window[0]
        return action
    
    def score(self):
        return sum(self.reward_window)/(len(self.reward_window)+1)

    def save(self):
        torch.save({'state_dict': self.model.state_dict(), 
                    'optimizer': self.optimizer.state_dict()},
                    'last_brain.pth')
    
    def load(self):
        if os.path.isfile('last_brain.pth'):
            print('=> Loading checkpoint...')
            checkpoint = torch.load('last_brain.pth')
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print('Done!')
        else:
            print('No such checkpoint found')
            
        
        
        
        
    
        

    
    
        

