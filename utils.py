import random
from collections import deque

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        # : Add this tuple to self.buffer
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        # : Return a random list of 'batch_size' experiences from the buffer
        return random.sample(self.buffer, batch_size)
        
    def size(self):
        return len(self.buffer)