import random
from collections import deque

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, transitions):
        self.buffer.append(transitions)

    def sample(self, batch_size, sequential: bool=False):
        if batch_size > len(self.buffer):
            batch_size = len(self.buffer)
        if sequential:
            rand = random.randint(0, len(self.buffer) - batch_size)
            batch = [self.buffer[i] for i in range(rand, rand + batch_size)]
            return zip(*batch)
        else:
            batch = random.sample(self.buffer, batch_size)
            return zip(*batch)
    
    def clear(self):
        self.buffer.clear()

    def __len__(self):
        return len(self.buffer)