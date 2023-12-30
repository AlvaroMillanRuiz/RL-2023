import numpy as np
import random
import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
from tqdm import tqdm as _tqdm

def tqdm(*args, **kwargs):
    return _tqdm(*args, **kwargs, mininterval=1)  # Safety, do not overflow buffer

class QNetwork(nn.Module):
    
    def __init__(self, num_hidden=128):
        nn.Module.__init__(self)
        self.l1 = nn.Linear(4, num_hidden)
        self.l2 = nn.Linear(num_hidden, 2)

    def forward(self, x):
        x = self.l1(x)
        x = F.relu(x)
        x = self.l2(x)
        return x

class ReplayMemory:
    
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    def push(self, transition):
        # YOUR CODE HERE
        if len(self.memory) == self.capacity:
            self.memory.pop(0)
            
        self.memory.append(transition)

    def sample(self, batch_size):
        # YOUR CODE HERE
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

def get_epsilon(it):
    # YOUR CODE HERE
    if it < 1000:
        epsilon = 1.0 - (0.95 * it / 1000.0)
    else:
        epsilon = 0.05

    epsilon = max(epsilon, 0.05)
    return epsilon

class EpsilonGreedyPolicy(object):
    """
    A simple epsilon greedy policy.
    """
    def __init__(self, Q, epsilon):
        self.Q = Q
        self.epsilon = epsilon
    
    def sample_action(self, obs):
        """
        This method takes a state as input and returns an action sampled from this policy.  

        Args:
            obs: current state

        Returns:
            An action (int).
        """
        # YOUR CODE HERE
        with torch.no_grad():
            obs = torch.tensor(obs, dtype=torch.float32)
            # Calculate Q-values for all actions
            q_values = self.Q(obs)
            # Choose an action using epsilon-greedy strategy
            if torch.rand(1).item() < self.epsilon:
                # Randomly select an action with probability epsilon
                return torch.randint(0, q_values.shape[-1], (1,)).item()
            
            return q_values.argmax().item()
        
    def set_epsilon(self, epsilon):
        self.epsilon = epsilon

def compute_q_vals(Q, states, actions):
    """
    This method returns Q values for given state action pairs.
    
    Args:
        Q: Q-net
        states: a tensor of states. Shape: batch_size x obs_dim
        actions: a tensor of actions. Shape: Shape: batch_size x 1

    Returns:
        A torch tensor filled with Q values. Shape: batch_size x 1.
    """
    # YOUR CODE HERE
    q_values_all_actions = Q(states)
    q_vals = torch.gather(q_values_all_actions, dim=1, index=actions)
    return q_vals

    
def compute_targets(Q, rewards, next_states, dones, discount_factor):
    """
    This method returns targets (values towards which Q-values should move).
    
    Args:
        Q: Q-net
        rewards: a tensor of rewards. Shape: Shape: batch_size x 1
        next_states: a tensor of states. Shape: batch_size x obs_dim
        dones: a tensor of boolean done flags (indicates if next_state is terminal) Shape: batch_size x 1
        discount_factor: discount
    Returns:
        A torch tensor filled with target values. Shape: batch_size x 1.
    """
    with torch.no_grad():
        next_q_values = Q(next_states)
        max_next_q_values = next_q_values.max(dim=1, keepdim=True)[0]
        targets = rewards + discount_factor * max_next_q_values * (1 - dones.float())
        
    return targets

def train(Q, memory, optimizer, batch_size, discount_factor):
    # DO NOT MODIFY THIS FUNCTION
    
    # don't learn without some decent experience
    if len(memory) < batch_size:
        return None

    # random transition batch is taken from experience replay memory
    transitions = memory.sample(batch_size)
    
    # transition is a list of 4-tuples, instead we want 4 vectors (as torch.Tensor's)
    state, action, reward, next_state, done = zip(*transitions)
    
    # convert to PyTorch and define types
    state = torch.tensor(state, dtype=torch.float)
    action = torch.tensor(action, dtype=torch.int64)[:, None]  # Need 64 bit to use them as index
    next_state = torch.tensor(next_state, dtype=torch.float)
    reward = torch.tensor(reward, dtype=torch.float)[:, None]
    done = torch.tensor(done, dtype=torch.uint8)[:, None]  # Boolean
    
    # compute the q value
    q_val = compute_q_vals(Q, state, action)
    with torch.no_grad():  # Don't compute gradient info for the target (semi-gradient)
        target = compute_targets(Q, reward, next_state, done, discount_factor)
    
    # loss is measured from error between current and newly expected Q values
    loss = F.smooth_l1_loss(q_val, target)

    # backpropagation of loss to Neural Network (PyTorch magic)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item() 

def run_episodes(train, Q, policy, memory, env, num_episodes, batch_size, discount_factor, learn_rate):
    
    optimizer = optim.Adam(Q.parameters(), learn_rate)
    
    global_steps = 0  # Count the steps (do not reset at episode start, to compute epsilon)
    episode_durations = []  
    for i in range(num_episodes):
        s = env.reset()
        
        steps = 0
        while True:
            a = policy.sample_action(s)
            # Store this transition in memory:
            s_prime, r, done, _ = env.step(a)
            memory.push((s, a, r, s_prime, done))
            s = s_prime
            
            loss = train(Q, memory, optimizer, batch_size, discount_factor)
            #s = s_prime
            steps += 1
            global_steps += 1
            # Update epsilon
            policy.set_epsilon(get_epsilon(global_steps))
            
            if done:
                if i % 10 == 0:
                    print("{2} Episode {0} finished after {1} steps"
                          .format(i, steps, '\033[92m' if steps >= 195 else '\033[99m'))
                    print("epsilon: ", policy.epsilon)
                episode_durations.append(steps)
                #plot_durations()
                break
    #example how to do it in TD
     #while not done:

<<<<<<< HEAD
            #a = policy.sample_action(s)

            #state_prime, r, done, _ = env.step(a)

            #next_action = policy.sample_action(state_prime)

            #Q_max = np.max(Q[state_prime])

            #Q[s, a] += alpha * (r + discount_factor * Q_max -  Q[s, a])

            #s = state_prime
            
            #i += 1
            #R += r

    #return episode_durations
=======
    return episode_durations
>>>>>>> 991d99720f8706e763851a1eff69d0c6ab6ad23c
