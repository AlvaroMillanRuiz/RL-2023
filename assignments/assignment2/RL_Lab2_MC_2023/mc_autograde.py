import numpy as np
from collections import defaultdict
from tqdm import tqdm as _tqdm

def tqdm(*args, **kwargs):
    return _tqdm(*args, **kwargs, mininterval=1)  # Safety, do not overflow buffer

class SimpleBlackjackPolicy(object):
    """
    A simple BlackJack policy that sticks with 20 or 21 points and hits otherwise.
    """
    def get_probs(self, states, actions):
        """
        This method takes a list of states and a list of actions and returns a numpy array that contains a probability
        of perfoming action in given state for every corresponding state action pair. 

        Args:
            states: a list of states.
            actions: a list of actions.

        Returns:
            Numpy array filled with probabilities (same length as states and actions)
        """
        # YOUR CODE HERE
        probs = []
        for s, a in zip(states, actions):
            if s[0] < 20 and a == 1:
                prob = 1
            elif s[0] >= 20 and a == 0:
                prob = 1
            else:
                prob = 0
            probs.append(prob)
            
        return np.array(probs)
    
    def sample_action(self, state):
        """
        This method takes a state as input and returns an action sampled from this policy.  

        Args:
            state: current state

        Returns:
            An action (int).
        """
        return int(state[0] < 20)

def sample_episode(env, policy):
    """
    A sampling routine. Given environment and a policy samples one episode and returns states, actions, rewards
    and dones from environment's step function and policy's sample_action function as lists.

    Args:
        env: OpenAI gym environment.
        policy: A policy which allows us to sample actions with its sample_action method.

    Returns:
        Tuple of lists (states, actions, rewards, dones). All lists should have same length. 
        Hint: Do not include the state after the termination in the list of states.
    """
    states = []
    actions = []
    rewards = []
    dones = []
    
    # YOUR CODE HERE
    done = False
    s = env.reset()

    while not done:
        states.append(s)
        a = policy.sample_action(s)
        actions.append(a)
        s, r, done, info = env.step(a)
        rewards.append(r)
        dones.append(done)
    
    return states, actions, rewards, dones

def mc_prediction(env, policy, num_episodes, discount_factor=1.0, sampling_function=sample_episode):
    """
    Monte Carlo prediction algorithm. Calculates the value function
    for a given policy using sampling.
    
    Args:
        env: OpenAI gym environment.
        policy: A policy which allows us to sample actions with its sample_action method.
        num_episodes: Number of episodes to sample.
        discount_factor: Gamma discount factor.
        sampling_function: Function that generates data from one episode.
    
    Returns:
        A dictionary that maps from state -> value.
        The state is a tuple, and the value is a float.
    """

    # Keeps track of current V and count of returns for each state
    # to calculate an update.
    V = defaultdict(float)
    returns_count = defaultdict(float)

    for i in tqdm(range(num_episodes)):
        # Generate an episode using the sampling function
        states, actions, rewards, dones = sampling_function(env, policy)
        visited_states = set()  # To keep track of visited states in the current episode

        # Initialize the return
        G = 0

        # Iterate over the episode in reverse order
        while states:
            state = states.pop()
            reward = rewards.pop()
            G = discount_factor * G + reward

            if state in states:
                continue
            
            returns_count[state] += 1
            V[state] += (1 / returns_count[state]) * (G - V[state])
            
                ## define G and append to returns
    return V

class RandomBlackjackPolicy(object):
    """
    A random BlackJack policy.
    """
    def get_probs(self, states, actions):
        """
        This method takes a list of states and a list of actions and returns a numpy array that contains 
        a probability of perfoming action in given state for every corresponding state action pair. 

        Args:
            states: a list of states.
            actions: a list of actions.

        Returns:
            Numpy array filled with probabilities (same length as states and actions)
        """
        # YOUR CODE HERE
        return np.ones(len(states)) * 0.5
    
    def sample_action(self, state):
        """
        This method takes a state as input and returns an action sampled from this policy.  

        Args:
            state: current state

        Returns:
            An action (int).
        """
        # YOUR CODE HERE
        return np.random.choice([0, 1])

def mc_importance_sampling(env, behavior_policy, target_policy, num_episodes, discount_factor=1.0,
                           sampling_function=sample_episode):
    """
    Monte Carlo prediction algorithm. Calculates the value function
    for a given target policy using behavior policy and ordinary importance sampling.
    
    Args:
        env: OpenAI gym environment.
        behavior_policy: A policy used to collect the data.
        target_policy: A policy whose value function we want to estimate.
        num_episodes: Number of episodes to sample.
        discount_factor: Gamma discount factor.
        sampling_function: Function that generates data from one episode.
    
    Returns:
        A dictionary that maps from state -> value.
        The state is a tuple, and the value is a float.
    """

    # Keeps track of current V and count of returns for each state
    # to calculate an update.
    V = defaultdict(float)
    returns_count = defaultdict(float)
    
    for _ in tqdm(range(num_episodes)):
        states, actions, rewards, dones = sampling_function(env, behavior_policy)
        W = 1.0
        G = 0.0
        for t in range(len(states) - 1, -1, -1):
            state = states[t]
            action = actions[t]
            reward = rewards[t]
            G = discount_factor * G + reward


            returns_count[state] += 1

            W *= target_policy.get_probs([state], [action]) / behavior_policy.get_probs([state], [action]) 
            V[state] += ((G*W - V[state]) / (returns_count[state]))

    return V

def mc_importance_sampling(env, behavior_policy, target_policy, num_episodes, discount_factor=1.0,
                           sampling_function=sample_episode):
    """
    Monte Carlo prediction algorithm. Calculates the value function
    for a given target policy using behavior policy and ordinary importance sampling.
    
    Args:
        env: OpenAI gym environment.
        behavior_policy: A policy used to collect the data.
        target_policy: A policy whose value function we want to estimate.
        num_episodes: Number of episodes to sample.
        discount_factor: Gamma discount factor.
        sampling_function: Function that generates data from one episode.
    
    Returns:
        A dictionary that maps from state -> value.
        The state is a tuple, and the value is a float.
    """

    # Keeps track of current V and count of returns for each state
    # to calculate an update.
    V = defaultdict(float)
    returns_count = defaultdict(float)
    
    for _ in tqdm(range(num_episodes)):
        states, actions, rewards, dones = sampling_function(env, behavior_policy)
        W = 1.0
        G = 0.0
        for t in range(len(states) - 1, -1, -1):
            state = states[t]
            action = actions[t]
            reward = rewards[t]
            G = discount_factor * G + reward


            returns_count[state] += 1

            W *= target_policy.get_probs([state], [action]) / behavior_policy.get_probs([state], [action]) 
            V[state] += ((G(W - V[state])) / (returns_count[state]))

    return V

def mc_importance_sampling(env, behavior_policy, target_policy, num_episodes, discount_factor=1.0,
                           sampling_function=sample_episode):
    """
    Monte Carlo prediction algorithm. Calculates the value function
    for a given target policy using behavior policy and ordinary importance sampling.
    
    Args:
        env: OpenAI gym environment.
        behavior_policy: A policy used to collect the data.
        target_policy: A policy whose value function we want to estimate.
        num_episodes: Number of episodes to sample.
        discount_factor: Gamma discount factor.
        sampling_function: Function that generates data from one episode.
    
    Returns:
        A dictionary that maps from state -> value.
        The state is a tuple, and the value is a float.
    """

    # Keeps track of current V and count of returns for each state
    # to calculate an update.
    V = defaultdict(float)
    returns_count = defaultdict(float)
    
    for _ in tqdm(range(num_episodes)):
        states, actions, rewards, dones = sampling_function(env, behavior_policy)
        W = 1.0
        G = 0.0
        for t in range(len(states) - 1, -1, -1):
            state = states[t]
            action = actions[t]
            reward = rewards[t]
            G = discount_factor * G + reward


            returns_count[state] += 1

            W *= target_policy.get_probs([state], [action]) / behavior_policy.get_probs([state], [action]) 
            V[state] += ((G*W - V[state]) / (returns_count[state]))

    return V
