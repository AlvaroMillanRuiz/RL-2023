import numpy as np
from collections import defaultdict

def policy_eval_v(policy, env, discount_factor=1.0, theta=0.00001):
    """
    Evaluate a policy given an environment and a full description of the environment's dynamics.
    
    Args:
        policy: [S, A] shaped matrix representing the policy.
        env: OpenAI env. env.P represents the transition probabilities of the environment.
            env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
            env.nS is a number of states in the environment. 
            env.nA is a number of actions in the environment.
        theta: We stop evaluation once our value function change is less than theta for all states.
        discount_factor: Gamma discount factor.
    
    Returns:
        Vector of length env.nS representing the value function.
    """
    # Start with an all 0 value function
    V = np.zeros(env.nS)
    
    while True:
        # Initialize a variable to track the maximum change in value function
        delta = 0.0
        
        # Loop over all states
        for s in range(env.nS):
            # Store the current value estimate for state s
            v = V[s]
            
            # Initialize a variable to accumulate the new value estimate for state s
            new_v = 0
            
            # Loop over all possible actions
            for a, action_prob in enumerate(policy[s]):
                # Loop over all possible outcomes of the action
                for prob, next_state, reward, done in env.P[s][a]:
                    # Update the new value estimate for state s
                    if done:
                        # If the next_state is a terminal state, only consider the reward
                        new_v += action_prob * prob * reward
                    else:
                        new_v += action_prob * prob * (reward + discount_factor * V[next_state])
            
            # Update the value function for state s
            V[s] = new_v
            
            # Update the maximum change in value function
            delta = max(delta, abs(v - V[s]))
        
        # Check if the maximum change is smaller than the threshold theta
        if delta < theta:
            break
    
    
    return V

    

def policy_iter_v(env, policy_eval_v=policy_eval_v, discount_factor=1.0):
    """
    Policy Iteration Algorithm. Iteratively evaluates and improves a policy
    until an optimal policy is found.
    
    Args:
        env: The OpenAI environment.
        policy_eval_v: Policy Evaluation function that takes 3 arguments:
            policy, env, discount_factor.
        discount_factor: Gamma discount factor.
        
    Returns:
        A tuple (policy, V). 
        policy is the optimal policy, a matrix of shape [S, A] where each state s
        contains a valid probability distribution over actions.
        V is the value function for the optimal policy.
    """
    # Start with a random policy
    policy = np.ones([env.nS, env.nA]) / env.nA
    
    while True:
        # Step 1: Policy Evaluation
        V = policy_eval_v(policy, env, discount_factor)
        
        # Initialize a variable to check if the policy is stable
        policy_stable = True
        
        # Step 2: Policy Improvement
        for s in range(env.nS):
            # Store the old action for state s
            old_action = np.argmax(policy[s])
            
            # Compute the Q-values for all actions in state s
            q_values = np.zeros(env.nA)
            for a in range(env.nA):
                for prob, next_state, reward, done in env.P[s][a]:
                    q_values[a] += prob * (reward + discount_factor * V[next_state])
            
            # Update the policy for state s to select the action that maximizes Q-value
            best_action = np.argmax(q_values)
            policy[s] = np.zeros((env.nA,))
            policy[s][best_action] = 1
            
            # Check if the old action and new action are different
            if old_action != best_action:
                policy_stable = False
        
        # If the policy is stable, break from the loop
        if policy_stable:
            break
    
    
    # Return the optimal policy and value function
    return policy, V



import numpy as np

def value_iter_q(env, theta=0.0001, discount_factor=1.0):
    """
    Q-value Iteration Algorithm.
    
    Args:
        env: OpenAI env. env.P represents the transition probabilities of the environment.
            env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
            env.nS is a number of states in the environment. 
            env.nA is a number of actions in the environment.
        theta: We stop evaluation once our value function change is less than theta for all state-action pairs.
        discount_factor: Gamma discount factor.
        
    Returns:
        A tuple (policy, Q) of the optimal policy and the optimal Q-value function.
    """
    
    # Initialize Q-value function with zeros
    Q = np.zeros((env.nS, env.nA))
    
    while True:
        delta = 0  # Initialize the change in Q-values
        
        # Loop over all states
        for s in range(env.nS):
            for a in range(env.nA):
                temp = Q[s, a]  # Store the current Q-value
                
                # Compute the Q-value using the Bellman equation
                Q[s][a] = sum(prob * (reward + discount_factor * np.max(Q[next_state])) for prob, next_state, reward, done in env.P[s][a])
                
                # Update the change in Q-values
                delta = max(delta, abs(temp - Q[s, a]))
        
        # Check if the change in Q-values is smaller than the threshold theta
        if delta < theta:
            break
    
    # Extract the optimal policy from the Q-values
    policy = np.zeros((env.nS, env.nA))
    for s in range(env.nS):
        best_action = np.argmax(Q[s])
        policy[s, best_action] = 1.0
    
    return policy, Q

import numpy as np

def value_iter_q(env, theta=0.0001, discount_factor=1.0):
    """
    Q-value Iteration Algorithm.
    
    Args:
        env: OpenAI env. env.P represents the transition probabilities of the environment.
            env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
            env.nS is a number of states in the environment. 
            env.nA is a number of actions in the environment.
        theta: We stop evaluation once our value function change is less than theta for all state-action pairs.
        discount_factor: Gamma discount factor.
        
    Returns:
        A tuple (policy, Q) of the optimal policy and the optimal Q-value function.
    """
    
    # Initialize Q-value function with zeros
    Q = np.zeros((env.nS, env.nA))
    
    while True:
        delta = 0  # Initialize the change in Q-values
        
        # Loop over all states
        for s in range(env.nS):
            for a in range(env.nA):
                temp = Q[s, a]  # Store the current Q-value
                
                # Compute the Q-value using the Bellman equation
                for prob, next_state, reward, done in env.P[s][a]:
                    Q[s, a] = sum(prob * (reward + discount_factor * np.max(Q[next_state])))
                
                # Update the change in Q-values
                delta = max(delta, abs(temp - Q[s, a]))
        
        # Check if the change in Q-values is smaller than the threshold theta
        if delta < theta:
            break
    
    # Extract the optimal policy from the Q-values
    policy = np.zeros((env.nS, env.nA))
    for s in range(env.nS):
        best_action = np.argmax(Q[s])
        policy[s, best_action] = 1.0
    
    return policy, Q

import numpy as np

def value_iter_q(env, theta=0.0001, discount_factor=1.0):
    """
    Q-value Iteration Algorithm.
    
    Args:
        env: OpenAI env. env.P represents the transition probabilities of the environment.
            env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
            env.nS is a number of states in the environment. 
            env.nA is a number of actions in the environment.
        theta: We stop evaluation once our value function change is less than theta for all state-action pairs.
        discount_factor: Gamma discount factor.
        
    Returns:
        A tuple (policy, Q) of the optimal policy and the optimal Q-value function.
    """
    
    # Initialize Q-value function with zeros
    Q = np.zeros((env.nS, env.nA))
    
    while True:
        delta = 0  # Initialize the change in Q-values
        
        # Loop over all states
        for s in range(env.nS):
            for a in range(env.nA):
                temp = Q[s, a]  # Store the current Q-value
                
                # Compute the Q-value using the Bellman equation
                new_q_value = 0.0
                for prob, next_state, reward, done in env.P[s][a]:
                    new_q_value += prob * (reward + discount_factor * np.max(Q[next_state]))
                
                Q[s, a] = new_q_value
                
                # Update the change in Q-values
                delta = max(delta, abs(temp - Q[s, a]))
        
        # Check if the change in Q-values is smaller than the threshold theta
        if delta < theta:
            break
    
    # Extract the optimal policy from the Q-values
    policy = np.zeros((env.nS, env.nA))
    for s in range(env.nS):
        best_action = np.argmax(Q[s])
        policy[s, best_action] = 1.0
    
    return policy, Q
