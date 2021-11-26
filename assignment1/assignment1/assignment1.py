from matplotlib import pyplot as plt
from gridWorld import gridWorld
import numpy as np
import os

def show_value_function(mdp, V):
    fig = mdp.render(show_state = False, show_reward = False)            
    for k in mdp.states():
        s = k if isinstance(k, tuple) else mdp.legal_states[k]
        fig.axes[0].annotate("{0:.3f}".format(V[k]), (s[1] - 0.1, s[0] + 0.1), size = 40/mdp.board_mask.shape[0])
    plt.show()
    
def show_policy(mdp, PI):
    fig = mdp.render(show_state = False, show_reward = False)
    action_map = {"U": "^", "D": "v", "L": "<-", "R": "->"}
    for k in mdp.states():
        s = k if isinstance(k, tuple) else mdp.legal_states[k]
        if mdp.terminal[s] == 0:
            fig.axes[0].annotate(action_map[PI[k]], (s[1] - 0.1, s[0] + 0.1), size = 100/mdp.board_mask.shape[0])
    plt.show()
    
####################  Problem 1: Value Iteration #################### 

def get_max_V(state, mdp, gamma, V):

    max_val = -np.inf
    best_action = None
    for action in mdp.actions(state):
        sum = 0
        for next_state in mdp.states():
            trans_prob = mdp.transition_probability(state, action, next_state)
            reward = mdp.reward(state)
            disc = gamma*V[next_state]
            sum += trans_prob*(reward + disc)
        if sum >= max_val:
            max_val = sum 
            best_action = action 
    return max_val, best_action



def value_iteration(mdp, gamma, theta = 1e-3):
    # Make a valuefunction, initialized to 0
    """
    YOUR CODE HERE:
    Problem 1a) Implement Value Iteration
    
    Input arguments:
        - mdp     Is the markov decision process, it has some usefull functions given below
        - gamma   Is the discount rate
        - theta   Is a small threshold for determining accuracy of estimation
    
    Some usefull functions of the grid world mdp:
        - mdp.states() returns a list of all states [0, 1, 2, ...]
        - mdp.actions(state) returns list of actions ["U", "D", "L", "R"] if state non-terminal, [] if terminal
        - mdp.transition_probability(s, a, s_next) returns the probability p(s_next | s, a)
        - mdp.reward(state) returns the reward of the state R(s)
    """
    V = np.zeros((len(mdp.states())))

    for state in mdp.states():
        if not len(mdp.actions(state)):
            V[state] = mdp.reward(state)

    while True:
        delta = 0
        for state in mdp.states():
            if not len(mdp.actions(state)):
                # Terminal state
                continue
            v = V[state].copy()
            V[state], _ = get_max_V(state, mdp, gamma, V)
            delta = max(delta, np.abs(v-V[state]))
        if delta < theta:
            return V
    
    

def policy(mdp, V):
    # Initialize the policy list of crrect length
    PI = np.random.choice(env.actions(), len(mdp.states()))
    
    """
    YOUR CODE HERE:
    Problem 1b) Implement Policy function 
    
    Input arguments:
        - mdp Is the markov decision problem
        - V   Is the optimal falue function, found with value iteration
    """
    max_val = -np.inf
    for state in mdp.states():
        val, best_action = get_max_V(state, mdp, gamma, V)
        if val >= max_val:
            PI[state] = best_action
    
    return PI

####################  Problem 2: Policy Iteration #################### 
def policy_evaluation(mdp, gamma, PI, V, theta = 1e-3):   
    """
    YOUR CODE HERE:
    Problem 2a) Implement Policy Evaluation
    
    Input arguments:  
        - mdp   Is the markov decision problem
        - gamma Is discount factor
        - PI    Is current policy
        - V     Is preveous value function guess
        - theta Is small threshold for determining accuracy of estimation
        
    Some useful tips:
        - If you decide to do exact policy evaluation, np.linalg.solve(A, b) can be used
          optionally scipy has a sparse linear solver that can be used
        - If you decide to do exact policy evaluation, note that the b vector simplifies
          since the reward R(s', s, a) is only dependant on the current state s, giving the 
          simplified reward R(s) 
    """
    raise Exception("Not implemented")
        
    return V

def policy_iteration(mdp, gamma):
    # Make a valuefunction, initialized to 0
    V = np.zeros((len(mdp.states())))
    
    # Create an arbitrary policy PI
    PI = np.random.choice(env.actions(), len(mdp.states()))
    
    """
    YOUR CODE HERE:
    Problem 2b) Implement Policy Iteration
    
    Input arguments:  
        - mdp   Is the markov decision problem
        - gamma Is discount factor

    Some useful tips:
        - Use the the policy_evaluation function from the preveous subproblem
    """
    raise Exception("Not implemented")
            
    return PI, V

if __name__ == "__main__":
    """
    Change the parameters below to change the behaveour, and map of the gridworld.
    gamma is the discount rate, while filename is the path to gridworld map. Note that
    this code has been written for python 3.x, and requiers the numpy and matplotlib
    packages

    Available maps are:
        - gridworlds/tiny.json
        - gridworlds/large.json
    """
    gamma   = 0.9
    abs_file_path = os.path.dirname(os.path.abspath(__file__))
    #filname = "gridworlds/tiny.json"
    filname = "gridworlds/large.json"
    filename = os.path.join(abs_file_path, filname)


    # Import the environment from file
    env = gridWorld(filename)

    # Render image
    fig = env.render(show_state = False)
    plt.show()
    
    # Run Value Iteration and render value function and policy
    V = value_iteration(mdp = env, gamma = gamma)
    show_value_function(env, V)
    
    PI = policy(env, V)
    show_policy(env, PI)
    
    # Run Policy Iteration and render value function and policy
    PI, V = policy_iteration(mdp = env, gamma = gamma)
    show_value_function(env, V)
    show_policy(env, PI)
