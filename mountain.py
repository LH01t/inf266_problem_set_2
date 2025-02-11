
import numpy as np
from typing import Tuple
import os

script_dir = os.path.dirname(__file__)




class Mountain():
    def __init__(self):
        
        self.actions = ["upforward", "forward", "downforward"]
        self.actions_to_dir = {"upforward":(-1,1), "forward":(0,1),  "downforward":(1,1)} # in terms of (i,j)/(y,x)
        
    def get_map(self)->np.ndarray:
        """
        return: map, a 100 by 31 matrix containing the terrain roughness at each position.  
        """
        return self.map

    def get_direction(self,action:str)->Tuple[int,int]:
        """
        input: 
            action: "upforward", "forward" or "downforward"
        return: (delta i, delta j) or equivalently (delta y, delta x)
        """
        self._check_action(action)
        return self.actions_to_dir[action]
        
        
    def _check_state(self,state:Tuple[int,int]):
        i,j =state
        if (i<0 or i>30):
            raise ValueError(f'Row position, {i} is beyond the gridworld')
        if (j<0 or j>99):
            raise ValueError(f'Column position, {j}  is beyond the gridworld')

    def _check_action(self,action:str):
        if action not in self.actions:
            raise ValueError('Action is wrongly specified. Choose from {0}'.format(self.actions))
    
    def next_state(self,state:Tuple[int,int],action:str):
        self._check_state(state)
        self._check_action(action)

        if action == "forward":
            if(state[1] <= 98): return (state[0],state[1]+1)
            else: return state
        if action == "upforward":
            if(state[0]>0 and state[1] <= 98): return (state[0]-1,state[1]+1)
            elif (state[0]==0 and state[1] <= 98): return (state[0],state[1]+1)
            else: return state
        if action == "downforward":
            if(state[0]<30 and state[1] <= 98): return (state[0]+1,state[1]+1)
            elif (state[0]==30 and state[1] <= 98): return (state[0],state[1]+1)
            else: return state

    def get_time(self,state:Tuple[int,int])->float:
        """
        We want to minimize accumulated time getting accross the terrain. The time it takes the sled to cover one unit of distance on the hill depends directly on how rough
        roughness is equivalent to the time.  

        input: 
            state: position (i,j) or equivalently (y,x)
        return: time, the roughness for the position
        """
        self._check_state(state)
        i,j =state
        if(j <= 98): return self.map[i,j]
        else: return 0

    def get_reward(self,state:Tuple[int,int],action:str)->float:
        """
        We want to minimize accumulated time  getting accross the terrain. The time it takes the sled to cover one unit of distance on the hill depends directly on how rough
        roughness is equivalent to the time. This is equivalent to maximizing the accumulated negative time

        input: 
            state: position (i,j) or equivalently (y,x)
            action: "upforward", "forward" or "downforward"
        return: reward, the negative roughness for the position
        """
        self._check_state(state)
        self._check_action(action)
        next_state = self.next_state(state,action)
        return -self.get_time(next_state)

class Mountain_one(Mountain):
    def __init__(self):
        super().__init__()
        file_path = os.path.join(script_dir, "the_hill.txt")
        self.map = np.genfromtxt(file_path, dtype=float) 

class Mountain_two(Mountain):
    def __init__(self):
        super().__init__()
        
        file_path = os.path.join(script_dir, "the_hill2.txt")
        self.map = np.genfromtxt(file_path, dtype=float) 



print("adad", os.path.dirname(__file__))

import numpy as np
from mountain import Mountain_one  # Assuming Mountain_one loads the_hill.txt

class PolicyEvaluation:
    def __init__(self, env, gamma=1.0, theta=1e-4):
        """
        env: an instance of Mountain_one (or Mountain_two)
        gamma: discount factor (set to 1.0 for a finite horizon)
        theta: small threshold for determining convergence
        """
        self.env = env
        self.gamma = gamma
        self.theta = theta
        self.rows, self.cols = env.get_map().shape
        # Initialize value function: one value per grid cell.
        self.V = np.zeros((self.rows, self.cols))

    def evaluate_policy(self, policy, order):
        """
        policy: a dictionary mapping each state (i,j) to an action string.
        order: 'top_to_bottom' or 'bottom_to_top' for state update ordering.
        """
        # Decide on the order of iterating over rows
        if order == 'top_to_bottom':
            row_indices = range(self.rows) # Iterate from row 0 to rows-1
        elif order == 'bottom_to_top':
            row_indices = range(self.rows - 1, -1, -1) # Iterates from  bottom to top
        else:
            raise ValueError("Ordering must be 'top_to_bottom' or 'bottom_to_top'.")

        iteration = 0 # Keeps track of sweeps through the state space 
        while True: # Until convergence 
            delta = 0 
            for j in range(self.cols - 1):  # For columns 0 to 98 (except the last one)
                for i in row_indices: 
                    state = (i, j) # current state as (row, column)
                    action = policy[state] # get action from policy for the current state 
                    next_state = self.env.next_state(state, action) # computes next state based on current state and action 
                    reward = self.env.get_reward(state, action) # computes the reward for taking action in current state
                    
                    # Bellman update (since transition is deterministic)
                    new_value = reward + self.gamma * self.V[next_state]
                    delta = max(delta, abs(new_value - self.V[state])) # Calculates absolute change in value for given state
                    self.V[state] = new_value # Updates the state value 
            
            iteration += 1
            if delta < self.theta: # checks if maximum change across all states are less than the threshold theta
                break # Exits when function has converged 
        
        print(f"Policy evaluation converged after {iteration} iterations.")
        return self.V

# Example usage:
env = Mountain_one()  # Load the mountain terrain
# Define a policy: π_str (always go "forward")
policy_str = {(i, j): "forward" for i in range(31) for j in range(99)}
# For the last column (j == 99), the value is terminal; you may not need to assign an action.
# Create evaluator and run policy evaluation.
evaluator = PolicyEvaluation(env, gamma=1.0, theta=1e-4)
V_top = evaluator.evaluate_policy(policy_str, order='top_to_bottom')
V_bottom = evaluator.evaluate_policy(policy_str, order='bottom_to_top')


