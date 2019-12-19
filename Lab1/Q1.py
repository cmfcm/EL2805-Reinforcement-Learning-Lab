import numpy as np
import matplotlib.pyplot as plt
import time
from IPython import display
import random
import os

# Implemented methods
methods = ['DynProg', 'ValIter']

# Some colours
LIGHT_RED    = '#FFC4CC'
LIGHT_GREEN  = '#95FD99'
BLACK        = '#000000'
WHITE        = '#FFFFFF'
LIGHT_PURPLE = '#E8D0FF'
LIGHT_ORANGE = '#FAE0C3'

class Maze:

    # Actions
    STAY       = 0
    MOVE_LEFT  = 1
    MOVE_RIGHT = 2
    MOVE_UP    = 3
    MOVE_DOWN  = 4

    # Give names to actions
    actions_names = {
        STAY: "stay",
        MOVE_LEFT: "move left",
        MOVE_RIGHT: "move right",
        MOVE_UP: "move up",
        MOVE_DOWN: "move down"
    }

    # Reward values
    STEP_REWARD = -1
    GOAL_REWARD = 0
    IMPOSSIBLE_REWARD = -100


    def __init__(self, maze, still=False):
        """ Constructor of the environment Maze.
        """
        self.still = still
        self.maze                     = maze
        self.actions                  = self.__actions(still=True)
        self.actions_m                = self.__actions(still = self.still)
        self.states, self.map         = self.__states()
        self.n_actions                = len(self.actions)
        self.n_actions_m              = len(self.actions_m)
        self.n_states                 = len(self.states)
        self.transition_probabilities = self.__transitions()
        self.rewards                  = self.cal_rewards()
        

    def __actions(self, still=False):
        actions = dict()
        if still:
            actions[self.STAY]       = (0, 0)
        actions[self.MOVE_LEFT]  = (0,-1)
        actions[self.MOVE_RIGHT] = (0, 1)
        actions[self.MOVE_UP]    = (-1,0)
        actions[self.MOVE_DOWN]  = (1,0)
        return actions
  

    # Define the state with size (maze.width * maze.height) ^ 2, which contains both A and B position
    def __states(self):   
        states = dict()
        map = dict()
        end = False
        s = 0
        # first consider A's position
        for i in range(self.maze.shape[0]):
            for j in range(self.maze.shape[1]):
                # A can't go through the wall, can ignore this state
                if self.maze[i,j] != 1:
                    # consider B's position
                    for m in range(self.maze.shape[0]):
                        for n in range(self.maze.shape[1]):
                            states[s] = (i,j,m,n) # i,j is A's pos, m,n is B's pos
                            map[(i,j,m,n)] = s
                            s += 1
        return states, map

    def __move(self, state, action, return_all_possible = False):                       #, action_m
        """ Makes a step in the maze, given a current position and an action.
            If the action STAY or an inadmissible action is used, the agent stays in place.
            :return tuple next_cell: Position (x,y) on the maze that agent transitions to.
        """
        # Compute the future position given current (state, action)
        row = self.states[state][0] + self.actions[action][0]
        col = self.states[state][1] + self.actions[action][1]
        # print(action_m)
        possible_actions = []
        if self.still:
            possible_actions.append(self.STAY)
        if self.states[state][2] - 1 >= 0:
            possible_actions.append(self.MOVE_UP)
        if self.states[state][2] + 1 < self.maze.shape[0]:
            possible_actions.append(self.MOVE_DOWN)
        if self.states[state][3] - 1 >= 0:
            possible_actions.append(self.MOVE_LEFT)
        if self.states[state][3] + 1 < self.maze.shape[1]:
            possible_actions.append(self.MOVE_RIGHT)

        # Is the future position an impossible one ?
        hitting_maze_walls =  (row == -1) or (row == self.maze.shape[0]) or \
                              (col == -1) or (col == self.maze.shape[1]) or \
                              (self.maze[row,col] == 1)

        if hitting_maze_walls:
            row = self.states[state][0]
            col = self.states[state][1]

        if return_all_possible:
            s = []
            for a_m in possible_actions:
                row_m = self.states[state][2] + self.actions_m[a_m][0]
                col_m = self.states[state][3] + self.actions_m[a_m][1]
                s.append(self.map[(row, col, row_m, col_m)])
            return s
        else:
            a_m = random.choice(possible_actions)
            row_m = self.states[state][2] + self.actions_m[a_m][0]
            col_m = self.states[state][3] + self.actions_m[a_m][1]
            return self.map[(row, col, row_m, col_m)]

    def __transitions(self):
        """ Computes the transition probabilities for every state action pair.
            :return numpy.tensor transition probabilities: tensor of transition
            probabilities of dimension S*S*A
        """
        # Initialize the transition probailities tensor (S,S,A)
        dimensions = (self.n_states,self.n_states,self.n_actions)
        transition_probabilities = np.zeros(dimensions)

        for s in range(self.n_states):
            for a in range(self.n_actions):
                possible_states = self.__move(s, a, return_all_possible=True)
                p = 1 / len(possible_states)
                for next_s in possible_states:
                    transition_probabilities[next_s, s, a] = p

        return transition_probabilities

    def cal_rewards(self):
        if self.still:
            monster_possible_action = range(5)
            len_monster_possible_action = 5
        else:
            monster_possible_action = range(1,5)
            len_monster_possible_action = 4 

        rewards = np.zeros((self.n_states, self.n_actions))
        
        for s in range(self.n_states):
            for a in range(self.n_actions):
                possible_states = self.__move(s, a, return_all_possible=True)
                tmp = 0
                for next_s in possible_states:
                    # reward for meeting minotaur
                    if (self.states[next_s][0], self.states[next_s][1]) == (self.states[next_s][2], self.states[next_s][3]):
                        tmp += self.IMPOSSIBLE_REWARD
                    # reward for hitting a wall
                    elif (self.states[s][0], self.states[s][1]) == (self.states[next_s][0], self.states[next_s][1]) and a != self.STAY:
                        tmp += self.IMPOSSIBLE_REWARD
                    # reward for reaching the exit
                    elif (self.states[s][0], self.states[s][1]) == (self.states[next_s][0], self.states[next_s][1]) and self.maze[(self.states[next_s][0], self.states[next_s][1])] == 2:
                        tmp += self.GOAL_REWARD
                    else:
                        tmp += self.STEP_REWARD
                    
                    rewards[s, a] = tmp / len(possible_states)    
        
        return rewards

    def simulate(self, start, policy, method):
        if method not in methods:
            error = 'ERROR: the argument method must be in {}'.format(methods)
            raise NameError(error)

        path = list()
        if method == 'DynProg':
            # Deduce the horizon from the policy shape
            horizon = policy.shape[1]
            # Initialize current state and time
            t = 0
            s = self.map[start]
            # Add the starting position in the maze to the path
            path.append(start)
            while t < horizon-1:
                # Get minotaur
                # next_action_m = self.get_minotaur_action(still=False)
                # Move to next state given the policy and the current state
                # next_s = self.__move(s,policy[s,t], next_action_m)
                next_s = self.__move(s, policy[s, t])
                    
                if self.states[next_s][0] == self.states[next_s][2] and self.states[next_s][1] == self.states[next_s][3]:
                    path.append(self.states[next_s])
#                     print('Eaten by Minotaur!')
                    return []
                elif self.maze[(self.states[next_s][0], self.states[next_s][1])] == 2:
                    path.append(self.states[next_s])
                    return path

                # Add the position in the maze corresponding to the next state to the path
                path.append(self.states[next_s])
                # Update time and state for next iteration
                t +=1
                s = next_s
        if method == 'ValIter':
            # Initialize current state, next state and time
            t = 1
            s = self.map[start]
            # Add the starting position in the maze to the path
            path.append(start)
            # Get minotaur
            # next_action_m = self.get_minotaur_action(still=False)
            # Move to next state given the policy and the current state
            # next_s = self.__move(s,policy[s], next_action_m)
            next_s = self.__move(s, policy[s])#, t

            if self.states[next_s][0] == self.states[next_s][2] and self.states[next_s][1] == self.states[next_s][3]:
                    path.append(self.states[next_s])
                    print('Eaten by Minotaur!')
                    return path

            # Add the position in the maze corresponding to the next state to the path
            path.append(self.states[next_s])
            # Loop while state is not the goal state
            # while [self.states[s][0], self.states[s][1]] != self.exit:
            while self.maze[(self.states[s][0], self.states[s][1])] != 2:
                # Update state
                s = next_s
                # Get minotaur
                next_action_m = self.get_minotaur_action(still=False)
                # Move to next state given the policy and the current state
#                 next_s = self.__move(s,policy[s], next_action_m)
                next_s = self.__move(s,policy[s])
                # Add the position in the maze corresponding to the next state to the path
                path.append(self.states[next_s])
                # Update time and state for next iteration
                t +=1
                # print(path)        

        return path[:-1]

        

def dynamic_programming(env, horizon):
    
    # The dynamic prgramming requires the knowledge of :
    # - Transition probabilities
    # - Rewards
    # - State space
    # - Action space
    # - The finite horizon
    p         = env.transition_probabilities
    r         = env.rewards
    n_states  = env.n_states
    n_actions = env.n_actions
    T         = horizon

    # The variables involved in the dynamic programming backwards recursions
    V      = np.zeros((n_states, T+1))
    policy = np.zeros((n_states, T+1))
    Q      = np.zeros((n_states, n_actions))


    # Initialization
    Q            = np.copy(r)
    V[:, T]      = np.max(Q,1)
    policy[:, T] = np.argmax(Q,1)

    # The dynamic programming bakwards recursion
    for t in range(T-1,-1,-1): 
        # Update the value function acccording to the bellman equation
        for s in range(n_states):
            for a in range(n_actions):
                # Update of the temporary Q values
                Q[s,a] = r[s,a] + np.dot(p[:,s,a],V[:,t+1])
        # Update by taking the maximum Q value w.r.t the action a
        V[:,t] = np.max(Q,1)
        # The optimal action is the one that maximizes the Q function
        policy[:,t] = np.argmax(Q,1)
    return V, policy


def value_iteration(env, gamma, epsilon):
    """ 
        :input Maze env           : The maze environment in which we seek to
                                    find the shortest path.
        :input float gamma        : The discount factor.
        :input float epsilon      : accuracy of the value iteration procedure.
        :return numpy.array V     : Optimal values for every state at every
                                    time, dimension S*T
        :return numpy.array policy: Optimal time-varying policy at every state,
                                    dimension S*T
    """
    # The value itearation algorithm requires the knowledge of :
    # - Transition probabilities
    # - Rewards
    # - State space
    # - Action space
    # - The finite horizon
    p         = env.transition_probabilities
    r         = env.rewards
    n_states  = env.n_states
    n_actions = env.n_actions

    # Required variables and temporary ones for the VI to run
    V   = np.zeros(n_states)
    Q   = np.zeros((n_states, n_actions))
    BV  = np.zeros(n_states)
    # Iteration counter
    n   = 0
    # Tolerance error
    tol = (1 - gamma)* epsilon/gamma

    # Initialization of the VI
    for s in range(n_states):
        for a in range(n_actions):
            Q[s, a] = r[s, a] + gamma*np.dot(p[:,s,a],V)
    BV = np.max(Q, 1)

    # Iterate until convergence
    while np.linalg.norm(V - BV) >= tol and n < 200:
        print("Iteration: ", n)
        # Increment by one the numbers of iteration
        n += 1
        # Update the value function
        V = np.copy(BV)
        # Compute the new BV
        for s in range(n_states):
            for a in range(n_actions):
                Q[s, a] = r[s, a] + gamma*np.dot(p[:,s,a],V)
        BV = np.max(Q, 1)
        # Show error
        #print(np.linalg.norm(V - BV))

    # Compute policy
    policy = np.argmax(Q,1)
    # Return the obtained policy
    return V, policy


def draw_maze(maze):

    # Map a color to each cell in the maze
    col_map = {0: WHITE, 1: BLACK, 2: LIGHT_GREEN, -6: LIGHT_RED, -1: LIGHT_RED};

    # Give a color to each cell
    rows,cols    = maze.shape
    colored_maze = [[col_map[maze[j,i]] for i in range(cols)] for j in range(rows)];

    # Create figure of the size of the maze
    fig = plt.figure(1, figsize=(cols,rows))

    # Remove the axis ticks and add title title
    ax = plt.gca()
    ax.set_title('The Maze')
    ax.set_xticks([])
    ax.set_yticks([])

    # Give a color to each cell
    rows,cols    = maze.shape
    colored_maze = [[col_map[maze[j,i]] for i in range(cols)] for j in range(rows)]

    # Create figure of the size of the maze
    fig = plt.figure(1, figsize=(cols,rows))

    # Create a table to color
    grid = plt.table(cellText=None,
                            cellColours=colored_maze,
                            cellLoc='center',
                            loc=(0,0),
                            edges='closed')
    # Modify the hight and width of the cells in the table
    tc = grid.properties()['child_artists']
    for cell in tc:
        cell.set_height(1.0/rows)
        cell.set_width(1.0/cols)


def animate_solution(maze, path):

    # Map a color to each cell in the maze
    col_map = {0: WHITE, 1: BLACK, 2: LIGHT_GREEN, -6: LIGHT_RED, -1: LIGHT_RED};

    # Size of the maze
    rows,cols = maze.shape

    # Create figure of the size of the maze
    fig = plt.figure(1, figsize=(cols,rows))

    # Remove the axis ticks and add title title
    ax = plt.gca()
    ax.set_title('Policy simulation')
    ax.set_xticks([])
    ax.set_yticks([])

    # Give a color to each cell
    colored_maze = [[col_map[maze[j,i]] for i in range(cols)] for j in range(rows)]

    # Create figure of the size of the maze
    fig = plt.figure(1, figsize=(cols,rows))

    # Create a table to color
    grid = plt.table(cellText=None,
                     cellColours=colored_maze,
                     cellLoc='center',
                     loc=(0,0),
                     edges='closed')

    # Modify the hight and width of the cells in the table
    tc = grid.properties()['child_artists']
    for cell in tc:
        cell.set_height(1.0/rows)
        cell.set_width(1.0/cols)


    # Update the color at each frame
    for i in range(len(path)):
        grid.get_celld()[(path[i][0], path[i][1])].set_facecolor(LIGHT_ORANGE)
        grid.get_celld()[(path[i][0], path[i][1])].get_text().set_text('Player')
        grid.get_celld()[(path[i][2], path[i][3])].set_facecolor(LIGHT_PURPLE)
        grid.get_celld()[(path[i][2], path[i][3])].get_text().set_text('Minotaur')
        if i > 0:
            if maze[path[i][0]][path[i][1]] == 2:#(, ) == (path[i-1][0], path[i-1][1])
                grid.get_celld()[(path[i][0], path[i][1])].set_facecolor(LIGHT_GREEN)
                grid.get_celld()[(path[i][0], path[i][1])].get_text().set_text('Player is out')
                grid.get_celld()[(path[i-1][2], path[i-1][3])].set_facecolor(col_map[maze[path[i-1][2], path[i-1][3]]])
                grid.get_celld()[(path[i-1][2], path[i-1][3])].get_text().set_text('')
            else:
                grid.get_celld()[(path[i-1][2], path[i-1][3])].set_facecolor(col_map[maze[path[i-1][2], path[i-1][3]]])
                grid.get_celld()[(path[i-1][2], path[i-1][3])].get_text().set_text('')
                grid.get_celld()[(path[i-1][0], path[i-1][1])].set_facecolor(col_map[maze[path[i-1][0], path[i-1][1]]])
                grid.get_celld()[(path[i-1][0], path[i-1][1])].get_text().set_text('')
        display.display(fig)
        display.clear_output(wait=True)
        time.sleep(1)


minotaur_still_flag = True
mode = 'DP'
T = 30

maze_map = [[0,0,1,0,0,0,0,0],[0,0,1,0,0,1,0,0],[0,0,1,0,0,1,1,1], \
    [0,0,1,0,0,1,0,0],[0,0,0,0,0,0,0,0],[0,1,1,1,1,1,1,0],[0,0,0,0,1,2,0,0]]

maze = Maze(np.array(maze_map) , minotaur_still_flag)



if mode == 'VI':
    if not os.path.exists('VI_policy.npy'):    
        
        discount =  1- 1/30
        value, policy = value_iteration(maze, discount, 0.0001)

        np.save('VI_policy.npy', np.array(policy))
    else: 
        policy =  np.load('VI_policy.npy') 


    path = maze.simulate((0,0,6,5), policy,'ValIter')



if mode == 'DP':
    
    if not os.path.exists('DP_policy.npy'):    
        value, policy = dynamic_programming(maze, T)

        np.save('DP_policy.npy',policy)
    else: 
        policy = np.load('DP_policy.npy')

    path = maze.simulate((0,0,6,5), policy,'DynProg')


print(path)

player = []
for i in range(len(path)):
    player.append((path[i][0],path[i][1]))




cnt = 0
for i in range(10000):
    # if i % 100 == 0:
    #     print(i) 
    path = maze.simulate((0,0,6,5), policy,'DynProg')
    if (path[-1][0], path[-1][1]) == (6,5):
        cnt += 1
print("count: " , str(cnt))        


# animate_solution(np.array(maze_map), path)


    
