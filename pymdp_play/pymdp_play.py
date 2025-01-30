import numpy as np
import pymdp
from pymdp.agent import Agent

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.cm as cm

from GridWorldEnv import  GridWorldEnv
from GenerativeModel import GenerativeModel

grid_dims = [5, 7] # dimensions of the grid (number of rows, number of columns)
num_grid_points = np.prod(grid_dims) # total number of grid locations (rows X columns)

# create a look-up table `loc_list` that maps linear indices to tuples of (y, x) coordinates
grid = np.arange(num_grid_points).reshape(grid_dims)
it = np.nditer(grid, flags=["multi_index"])

loc_list = []
while not it.finished:
    loc_list.append(it.multi_index)
    it.iternext()

# (y, x) coordinate of the first cue's location, and then a list of the (y, x) coordinates of the possible locations of the second cue, and their labels (`L1`, `L2`, ...)
cue1_location = (2, 0)

cue2_loc_names = ['L1', 'L2', 'L3', 'L4']
cue2_locations = [(0, 2), (1, 3), (3, 3), (4, 2)]

# names of the reward conditions and their locations
reward_conditions = ["TOP", "BOTTOM"]
reward_locations = [(1, 5), (3, 5)]

# list of dimensionalities of the hidden states -- useful for creating generative model later on
num_states = [num_grid_points, len(cue2_locations), len(reward_conditions)]

# Names of the cue1 observation levels, the cue2 observation levels, and the reward observation levels
cue1_names = ['Null'] + cue2_loc_names # signals for the possible Cue 2 locations, that only are seen when agent is visiting Cue 1
cue2_names = ['Null', 'reward_on_top', 'reward_on_bottom']
reward_names = ['Null', 'Cheese', 'Shock']

num_obs = [num_grid_points, len(cue1_names), len(cue2_names), len(reward_names)]

# initialize `num_controls`
num_controls = [5, 1, 1]
actions = ["UP", "DOWN", "LEFT", "RIGHT", "STAY"]
generative_model = GenerativeModel(num_states, num_obs, num_controls)

(A, B, C, D) = generative_model.return_gen_model()

my_agent = Agent(A = A, B = B, C = C, D = D, policy_len = 4)

my_env = GridWorldEnv(starting_loc = (0,0), cue1_loc = (2, 0), cue2 = 'L4', reward_condition = 'BOTTOM')

loc_obs, cue1_obs, cue2_obs, reward_obs = my_env.reset()

history_of_locs = [loc_obs]
obs = [loc_list.index(loc_obs), cue1_names.index(cue1_obs), cue2_names.index(cue2_obs), reward_names.index(reward_obs)]

T = 10  # number of total timesteps

for t in range(T):
    qs = my_agent.infer_states(obs)

    my_agent.infer_policies()
    chosen_action_id = my_agent.sample_action()

    movement_id = int(chosen_action_id[0])

    choice_action = actions[movement_id]

    print(f'Action at time {t}: {choice_action}')

    loc_obs, cue1_obs, cue2_obs, reward_obs = my_env.step(choice_action)

    obs = [loc_list.index(loc_obs), cue1_names.index(cue1_obs), cue2_names.index(cue2_obs),
           reward_names.index(reward_obs)]

    history_of_locs.append(loc_obs)

    print(f'Grid location at time {t}: {loc_obs}')

    print(f'Reward at time {t}: {reward_obs}')

print(my_agent.qs)

all_locations = np.vstack(history_of_locs).astype(
    float)  # create a matrix containing the agent's Y/X locations over time (each coordinate in one row of the matrix)

fig, ax = plt.subplots(figsize=(10, 6))

# create the grid visualization
X, Y = np.meshgrid(np.arange(grid_dims[1]+1), np.arange(grid_dims[0]+1))
h = ax.pcolormesh(X, Y, np.ones(grid_dims), edgecolors='k', vmin = 0, vmax = 30, linewidth=3, cmap = 'coolwarm')
ax.invert_yaxis()

# Put gray boxes around the possible reward locations
reward_top = ax.add_patch(patches.Rectangle((reward_locations[0][1],reward_locations[0][0]),1.0,1.0,linewidth=5,edgecolor=[0.5, 0.5, 0.5],facecolor=[0.5, 0.5, 0.5]))
reward_bottom = ax.add_patch(patches.Rectangle((reward_locations[1][1],reward_locations[1][0]),1.0,1.0,linewidth=5,edgecolor=[0.5, 0.5, 0.5],facecolor=[0.5, 0.5, 0.5]))

text_offsets = [0.4, 0.6]

cue_grid = np.ones(grid_dims)
cue_grid[cue1_location[0],cue1_location[1]] = 15.0
for ii, loc_ii in enumerate(cue2_locations):
  row_coord, column_coord = loc_ii
  cue_grid[row_coord, column_coord] = 5.0
  ax.text(column_coord+text_offsets[0], row_coord+text_offsets[1], cue2_loc_names[ii], fontsize = 15, color='k')
h.set_array(cue_grid.ravel())

plt.show()

