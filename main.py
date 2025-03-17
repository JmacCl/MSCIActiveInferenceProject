import sys

import yaml

from experiment import experiment_set_up


#
#
# grid_dims = [5, 7] # dimensions of the grid (number of rows, number of columns)
# num_grid_points = np.prod(grid_dims) # total number of grid locations (rows X columns)
#
# # create a look-up table `loc_list` that maps linear indices to tuples of (y, x) coordinates
# grid = np.arange(num_grid_points).reshape(grid_dims)
# it = np.nditer(grid, flags=["multi_index"])
#
# loc_list = []
# while not it.finished:
#     loc_list.append(it.multi_index)
#     it.iternext()
#
# # (y, x) coordinate of the first cue's location, and then a list of the (y, x) coordinates of the possible locations of the second cue, and their labels (`L1`, `L2`, ...)
# cue1_location = (2, 0)
#
# cue2_loc_names = ['L1', 'L2', 'L3', 'L4']
# cue2_locations = [(0, 2), (1, 3), (3, 3), (4, 2)]
#
# # names of the reward conditions and their locations
# reward_conditions = ["TOP", "BOTTOM"]
# reward_locations = [(1, 5), (3, 5)]

def main():
    with open(sys.argv[1], 'r') as stream:
        config_data = yaml.safe_load(stream)
    experiment_set_up(config_data)


if __name__ == "__main__":
    main()
