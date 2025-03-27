
"""
A defined Gridworld geneartive model for a given agent to use to represent its
beleifs about its transitions, observation noadliteis and state factors dimensions


"""

from Model.POMDP.POMDP import POMDP

class GridWorldModel(POMDP):

    def __init__(self, yaml_env):
        # Set up base state factor information
        env = yaml_env["environment"]
        self.grid_dims = env['grid_dimensions']
        self.loc_list, self.num_grid_points = define_grid_space(self.grid_dims)
        self.factors = [self.num_grid_points]
        self.start_location = random.choice(self.loc_list)