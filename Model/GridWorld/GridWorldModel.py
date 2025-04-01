"""
A defined Gridworld geneartive model for a given agent to use to represent its
beleifs about its transitions, observation noadliteis and state factors dimensions


"""
import numpy as np

from Model.POMDP.POMDP import POMDP
from Model.GridWorld.GridWorldEnv import GridWorldEnv

from pymdp import utils, maths


class GridWorldModel(POMDP):

    def __init__(self, grid_world: GridWorldEnv, agent_yaml):
        self.grid_world = grid_world
        # Set up base state factor information
        self.grid_dims = grid_world.shape

        self.factors = [len(grid_world.loc_list), len(grid_world.reward_locations)]
        self.start_location = grid_world.start_state
        self.trap = grid_world.trap_location

        self.modalities = [len(grid_world.loc_list), 1 + len(grid_world.boundary_locations),
                           1 + len(grid_world.reward_locations)]

        self.controls = agent_yaml["actions"]
        self.num_controls = [len(self.controls), 1, 1]

        observation_offset = agent_yaml["properties"]["observation_offset"]

        self.observation_model = self.observation_model(observation_offset)
        self.learning_observation = self.dir_observation_model(observation_offset)

        transition_offset = agent_yaml["properties"]["transition_offset"]

        self.transition_model = self.transition_model(transition_offset)
        self.learning_transition = self.dir_transition_model(transition_offset)

        self.reward_model = self.reward_model()

        self.prior_model = self.prior_model(self.grid_world.start_state)
        self.learning_prior = self.dir_prior_model()

    def observation_model(self, observation_offset) -> np.ndarray:

        # Define Observation Model P(o|s) shape
        A_m_shapes = [[o_dim] + self.factors for o_dim in
                      self.modalities]  # list of shapes of modality-specific A[m] arrays
        A = utils.obj_array_zeros(A_m_shapes)

        # Define state grid information modalities
        distribution = 1 - observation_offset
        if distribution == 0:
            dis_pint = observation_offset / (A_m_shapes[0][1])
            arr = np.full(A_m_shapes[0][:2], dis_pint)
        else:
            dis_pint = observation_offset / (A_m_shapes[0][1] - 1)
            arr = np.full(A_m_shapes[0][:2], dis_pint)
            np.fill_diagonal(arr, distribution)

        A[0][..., 0] = arr
        A[0][..., 1] = np.eye(self.modalities[0], self.factors[0])

        # Define boundary observations for each grid position

        for x in range(self.grid_dims[0]):
            for y in range(self.grid_dims[1]):
                coord = (y, x)
                if coord in self.grid_world.loc_list:
                    state_index = self.grid_world.loc_list.index((y, x))
                    bound_index = self.grid_world.boundary_locations[coord]
                    A[1][bound_index, state_index, 0] = 1

        goal_index = self.grid_world.loc_list.index(tuple(self.grid_world.reward_locations["Goal"]))

        A[1][0, :, 1] = 1
        A[1][0, goal_index, 1] = 1
        A[2][0,] = 1.0  # default makes Null the most likely observation everywhere
        A[2][0, goal_index, :] = 0.0
        A[2][1, goal_index, 0] = 1.0
        A[2][2, goal_index, 1] = 1.0

        # fill out the contingencies when the agent is in the "BOTTOM" reward location
        if self.trap:
            trap_index = self.grid_world.loc_list.index(tuple(self.grid_world.trap_location))
            A[1][0, trap_index, 1] = 1
            A[2][0, trap_index, :] = 0.0
            A[2][1, trap_index, 0] = 1.0
            A[2][2, trap_index, 1] = 1.0
        return A

    def dir_observation_model(self, observation_offset) -> np.ndarray:
        if observation_offset:
            pA = utils.dirichlet_like(self.observation_model, scale=1)
        else:
            pA = None

        return pA

    def transition_model(self, transition_offset) -> np.ndarray:
        B = utils.obj_array_zeros([[ns, ns, self.num_controls[f]] for f, ns in enumerate(self.factors)])

        # fill out `B[0]` using the
        for action_id, action_label in enumerate(self.controls):

            for curr_state, grid_location in enumerate(self.grid_world.loc_list):

                y, x = grid_location

                if action_label == "UP":
                    next_y = y - 1 if y > 0 else y
                    next_x = x
                elif action_label == "DOWN":
                    next_y = y + 1 if y < (self.grid_dims[0] - 1) else y
                    next_x = x
                elif action_label == "LEFT":
                    next_x = x - 1 if x > 0 else x
                    next_y = y
                elif action_label == "RIGHT":
                    next_x = x + 1 if x < (self.grid_dims[1] - 1) else x
                    next_y = y
                elif action_label == "STAY":
                    next_x = x
                    next_y = y

                new_location = (next_y, next_x)
                if new_location in self.grid_world.loc_list:
                    next_state = self.grid_world.loc_list.index(new_location)
                    B[0][next_state, curr_state, action_id] = 1.0
                else:
                    B[0][curr_state, curr_state, action_id] = 1.0
                # else:
                #     next_state = self.grid_world.loc_list.index(new_location)
                #     B[0][next_state, curr_state, action_id] = 0

        B[1][:, :, 0] = np.eye(self.factors[1], self.factors[1])

        return B

    def dir_transition_model(self, transition_modality) -> np.ndarray:
        pass

    def reward_model(self) -> np.ndarray:
        C = utils.obj_array_zeros(self.modalities)

        # desired_loc_index = self.loc_list.index(self.reward_locations)

        C[2][1] = 2
        if self.trap is not None:
            C[2][2] = -4

        return C

    def prior_model(self, start_location) -> np.ndarray:
        D = utils.obj_array_uniform(self.factors)
        D[0] = utils.onehot(self.grid_world.loc_list.index(self.start_location), len(self.grid_world.loc_list))
        return D

    def dir_prior_model(self) -> np.ndarray:
        # random location is always renadmoly located
        pD = utils.dirichlet_like(self.prior_model, scale=1)
        return pD
