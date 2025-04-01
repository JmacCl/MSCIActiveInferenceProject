import sys

import yaml

# from experiment import experiment_set_up
from new_experiment import experiment_set_up




def main():
    with open(sys.argv[1], 'r') as stream:
        config_data = yaml.safe_load(stream)
    experiment_set_up(config_data)


if __name__ == "__main__":
    main()
