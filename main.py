import sys
import yaml
from experiment_script import experiment_set_up




def main():
    with open(sys.argv[1], 'r') as stream:
        config_data = yaml.safe_load(stream)
    experiment_set_up(config_data)


if __name__ == "__main__":
    main()
