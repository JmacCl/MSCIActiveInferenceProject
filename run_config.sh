# shellcheck disable=SC1104
# shellcheck disable=SC1128
#!/bin/bash

# Define the directory containing YAML config files
CONFIG_DIR="./configs"

# Define the Python script to run
PYTHON_SCRIPT="main.py"

# Loop through each YAML file and run the Python script
for config in "$CONFIG_DIR"/*.yaml; do
    echo "Running with config: $config"
    python "$PYTHON_SCRIPT" "$config"
done

echo "All runs completed!"