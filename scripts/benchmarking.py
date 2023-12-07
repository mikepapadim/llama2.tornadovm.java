import argparse
import csv
import numpy as np
import re
import subprocess

# Parse command-line arguments
parser = argparse.ArgumentParser(
    description="Run benchmark script with specified number of iterations."
)
parser.add_argument(
    "--num_runs", type=int, default=100, help="Number of runs per input"
)
args = parser.parse_args()

# List of dataset names
datasets = ["stories15M", "stories42M", "stories100M"]

# List of vector configurations
vector_configs = [
    "-Dllama2.VectorFloat16=true",
    "-Dllama2.VectorFloat8=true",
    "-Dllama2.VectorFloat4=true",
    "",
]

# Range of inputs
input_range = [
    i for i in range(50, 501) if 32000 % i == 0
]  # Numbers that divide 32000 with remainder 0

# Number of runs per input
num_runs = args.num_runs

# Loop through each dataset
for dataset in datasets:
    # Loop through each vector configuration
    for vector_config in vector_configs:
        # Define the base command for the current dataset and vector configuration

        # Lists to store achieved tok/s values for each run
        toks_per_sec_values = []
        max_values = []

        # Loop through each input
        for dim in input_range:
            dim_values = []

            base_command = 'tornado --jvm=" {} -Ds0.t0.local.workgroup.size={} " -cp target/tornadovm-llama-gpu-1.0-SNAPSHOT.jar io.github.mikepapadim.Llama2 {}.bin'.format(
                vector_config, dim, dataset
            )
            # Run the command 'num_runs' times
            for _ in range(num_runs):
                # Construct the full command with the current input
                full_command = base_command.format(dim)

                # Run the command and capture the output
                result = subprocess.run(
                    full_command,
                    shell=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                # time.sleep(1)  # Add a delay of 1 second

                # Find the line containing "achieved tok/s:" and extract the value
                match = re.search(r"achieved tok/s:\s*([0-9.]+)", result.stderr)

                if match:
                    tok_per_sec = float(match.group(1))
                    dim_values.append(tok_per_sec)
                else:
                    print(
                        f"Warning: No match found in output for dim={dim}. Output:\n{result.stdout}"
                    )

            if dim_values:
                # Calculate and print the average, geometric mean, and maximum for the current input
                avg_value = sum(dim_values) / len(dim_values)
                geo_mean_value = (np.prod(dim_values)) ** (1 / len(dim_values))
                max_value = max(dim_values)

                print(
                    f"Dataset: {dataset}, Vector Config: {vector_config}, Input: {dim}, "
                    f"Average tok/s: {avg_value}, Geometric mean tok/s: {geo_mean_value}, Max tok/s: {max_value}"
                )

                # Store the values for further analysis or plotting
                toks_per_sec_values.append(dim_values)
                max_values.append(max_value)
            else:
                print(
                    f"Dataset: {dataset}, Vector Config: {vector_config}, Input: {dim}, No valid tok/s values found."
                )

        # Save the results to a CSV file for each dataset and vector configuration
        output_file = f'result_{vector_config.replace("=true", "").replace(".", "")}_{dataset}.csv'
        with open(output_file, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(
                ["Input", "Average tok/s", "Geometric mean tok/s", "Max tok/s"]
            )

            for i, dim in enumerate(input_range):
                if i < len(toks_per_sec_values) and toks_per_sec_values[i]:
                    avg_value = sum(toks_per_sec_values[i]) / len(
                        toks_per_sec_values[i]
                    )
                    geo_mean_value = (np.prod(toks_per_sec_values[i])) ** (
                        1 / len(toks_per_sec_values[i])
                    )
                    max_value = max_values[i]
                    writer.writerow([dim, avg_value, geo_mean_value, max_value])
                else:
                    writer.writerow([dim, "No valid tok/s values found."])
