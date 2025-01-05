import os
import json
import argparse
import re

# Set up argument parser
parser = argparse.ArgumentParser(description='Process some directories.')
parser.add_argument('--directory_name1', type=str, help='The name of the directory to search in')
parser.add_argument('--directory_name2', type=str, help='The name of the directory to search in')

args = parser.parse_args()

# Directories to search
directories = [f"./outputs/{args.directory_name1}/{args.directory_name2}/run_{i}_sample100" for i in range(1, 11)]

# File range
file_range = range(100)

# Initialize a dictionary to store counts for each file
yes_counts = {f"{i}.json": 0 for i in file_range}
no_counts = {f"{i}.json": 0 for i in file_range}

# Iterate through each directory
for directory in directories:
    print(f"Checking directory: {directory}")
    for file_name in yes_counts.keys():
        file_path = os.path.join(directory, file_name)
        if os.path.exists(file_path):
            print(f"Processing file: {file_path}")
            try:
                with open(file_path, 'r') as file:
                    data = json.load(file)
                    if data["output"]["initial_prediction"] != data["label"]:
                        prediction_section = data.get("prediction", "").strip().lower()
                        match = re.search(r'prediction:\s*```(yes|no)```', prediction_section)

                        if match:
                            prediction_value = match.group(1)
                            print(f"Found prediction: {prediction_value} in {file_name}")
                            if prediction_value == "yes":
                                yes_counts[file_name] += 1
                            elif prediction_value == "no":
                                no_counts[file_name] += 1
                        else:
                            print(f"No prediction match found in {file_name}")
                    else:
                        print(f"Skipping file {file_name} due to initial prediction match")
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
        else:
            print(f"File does not exist: {file_path}")

# Initialize a dictionary to store success rates only for valid files
valid_success_rates = {}

# Initialize a dictionary to store failure rates only for valid files
valid_failure_rates = {}

# Calculate success rates only for valid files
for file_name in yes_counts.keys():
    if yes_counts[file_name] + no_counts[file_name] > 0:
        valid_success_rates[file_name] = yes_counts[file_name] / (no_counts[file_name] + yes_counts[file_name])
        valid_failure_rates[file_name] = no_counts[file_name] / (no_counts[file_name] + yes_counts[file_name])

# Calculate average success rate
if valid_success_rates:
    avg_success_rate = sum(valid_success_rates.values()) / len(valid_success_rates)
    valid_success_rates['avg_success_rate'] = avg_success_rate

# Calculate average failure rate
if valid_failure_rates:
    avg_failure_rate = sum(valid_failure_rates.values()) / len(valid_failure_rates)
    valid_failure_rates['avg_failure_rate'] = avg_failure_rate

# Print success rates
for file_name, rate in valid_success_rates.items():
    print(f"{file_name}: Success rate is {rate:.2%}")

# Print failure rates
for file_name, rate in valid_failure_rates.items():
    print(f"{file_name}: Failure rate is {rate:.2%}")

# Ensure the 'sr-rate' directory exists
os.makedirs('sr-rate', exist_ok=True)

# Print results and save success rates to JSON only for valid files
output_file = f'sr-rate/sr_results-{args.directory_name1}-{args.directory_name2}-success.json'
with open(output_file, 'w') as json_file:
    json.dump(valid_success_rates, json_file, indent=4, ensure_ascii=False)

# Save failure rates to JSON only for valid files
output_file_failure = f'sr-rate/sr_results-{args.directory_name1}-{args.directory_name2}-failure.json'
with open(output_file_failure, 'w') as json_file:
    json.dump(valid_failure_rates, json_file, indent=4, ensure_ascii=False)
