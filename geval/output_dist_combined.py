import os
import json

# Define the input files
input_files = [
    'no_counts_results-medqa-consistency.json',
    'no_counts_results-medqa-correctness.json',
    'no_counts_results-medqa-explainability.json'
]

# Initialize a dictionary to store combined results
combined_results = {f"{i}.json": {"consistency": 0, "correctness": 0, "explainability": 0} for i in range(100)}

# Load data from each input file and combine
for input_file in input_files:
    with open(input_file, 'r') as file:
        data = json.load(file)
        for key, value in data.items():
            if key in combined_results:
                if "consistency" in input_file:
                    combined_results[key]["consistency"] = value
                elif "correctness" in input_file:
                    combined_results[key]["correctness"] = value
                elif "explainability" in input_file:
                    combined_results[key]["explainability"] = value

# Save the combined results to a JSON file
output_file = 'combined_no_counts_results-medqa.json'
with open(output_file, 'w') as json_file:
    json.dump(combined_results, json_file, indent=4, ensure_ascii=False)

print(f"Combined results saved to {output_file}")