from donutplot.params import *
import pandas as pd
import json

file_path = TEST_PATH + "metadata.jsonl"

# Open the JSONL file for reading
gt = []
with open(file_path, "r") as file:
    # Read each line and parse it as JSON
    for line in file:
        try:
            # Parse the JSON object from the line
            data = json.loads(line)

            # Process the data (for example, print it)
            print(data)
            gt.append(data)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")

breakpoint()
