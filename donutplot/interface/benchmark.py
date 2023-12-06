from donutplot.params import *
from donutplot.interface.predict import make_prediction
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial import cKDTree
import numpy as np
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
            gt.append(data)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")

result = make_prediction(TEST_PATH + "0000.jpg")
gt[0]["ground_truth"]


def series_to_df(series):
    df = pd.DataFrame()
    for serie in series:
        temp = pd.DataFrame(serie)
        df = pd.concat([df, temp], ignore_index=True)
    return df


result_df = series_to_df(result["data_dicts"])
gt_df = series_to_df(gt[0]["ground_truth"]["data_dicts"])

neigh = NearestNeighbors(n_neighbors=1)

scaler = MinMaxScaler()

gt_df.iloc[:, 1:] = scaler.fit_transform(X=gt_df.iloc[:, 1:])


result_df.iloc[:, 1:] = scaler.transform(result_df.iloc[:, 1:])


neigh.fit(gt_df.iloc[:, 1:])
distances, indices = neigh.kneighbors(result_df.iloc[:, 1:], 5, return_distance=True)
breakpoint()
if len(np.unique(indices[:, 0])) < indices[:, 0].shape[0]:
    a, b = np.unique(indices[:, 0], return_counts=True)
    indices[b > 1, 0]


breakpoint()
sorted_df = result_df.set_index(indices.flatten()).sort_index()
mean_distance = np.mean(distances)
max_distance = np.max(distances)
min_distance = np.min(distances)

breakpoint()
