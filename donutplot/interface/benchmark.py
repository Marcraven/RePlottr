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

# benchmarking evaluation


def series_to_df(series):
    df = pd.DataFrame()
    for serie in series:
        temp = pd.DataFrame(serie)
        df = pd.concat([df, temp], ignore_index=True)
    return df


precision_list = np.ndarray(len(gt))
recall_list = np.ndarray(len(gt))
error_list = np.ndarray(len(gt))


for i in range(len(gt)):
    result = make_prediction(TEST_PATH + str(i).zfill(4) + ".jpg")
    breakpoint()
    result_df = series_to_df(result["data_dicts"])
    gt_df = series_to_df(gt[i]["ground_truth"]["data_dicts"])

    neigh = NearestNeighbors(n_neighbors=1)
    scaler = MinMaxScaler()

    gt_df.iloc[:, 1:] = scaler.fit_transform(X=gt_df.iloc[:, 1:])

    result_df.iloc[:, 1:] = scaler.transform(result_df.iloc[:, 1:])

    neigh.fit(gt_df.iloc[:, 1:])

    distances, indices = neigh.kneighbors(
        result_df.iloc[:, 1:], 1, return_distance=True
    )

    knnd_result = pd.concat(
        [result_df, pd.Series(indices.flatten()), pd.Series(distances.flatten())],
        axis=1,
    ).sort_values(by=0)

    original_points = gt_df.shape[0]
    true_positives = len(knnd_result.iloc[:, 3].unique())
    false_positives = len(knnd_result.iloc[:, 3]) - true_positives

    correctly_assigned = np.array(knnd_result.iloc[:, 0]) == np.array(
        gt_df.iloc[knnd_result.iloc[:, 3], 0]
    )

    precision = round(sum(correctly_assigned) * 100 / len(correctly_assigned), 2)
    recall = round(true_positives * 100 / original_points, 2)
    error = round(np.mean(distances) * 100, 2)

    precision_list[i] = precision
    recall_list[i] = recall
    error_list[i] = error

precision_mean = np.mean(precision_list)
recall_mean = np.mean(recall_list)
error_mean = np.mean(error_list)

print(
    f"Number of images use for the benchmark:{len(gt)} \nMean precision = {precision_mean}% \nMean recall = {recall_mean}% \nMean spatial error is {error_mean}%"
)
