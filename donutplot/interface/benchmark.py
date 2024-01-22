from donutplot.params import *
from donutplot.interface.predict import make_prediction
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial import cKDTree
import numpy as np
import pandas as pd
import json

file_path = os.path.join(TEST_PATH, "metadata.jsonl")

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


def series_to_df(series):
    df = pd.DataFrame()
    for serie in series:
        temp = pd.DataFrame(serie)
        df = pd.concat([df, temp], ignore_index=True)
    return df


precision_list = []
recall_list = []
error_list = []

failed_ones = 0
try:
    for i in range(len(gt)):
        result = make_prediction(os.path.join(TEST_PATH, str(i).zfill(4) + ".jpg"))
        if result["status"] != "success":
            failed_ones += 1
        else:
            result_df = series_to_df(result["prediction"]["data_dicts"])
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
                [
                    result_df,
                    pd.Series(indices.flatten()),
                    pd.Series(distances.flatten()),
                ],
                axis=1,
            ).sort_values(by=0)

            original_points = gt_df.shape[0]
            true_positives = len(knnd_result.iloc[:, 3].unique())
            false_positives = len(knnd_result.iloc[:, 3]) - true_positives

            correctly_assigned = np.array(knnd_result.iloc[:, 0]) == np.array(
                gt_df.iloc[knnd_result.iloc[:, 3], 0]
            )

            precision = round(
                sum(correctly_assigned) * 100 / len(correctly_assigned), 2
            )
            recall = round(true_positives * 100 / original_points, 2)
            error = round(np.mean(distances) * 100, 2)
            if error < np.e:
                precision_list.append(precision)
                recall_list.append(recall)
                error_list.append(error)
                print(f"Image {i+1} of {len(gt)}")
            else:
                failed_ones += 1
finally:
    failed_fraction = failed_ones * 100 / i
    precision_mean = sum(precision_list) / len(precision_list)
    recall_mean = sum(recall_list) / len(recall_list)
    error_mean = np.mean(np.array(error_list))
    error_max = np.max(np.array(error_list))
    error_median = np.median(np.array(error_list))

    print(
        f"""
        Number of images use for the benchmark:{i}
        Failed: {failed_fraction}%
        Figure size = {FIGSIZE_WIDTH_TRAINING_MODE} x {FIGSIZE_HEIGHT_TRAINING_MODE} inches
        Resolution = {DPI_TRAINING_MODE} DPI
        Mean precision = {precision_mean}%
        Mean recall = {recall_mean}%
        Mean spatial error is {error_mean}%
        Max spatial error is {error_max}%
        Median spatial error is {error_median}%"""
    )
