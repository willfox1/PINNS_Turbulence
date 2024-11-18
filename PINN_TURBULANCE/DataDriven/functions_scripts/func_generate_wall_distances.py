import os
import re
import numpy as np
import pandas as pd
from tqdm import tqdm

def calculate_wall_distance(data):
    wall_distances = []

    unique_x_values = np.unique(data[:, 0])

    for x_value in tqdm(unique_x_values, desc="Calculating wall distances", unit="x"):
        rows_at_x = data[data[:, 0] == x_value]
        sorted_rows = rows_at_x[np.argsort(rows_at_x[:, 1])]

        wall_distance = np.zeros(len(sorted_rows))
        for i in range(1, len(sorted_rows)):
            wall_distance[i] = wall_distance[i - 1] + 1

        wall_distances.extend(wall_distance)

    return np.array(wall_distances)

def main():
    folder_path = "DNS_29_Periodic_Hills"
    output_folder = "walldist"
    os.makedirs(output_folder, exist_ok=True)

    all_files = os.listdir(folder_path)

    for file_name in all_files:
        if file_name.endswith(".dat"):
            file_path = os.path.join(folder_path, file_name)

            match = re.match(r'alph(\d+)-(\d+)-(\d+).dat', file_name)
            if match:
                slope, length, height = map(float, match.groups())
            else:
                print(f"Warning: Unable to extract slope, length, and height from file name: {file_name}")
                continue

            data = np.loadtxt(file_path, skiprows=20)
            wall_distances = calculate_wall_distance(data)

            df_wall_distance = pd.DataFrame({"WallDistance": wall_distances})
            output_path = os.path.join(output_folder, f"{file_name.replace('.dat', '_walldist.csv')}")
            df_wall_distance.to_csv(output_path, index=False)

if __name__ == "__main__":
    main()
