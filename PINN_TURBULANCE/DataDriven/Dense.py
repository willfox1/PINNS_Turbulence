import os
import re
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout
import keras.backend as K
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
def load_wall_distances(folder_path):
    all_wall_distances = []

    for file_name in os.listdir(folder_path):
        if file_name.endswith("_walldist.csv"):
            file_path = os.path.join(folder_path, file_name)
            wall_distances_df = pd.read_csv(file_path)
            all_wall_distances.extend(wall_distances_df["WallDistance"].to_numpy())

    # Normalize wall distances based on the minimum and maximum across all files
    scaler_wall_distance = MinMaxScaler()
    all_wall_distances_normalized = scaler_wall_distance.fit_transform(np.array(all_wall_distances).reshape(-1, 1))

    return all_wall_distances_normalized.flatten()

def save_predictions_to_file(predictions_file_path, x_test, predictions):
    # Create a DataFrame with x, y, and predicted vorticity_mean
    df = pd.DataFrame(np.column_stack([x_test[:, :2], predictions]), columns=["x", "y", "u_mean", "v_mean", "p_mean", "uu", "vv", "uv"])

    # Save the DataFrame to the specified file with the correct header and without empty rows
    with open(predictions_file_path, 'w') as file:
        file.write('TITLE = "predictions"\n')
        file.write('VARIABLES = "x"\n"y"\n"u_mean"\n"v_mean"\n"w_mean"\n"p_mean"\n"dissipation_mean"\n"vorticity_mean"\n')
        file.write('"uu"\n"vv"\n"ww"\n"uv"\n"uw"\n"vw"\n"pp"\n')
        file.write('ZONE T="pehill"\n')
        file.write(f'I={len(x_test)} ')
        file.write('J=1\n')
        file.write('DATAPACKING=POINT\n')
        file.write('DT=(SINGLE SINGLE SINGLE SINGLE SINGLE SINGLE SINGLE SINGLE SINGLE SINGLE SINGLE SINGLE SINGLE SINGLE SINGLE)\n')
        df.to_csv(file, index=False, header=False, sep=" ")

def main():
    folder_path = "DNS_29_Periodic_Hills"
    walldist_folder = "walldist"

    all_wall_distances_normalized = load_wall_distances(walldist_folder)

    all_x = []
    all_y = []

    x_test = None
    y_test = None

    for file_name in os.listdir(folder_path):
        if file_name.endswith(".dat"):
            file_path = os.path.join(folder_path, file_name)

            match = re.match(r'alph(\d+)-(\d+)-(\d+).dat', file_name)
            if match:
                slope, length, height = map(float, match.groups())
            else:
                print(f"Warning: Unable to extract slope, length, and height from file name: {file_name}")
                continue

            data = np.loadtxt(file_path, skiprows=20)

            # Ensure the length of wall_distances matches the number of rows in data
            wall_distances = np.array(all_wall_distances_normalized)[:len(data)]

            x = np.column_stack([data[:, :2], wall_distances,
                                 slope * np.ones_like(data[:, :1]),
                                 length * np.ones_like(data[:, :1]),
                                 height * np.ones_like(data[:, :1])])
            y = data[:, [2, 3, 5, 8, 9, 11]]

            # Choose the split based on the file name condition
            if file_name != "alph05-4071-3036.dat":
                all_x.append(x)
                all_y.append(y)
            else:
                x_test = x
                y_test = y

    all_x = np.concatenate(all_x, axis=0)
    all_y = np.concatenate(all_y, axis=0)

    # Normalize features using MinMaxScaler
    scaler_input = MinMaxScaler()
    all_x_normalized = scaler_input.fit_transform(all_x)
    x_test_normalized = scaler_input.transform(x_test)
    
    # Model architecture with added dropout layers
    model = Sequential()
    model.add(Dense(128, activation='relu', input_shape=(6,)))
    model.add(BatchNormalization())
    model.add(Dense(64, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(64, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(32, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(32, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(16, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(16, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(8, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(8, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(6))

    # Learning rate schedule
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=1e-6)

    # Early stopping based on validation loss
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    model.compile(optimizer='adam', loss='mae', metrics=['mae'])

    # Additional: Store the training history for plotting
    history = model.fit(all_x_normalized, all_y, epochs=75, batch_size=128, validation_split=0.2, callbacks=[reduce_lr, early_stop])

    predictions = model.predict(x_test_normalized)

    for i in range(1):
        rmse = np.sqrt(mean_squared_error(y_test[:, i], predictions[:, i]))
        r2 = r2_score(y_test[:, i], predictions[:, i])
        print(f'Root Mean Squared Error (RMSE) for {["u_mean", "v_mean", "p_mean", "uu", "vv", "uv"][i]}: {rmse}')
        print(f'R-squared for {["u_mean", "v_mean", "p_mean", "uu", "vv", "uv"][i]}: {r2}')

    # Create the 'plots' folder if it doesn't exist
    if not os.path.exists("plots"):
        os.makedirs("plots")

    # Save the predictions to a new file in the "plots" folder
    predictions_file_path = os.path.join("plots", "predictions.dat")
    save_predictions_to_file(predictions_file_path, x_test, predictions)

    # Plot the training loss against epochs
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join("plots", "training_validation_loss.png"))
    plt.close()

    # Scatter plot for u_mean
    plt.scatter(y_test[:, 0], predictions[:, 0], alpha=0.5)
    plt.xlabel('actual u_mean')
    plt.ylabel('predicted u_mean')
    plt.title('actual vs predicted u_mean')
    x = np.linspace(-0.4, 1.4, 2)
    y = x
    plt.plot(y, x, color='r', linestyle='--')
    plt.savefig(os.path.join("plots", "scatter_u_mean.png"))
    plt.close()
    
    # Scatter plot for v_mean
    plt.scatter(y_test[:, 1], predictions[:, 1], alpha=0.5)
    plt.xlabel('actual v_mean')
    plt.ylabel('predicted v_mean')
    plt.title('actual vs predicted v_mean')
    x = np.linspace(-0.24, 0.3, 2)
    y = x
    plt.plot(y, x, color='r', linestyle='--')
    plt.savefig(os.path.join("plots", "scatter_v_mean.png"))
    plt.close()
    
    # Scatter plot for p_mean
    plt.scatter(y_test[:, 2], predictions[:, 2], alpha=0.5)
    plt.xlabel('actual p_mean')
    plt.ylabel('predicted p_mean')
    plt.title('actual vs predicted p_mean')
    x = np.linspace(-0.003, 0.006, 2)
    y = x
    plt.plot(y, x, color='r', linestyle='--')
    plt.savefig(os.path.join("plots", "scatter_p_mean.png"))
    plt.close()

    # Create the contour plot for u_mean.
    contour_plot_u = plt.tricontourf(x_test[:, 0], x_test[:, 1], predictions[:, 0], cmap='viridis')
    plt.colorbar(contour_plot_u, label='u_mean')
    plt.xlabel('x Values')
    plt.ylabel('y Values')
    plt.title('predicted u_mean contour plot')
    plt.savefig(os.path.join("plots", "contour_u_mean.png"))
    plt.close()

    # Create the contour plot for v_mean.
    contour_plot_u = plt.tricontourf(x_test[:, 0], x_test[:, 1], predictions[:, 1], cmap='viridis')
    plt.colorbar(contour_plot_u, label='v_mean')
    plt.xlabel('x Values')
    plt.ylabel('y Values')
    plt.title('predicted v_mean contour plot')
    plt.savefig(os.path.join("plots", "contour_v_mean.png"))
    plt.close()
    
    # Create the contour plot for p_mean.
    contour_plot_u = plt.tricontourf(x_test[:, 0], x_test[:, 1], predictions[:, 2], cmap='viridis')
    plt.colorbar(contour_plot_u, label='p_mean')
    plt.xlabel('x Values')
    plt.ylabel('y Values')
    plt.title('predicted p_mean contour plot')
    plt.savefig(os.path.join("plots", "contour_p_mean.png"))
    plt.close()

if __name__ == "__main__":
    main()
