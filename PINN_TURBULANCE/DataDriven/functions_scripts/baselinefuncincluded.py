import os
import re
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout
import keras.backend as K
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
from sklearn.metrics import mean_squared_error, r2_score
from keras.callbacks import ReduceLROnPlateau, EarlyStopping

# set tensorflow to use cpu instead of gpu
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# define boundary nodes
def boundary_points(file_path):
    # Read data from the file skipping the first 20 rows
    data = np.loadtxt(file_path, skiprows=20)

    # Extract x and y coordinates
    x = data[:, 0]
    y = data[:, 1]

    # Find boundary points based on the rules
    right_boundary = data[(data[:, 0] == np.min(data[:, 0])) & (data[:, 2] != 0) & (data[:, 3] != 0) & (data[:, 5] != 0)]
    left_boundary = data[(data[:, 0] == np.max(data[:, 0])) & (data[:, 2] != 0) & (data[:, 3] != 0) & (data[:, 5] != 0)]

    # Identify bottom boundary points
    bottom_boundary = np.zeros((len(data), 2))
    unique_x_values = np.unique(data[:, 0])
    for x_value in unique_x_values:
        col_data = data[data[:, 0] == x_value]
        col_data = col_data[col_data[:, 2] != 0]  # Non-zero u_mean
        col_data = col_data[col_data[:, 3] != 0]  # Non-zero v_mean
        col_data = col_data[col_data[:, 5] != 0]  # Non-zero p_mean
        if len(col_data) > 0:
            bottom_boundary[data[:, 0] == x_value] = col_data[np.argmin(col_data[:, 1])][:2]

    # Identify top boundary points
    top_boundary = np.zeros((len(data), 2))
    for x_value in unique_x_values:
        col_data = data[data[:, 0] == x_value]
        col_data = col_data[col_data[:, 2] != 0]  # Non-zero u_mean
        col_data = col_data[col_data[:, 3] != 0]  # Non-zero v_mean
        col_data = col_data[col_data[:, 5] != 0]  # Non-zero p_mean
        if len(col_data) > 0:
            top_boundary[data[:, 0] == x_value] = col_data[np.argmax(col_data[:, 1])][:2]

    # Plot the scatter plot with all boundary points
    plt.scatter(x, y, s=1, label='Data Points')
    plt.scatter(right_boundary[:, 0], right_boundary[:, 1], color='red', label='Right Boundary', marker='x')
    plt.scatter(left_boundary[:, 0], left_boundary[:, 1], color='blue', label='Left Boundary', marker='x')
    plt.scatter(bottom_boundary[:, 0], bottom_boundary[:, 1], color='green', label='Bottom Boundary', marker='x')
    plt.scatter(top_boundary[:, 0], top_boundary[:, 1], color='orange', label='Top Boundary', marker='x')
    plt.title('Scatter Plot with All Boundary Points')
    plt.xlabel('X Nodal Positions')
    plt.ylabel('Y Nodal Positions')
    plt.legend()
    plt.show()
    
    return [right_boundary, left_boundary, top_boundary, bottom_boundary]
    
# define navier stoke equations
def navier_stokes(x, y):
    
    # define constants
    rho = 1.0
    nu = 0.01
    eps = 1e-8

    # define variables from predictions
    u, v, p = y[:, 0:1], y[:, 1:2], y[:, 2:3]

    # define first derivatives from gradients function
    du = tf.gradients(u, x)[0]
    dv = tf.gradients(v, x)[0]
    dp = tf.gradients(p, x)[0]

    # define first partial derivatives from gradients function
    p_x, p_y = dp[:, 0:1], dp[:, 1:2]
    u_x, u_y = du[:, 0:1], du[:, 1:2]
    v_x, v_y = dv[:, 0:1], dv[:, 1:2]
    
    # define second partial derivatives from gradients function
    u_xx = tf.gradients(u_x, x)[0][:, 0:1] if tf.gradients(u_x, x)[0] is not None else tf.zeros_like(u_x)
    u_yy = tf.gradients(u_y, x)[0][:, 1:2] if tf.gradients(u_y, x)[0] is not None else tf.zeros_like(u_y)

    v_xx = tf.gradients(v_x, x)[0][:, 0:1] if tf.gradients(v_x, x)[0] is not None else tf.zeros_like(v_x)
    v_yy = tf.gradients(v_y, x)[0][:, 1:2] if tf.gradients(v_y, x)[0] is not None else tf.zeros_like(v_y)

    # define momentum in x and y and continuity equation
    continuity = u_x + v_y
    x_momentum = u * u_x + v * u_y + 1 / rho * p_x - nu * (u_xx + u_yy)
    y_momentum = u * v_x + v * v_y + 1 / rho * p_y - nu * (v_xx + v_yy)

    # return equations results
    return [continuity, x_momentum, y_momentum]

# define custom loss function
def custom_loss(y_true, y_pred, right_boundary, left_boundary, top_boundary, bottom_boundary):
    
    # Define variables from predictions
    u, v, p = y_pred[:, 0:1], y_pred[:, 1:2], y_pred[:, 2:3]

    # Calculate Navier-Stokes terms
    continuity, x_momentum, y_momentum = navier_stokes(y_pred, y_pred)

    # Calculate k.square error in continuity, x, and y momentum
    pde_term = K.mean(K.square(continuity)) + K.mean(K.square(x_momentum)) + K.mean(K.square(y_momentum))

    # Combine the terms in the total loss
    total_loss = pde_term 

    # Return total loss
    return total_loss

# define wall distance function
def load_wall_distances(folder_path):
    # initialise list of wall distances
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

  # Create DataFrame
  df = pd.DataFrame(np.column_stack([x_test[:, :2], predictions]), 
                    columns=["x", "y", "u_mean", "v_mean", "p_mean", "uu", "vv", "uv"])

  # Write predictions file
  with open(predictions_file_path, 'w') as file:
    file.write('TITLE = "predictions"\n')
    file.write('VARIABLES = "x"\n"y"\n"u_mean"\n"v_mean"\n"w_mean"\n"p_mean"\n"dissipation_mean"\n"vorticity_mean"\n')
    file.write('"uu"\n"vv"\n"ww"\n"uv"\n"uw"\n"vw"\n"pp"\n')
    file.write('ZONE T="pehill"\n') 
    file.write(f'I={len(x_test)} ')
    file.write('J=1\n')
    file.write('DATAPACKING=POINT\n')
    file.write('DT=(SINGLE SINGLE SINGLE SINGLE SINGLE SINGLE SINGLE SINGLE SINGLE SINGLE SINGLE SINGLE SINGLE SINGLE SINGLE)\n')
    
    # Write DataFrame to file
    df.to_csv(file, index=False, header=False, sep=" ")
    
def main():
    folder_path = "DNS_29_Periodic_Hills"
    walldist_folder = "walldist"

    all_wall_distances_normalized = load_wall_distances(walldist_folder)

    x_test = None
    y_test = None

    for file_name in os.listdir(folder_path):
        if file_name == "alph05-4071-3036.dat":
            file_path = os.path.join(folder_path, file_name)

            match = re.match(r'alph(\d+)-(\d+)-(\d+).dat', file_name)
            if match:
                slope, length, height = map(float, match.groups())
            else:
                print(f"Warning: Unable to extract slope, length, and height from file name: {file_name}")
                continue

            data = np.loadtxt(file_path, skiprows=20)

            wall_distances = np.array(all_wall_distances_normalized)[:len(data)]

            x_test = np.column_stack([data[:, :2], wall_distances,
                                     slope * np.ones_like(data[:, :1]),
                                     length * np.ones_like(data[:, :1]),
                                     height * np.ones_like(data[:, :1])])
            y_test = data[:, [2, 3, 5, 8, 9, 11]]

    # Normalize features using MinMaxScaler
    scaler_input = MinMaxScaler()
    x_test_normalized = scaler_input.fit_transform(x_test)
    
    right_boundary, left_boundary, top_boundary, bottom_boundary = boundary_points(file_path)
    
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
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=False)

    model.compile(optimizer='adam', loss=lambda y_true, y_pred: custom_loss(y_true, y_pred, right_boundary, left_boundary, top_boundary, bottom_boundary), metrics=['mae'])

    # Additional: Store the training history for plotting
    history = model.fit(x_test_normalized, y_test, epochs=50, batch_size=16, validation_split=0.1, callbacks=[reduce_lr, early_stop])

    predictions = model.predict(x_test_normalized)

    for i in range(1):
        rmse = np.sqrt(mean_squared_error(y_test[:, i], predictions[:, i]))
        r2 = r2_score(y_test[:, i], predictions[:, i])
        print(f'Root Mean Squared Error (RMSE) for {["u_mean", "v_mean", "p_mean", "uu", "vv", "uv"][i]}: {rmse}')
        print(f'R-squared for {["u_mean", "v_mean", "p_mean", "uu", "vv", "uv"][i]}: {r2}')

    # Create the 'plots' folder if it doesn't exist
    if not os.path.exists("plots"):
        os.makedirs("plots", exist_ok=True)

    # Save predictions
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

    # Scatter plot forwD u_mean
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
