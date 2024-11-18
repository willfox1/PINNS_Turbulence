import numpy as np
from tensorflow.keras import models, layers, optimizers
from PINN_phill import PINNs
from train_configs import phill_config
from matplotlib import pyplot as plt
from time import time
from error import l2norm_err

def load_data(file_path):
    '''
    Load data from a file and extract positions, scalars, and create reference data.
    
    Parameters:
        file_path (str): The path to the file containing the data.
    
    Returns:
        tuple: A tuple containing the x and y coordintes, and the reference data.
    
    Function:
        - Load data from the file using numpy's load function.
        - Extract positions and scalars from the 'pos' and 'scal' headers in the .npz data file.
        - Extract x and y coordinates from the 'pos' positions header, with x in positions[0] and y in positions[1].
        - Extract velocity components etc. from the 'scal' scalars header as shown.
        - Set the initial values of the scalars to zero to avoid index error.
        - Create a reference data array stacking containing the scalar components only.
        - Return the x and y coordinates and the reference data.
    '''
    
    data = np.load(file_path)
    positions = data['pos']
    scalars = data['scal']

    x_coordinates = positions[0]
    y_coordinates = positions[1]

    indices = (x_coordinates[0] > 0) & (x_coordinates[0] < 1)

    u_velocity = scalars[0][:, indices]
    v_velocity = scalars[1][:, indices]
    pressure = scalars[2][:, indices]
    uv_product = scalars[4][:, indices] - u_velocity*v_velocity
    uu_product = scalars[5][:, indices] - u_velocity**2
    vv_product = scalars[3][:, indices] - v_velocity**2
           
    x_coordinates = positions[0][:, indices]
    y_coordinates = positions[1][:, indices]

    u_velocity[0] = 0.0
    v_velocity[0] = 0.0
    uv_product[0] = 0.0
    uu_product[0] = 0.0
    vv_product[0] = 0.0

    reference_data = np.stack((u_velocity, v_velocity, pressure, uv_product, uu_product, vv_product))

    return x_coordinates, y_coordinates, reference_data

def prepare_data(x_coordinates, y_coordinates, reference_data):
    '''
    Prepare data for training by selecting collocation points and boundary conditions.
    
    Parameters:
        x_coordinates (np.ndarray): The x coordinates of the data points.
        y_coordinates (np.ndarray): The y coordinates of the data points.
        reference_data (np.ndarray): The reference data of the data points.
    
    Returns:
        tuple: A tuple containing the collocation points and boundary conditions.
    
        Function:
        - Select collocation points from the x and y coordinates.
        - Plot the collocation points and save the figure in the figs folder.
        - Create an boolean array to select boundary points.
        - Extract the velocity components and their product at the boundary points.
        - Stack the boundary points and their corresponding values to create a boundary conditions array.
        - Return the collocation points and the boundary conditions.
    '''
    
    x_test = x_coordinates[::10, ::2]
    y_test = y_coordinates[::10, ::2]
    
    plt.figure(figsize=(16, 9))
    plt.scatter(x_coordinates, y_coordinates, color='r', marker='o', s=0.5, label='Mesh')
    plt.scatter(x_test, y_test, color='b', marker='o', s=0.5, label='Collocation points')
    plt.xlabel('X nodal positions')
    plt.ylabel('Y nodal positions')
    plt.legend(['Mesh', 'Collocation points'])
    plt.savefig('Project/figs/enforced_collocation_points.png', format='png', dpi = 1200)
     
    collocation_points = np.concatenate((x_test.reshape((-1, 1)), y_test.reshape((-1, 1))), axis=1)

    boundary_indices = np.zeros(x_coordinates.shape, dtype=bool)
    boundary_indices[[0, -1]] = True
    boundary_indices[:, [0, -1]] = True

    x_boundary = x_coordinates[boundary_indices].flatten()
    y_boundary = y_coordinates[boundary_indices].flatten()

    u_boundary = reference_data[0][boundary_indices].flatten()
    v_boundary = reference_data[1][boundary_indices].flatten()
    uv_boundary = reference_data[3][boundary_indices].flatten()
    uu_boundary = reference_data[4][boundary_indices].flatten()
    vv_boundary = reference_data[5][boundary_indices].flatten()
    p_boundary = reference_data[2][boundary_indices].flatten()

    boundary_conditions = np.array([x_boundary, y_boundary, u_boundary, v_boundary, uv_boundary, uu_boundary, vv_boundary, p_boundary]).T

    selected_boundary_indices = np.random.choice([False, True], len(boundary_conditions), p=[0.5, 0.5])
    boundary_conditions = boundary_conditions[selected_boundary_indices]

    return collocation_points, boundary_conditions

def build_model(input_dimension, output_dimension, neurons_per_layer, number_of_layers, activation_function):
    '''
    Build a neural network model with a specified architecture.
    
    Parameters:
        input_dimension (int): The input dimension of the model.
        output_dimension (int): The output dimension of the model.
        neurons_per_layer (int): The number of neurons per hidden layer.
        number_of_layers (int): The number of hidden layers in the model.
        activation_function (str): The activation function to use in the hidden layers.
    
    Returns:
        tensorflow.keras.Model: The built neural network model.
    
    Function:
        - Create an input layer with the specified input dimension.
        - Add the specified number of hidden layers to the model, each with the specified number of neurons and activation function.
        - Create an output layer with the specified output dimension.
        - Create a model with the input and output layers.
        - Return the built model.
    '''

    input_layer = layers.Input(shape=(input_dimension,))

    hidden_layer = input_layer

    for _ in range(number_of_layers):
        hidden_layer = layers.Dense(neurons_per_layer, activation=activation_function)(hidden_layer)

    output_layer = layers.Dense(output_dimension)(hidden_layer)

    model = models.Model(input_layer, output_layer)

    return model


def train_model(model, boundary_conditions, collocation_points, adam_iterations, learning_rate):
    '''
    Train the model using the PINN approach with the specified boundary conditions and collocation points.
    
    Parameters:
        model (tensorflow.keras.Model): The neural network model to train.
        boundary_conditions (np.ndarray): The boundary conditions for the model.
        collocation_points (np.ndarray): The collocation points for the model.
        adam_iterations (int): The number of Adam iterations to use during training.
        learning_rate (float): The learning rate to use during training.
    
    Returns:
        tuple: A tuple containing the trained model, training history, and computation time.
    
    Function:
        - Create an Adam optimizer with the specified learning rate.
        - Create a PINNs object with the model, optimizer, and number of Adam iterations.
        - Record the start time.
        - Train the model using the PINN approach with the specified boundary conditions and collocation points.
        - Record the end time. 
        - Compute the computation time as the difference between the end and start times.
        - Return the trained model, training history, and computation time.
    
    '''

    optimizer = optimizers.Adam(learning_rate)

    pinn = PINNs(model, optimizer, adam_iterations)

    start_time = time()
   
    training_history = pinn.fit(boundary_conditions, collocation_points)
    
    end_time = time()
    
    computation_time = end_time - start_time

    return pinn, training_history, computation_time


def predict_and_save(pinn, x_coordinates, y_coordinates, reference_data, test_name, training_history, computation_time):
    '''
    Predict the flow variables using the trained model and save the results.
    
    Parameters:
        pinn (PINNs): The trained PINN model.
        x_coordinates (np.ndarray): The x coordinates of the data points.
        y_coordinates (np.ndarray): The y coordinates of the data points.
        reference_data (np.ndarray): The reference data for the scalar flow variables.
        test_name (str): The name to use for saving the results.
        training_history (dict): The training history of the model.
        computation_time (float): The computation time of the training process.
    
    Returns:
        None
    
    Function:
        - Create an array of collocation points from the x and y coordinates
        - Predict the flow variables using the trained model.
        - Reshape the predicted flow variables to match the shape of the reference data.
        - Stack the predicted flow variables to create a prediction array.
        - Compute the error between the predicted and reference data.
        - Save the prediction, reference data, training history, and computation time to a file.
        - Save the model to a file.
        - Print a message indicating that the prediction and model have been saved.
    '''

    collocation_points_prediction = np.array([x_coordinates.flatten(), y_coordinates.flatten()]).T


    prediction = pinn.predict(collocation_points_prediction)

    u_prediction = prediction[:, 0].reshape(reference_data[0].shape)
    v_prediction = prediction[:, 1].reshape(reference_data[0].shape)
    uv_prediction = prediction[:, 2].reshape(reference_data[0].shape)
    uu_prediction = prediction[:, 3].reshape(reference_data[0].shape)
    vv_prediction = prediction[:, 4].reshape(reference_data[0].shape)
    p_prediction = prediction[:, 5].reshape(reference_data[0].shape)

    prediction = np.stack((u_prediction, v_prediction, p_prediction, uv_prediction, uu_prediction, vv_prediction))

    error = l2norm_err(reference_data, prediction)

    np.savez_compressed('Project/pred/res_phill' + test_name, pred=prediction, ref=reference_data, hist=training_history, ct=computation_time)

    model.save('Project/models/model_phill' + test_name + '.h5')

    print("INFO: Prediction and model have been saved!")


if __name__ == "__main__":
    '''
    Function:
        - Load the data from a file using the load_data function.
        - Prepare the data for training using the prepare_data function.
        - Set the input and output dimensions of the model.
        - Set the number of neurons per layer, number of layers, and activation function of the model.
        - Set the number of Adam iterations and learning rate of the model.
        - Set the number of collocation points and collocation point type.
        - Construct the test name based on the model configuration.
        - Build the model using the build_model function.
        - Train the model using the PINN approach using the train_model function.
        - Predict the flow variables using the trained model and save the results using the predict_and_save function.
    '''

    x_coordinates, y_coordinates, reference_data = load_data('Project/data/mesh.npz')

    collocation_points, boundary_conditions = prepare_data(x_coordinates, y_coordinates, reference_data)

    input_dimension = 2
    output_dimension = boundary_conditions.shape[1] - input_dimension

    neurons_per_layer = phill_config.n_neural
    number_of_layers = phill_config.n_layer
    activation_function = phill_config.act
    adam_iterations = phill_config.n_adam
    learning_rate = 1e-3

    number_of_collocation_points = 2430
    collocation_point_type = 'grid'

    test_name = f'_{neurons_per_layer}_{number_of_layers}_{activation_function}_{adam_iterations}_{number_of_collocation_points}_{collocation_point_type}'

    model = build_model(input_dimension, output_dimension, neurons_per_layer, number_of_layers, activation_function)

    pinn, training_history, computation_time = train_model(model, boundary_conditions, collocation_points, adam_iterations, learning_rate)

    predict_and_save(pinn, x_coordinates, y_coordinates, reference_data, test_name, training_history, computation_time)
