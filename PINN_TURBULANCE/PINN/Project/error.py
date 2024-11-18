import numpy as np

def l2norm_err(ref, pred):
    """
    Computes the L2 norm error between the reference and predicted values.

    Parameters:
        ref (np.ndarray): The reference values.
        pred (np.ndarray): The predicted values.

    Returns:
        np.ndarray: The L2 norm error between the reference and predicted values.
    
    Function:
        - Computes the L2 norm of the difference between the reference and predicted values.
        - Computes the L2 norm of the reference values.
        - Computes the L2 norm error as the ratio of the L2 norm of the difference to the L2 norm of the reference values, multiplied by 100.
        - Returns the L2 norm error.
    """
    return np.linalg.norm(ref - pred, axis = (1, 2)) / np.linalg.norm(ref, axis = (1, 2)) * 100
