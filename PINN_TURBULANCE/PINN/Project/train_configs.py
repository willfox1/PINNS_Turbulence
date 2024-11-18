class phill_config:
    """
    Train configs for PINN neural network.
    
    Parameters:
        act (stt): Activation function used for MLP. 
        n_adam (int): Number of steps used for supervised training.
        n_neural (int): Hidden dim fo each MLP layer (N_h).
        n_layer (int): Total MLP layers used in model (N_l).
        method (str): Optimizer for unsupervised training.
    
    Returns:
        None
    
    Function:
        None
    """
    
    act = "tanh"
    n_adam = 1000
    n_neural = 20
    n_layer = 8  
    method = "L-BFGS-B"
