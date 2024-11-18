import numpy as np
import tensorflow as tf
from tensorflow.keras import models
from lbfgs import optimizer as lbfgs_op


class PINNs(models.Model):
    '''
    PINNs training class
    '''
    def __init__(self, model, optimizer, epochs, **kwargs):
        '''
        Initialises the PINNs class.
        
        Parameters:
            model (tensorflow.keras.Model): The neural network model to train.
            optimizer (tensorflow.keras.optimizers.Optimizer): The optimizer to use during training.
            epochs (int): The number of epochs to train for.
        
        Returns:
            None
        
        Function:
            None
        '''

        super(PINNs, self).__init__(**kwargs)
        self.model = model
        self.optimizer = optimizer
        self.epochs = epochs
        self.hist = []
        self.epoch = 0
        self.sopt = lbfgs_op(self.trainable_variables)
        self.nu = 1/5600        # kinematic viscosity
              
    @tf.function
    def net_f(self, cp):
        '''
        Computes the residual of the PDE gor the given collocation points.
        
        Parameters:
            cp (tf.Tensor): The collocation points.
            
        Returns:
            tf.Tensor: The residual of the PDE for the given collocation points.
        
        Function:
            - Scales the collocation points using the 'scalex_r' function.
            - Extracts the x and y coordinates from the collocation points.
            - Computes the predicted values using the neural network model.
            - Scales the predicted values using the 'scale_r' function.
            - Computes first and second order partial differentials of the scalar fields using TensorFlow#'s gradient tape.
            - Computes the residual of the PDEs using gradients and kinematic viscosity (density is 1 for water in this case).
            - Returns the residual of the PDEs.
        '''
        
        cp = self.scalex_r(cp)
        x = cp[:, 0]
        y = cp[:, 1]
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x)
            tape.watch(y)
            X = tf.stack([x, y], axis = -1)
            X = self.scalex(X)
            pred = self.model(X)
            pred = self.scale_r(pred)
            U = pred[:, 0]
            V = pred[:, 1]
            uv = pred[:, 2]
            uu = pred[:, 3]
            vv = pred[:, 4]
            P = pred[:, 5]


            U_x = tape.gradient(U, x)
            U_y = tape.gradient(U, y)
            V_x = tape.gradient(V, x)
            V_y = tape.gradient(V, y)
        U_xx = tape.gradient(U_x, x)
        U_yy = tape.gradient(U_y, y)
        V_xx = tape.gradient(V_x, x)
        V_yy = tape.gradient(V_y, y)
        P_x = tape.gradient(P, x)
        P_y = tape.gradient(P, y)
        uv_y = tape.gradient(uv, y)
        uv_x = tape.gradient(uv, x)
        uu_x = tape.gradient(uu, x)
        vv_y = tape.gradient(vv, y)
        
      
              
        f1 = U * U_x + V * U_y + P_x -  self.nu * (U_xx + U_yy) + uu_x + uv_y
        f2 = U * V_x + V * V_y + P_y -  self.nu * (V_xx + V_yy) + uv_x + vv_y
        f3 = U_x + V_y
        
        f = tf.stack([f1, f2, f3], axis = -1)
        return f
    
    
    @tf.function
    def train_step(self, bc, cp):
        '''
        Performs a single training step using the given boundary conditions and collocation points.
        
        Parameters:
            bc (tf.Tensor): The boundary conditions.
            cp (tf.Tensor): The collocation points.
            
        Returns:
            tuple: A tuple containing the loss, gradients, and training history for the current step.
        
        Function:
            - Extracts the input and output values from the boundary conditions.
            - Computes the predicted values using the neural network model.
            - Computes the residual of the PDE using the 'net_f' function.
            - Computes the loss as the mean squared error between the predicted values and true output values, and the mean squared error of the residual.
            - Computes the gradients of the loss with respect to the trainable variables using TensorFlow#'s gradient tape.
            - Returns the loss, gradients, and training history for the current step.
        '''
        X = bc[:, :2]                       # This can be changed to change the values which are being compared at the boundary condition.
        y = bc[:, 2:]                       # This can be changed to change the values which are being compared at the boundary condition.
        with tf.GradientTape() as tape:
            u_p_bc = self.model(X)          # This should be changed to match only the values which are being compared at the boundary condition (in this case all values)
            
            f = self.net_f(cp)
            
            loss_bc = tf.reduce_mean(tf.square(y - u_p_bc)) # Or it can be changed here by specifying only the columns which we want to compare
            loss_f = tf.reduce_mean(tf.square(f))
            
            loss_u = loss_bc
            loss = loss_u + loss_f
            
        trainable_vars = self.trainable_variables
        grads = tape.gradient(loss, trainable_vars)
        
        l1 = tf.reduce_mean(loss)
        l2 = tf.reduce_mean(loss_u)
        l3 = tf.reduce_mean(loss_f)
        
        tf.print('loss:', l1, 'loss_u:', l2, 'loss_f:', l3)
        return loss, grads, tf.stack([l1, l2, l3])
    
    # @tf.function
    def fit_scale(self, y):
        '''
        Scales the given output values to the range [-1, 1].
        
        Parameters:
            y (tf.Tensor): The output values to scale.
            
        Returns:
            tf.Tensor: The scaled output values.
        
        Function:
            - Computes the maximum absolute value of the output values.
            - Scales the output values by dividing them by the maximum absolute value.
            - Returns the scaled output values.
        '''

        ymax = tf.reduce_max(tf.abs(y), axis = 0)
        self.ymax = ymax
        return y / ymax
    
    @tf.function
    def scale(self, y):
        '''
        Scales the given ouptut values using the fitted scale.
        
        Parameters:
            y (tf.Tensor): The output values to scale.
            
        Returns:
            tf.Tensor: The scaled output values.
        
        Function:
            - Scales the output values using the fitted scale.
            - Returns the scaled output values.
        '''

        return y / self.ymax
    
    @tf.function
    def scale_r(self, ys):
        '''
        Reverses the scaling of the given output values using the fitted scale.
        
        Parameters:
            ys (tf.Tensor): The output values to scale.
            
        Returns:
            tf.Tensor: The scaled output values.
        
        Function:
            - Reverses the scaling of the output values using the fitted scale.
            - Returns the scaled output values.
        '''
        
        return ys * self.ymax
    
    # @tf.function
    def fit_scalex(self, x):
        '''
        Scales the given input values to the range [-1, 1].
        
        Parameters:
            x (tf.Tensor): The input values to scale.
            
        Returns:
            tf.Tensor: The scaled input values.
        
        Function:
            - Computes the maximum and minimum values of the input values.
            - Scales the input values to the range [-1, 1] using the maximum and minimum values.
            - Returns the scaled input values.
        '''
        
        xmax = tf.reduce_max(tf.abs(x), axis = 0)
        xmin = tf.reduce_min(x, axis = 0)
        self.xmax = xmax
        self.xmin = xmin
        xs = ((x - xmin) / (xmax - xmin))
        return xs
    
    @tf.function
    def scalex(self, x):
        '''
        Scales the given input values using the fitted scale.
        
        Parameters:
            x (tf.Tensor): The input values to scale.
            
        Returns:
            tf.Tensor: The scaled input values.
        
        Function:
            - Scales the input values using the fitted scale.
            - Returns the scaled input values.
        '''
        
        xs = ((x - self.xmin) / (self.xmax - self.xmin)) 
        return xs
    
    @tf.function
    def scalex_r(self, xs):
        '''
        Reverses the scaling of the given input values using the fitted scale.
        
        Parameters:
            xs (tf.Tensor): The input values to scale.
            
        Returns:
            tf.Tensor: The scaled input values.
        
        Function:
            - Reverses the scaling of the input values using the fitted scale.
            - Returns the original input values.
        '''
        
        x = (xs) * (self.xmax - self.xmin) + self.xmin
        return x
    
    def fit(self, bc, cp):
        '''
        Trains the model using the given boundary conditions and collocation points.
        
        Parameters:
            bc (tf.Tensor): The boundary conditions.
            cp (tf.Tensor): The collocation points.
            
        Returns:
            np.ndarray: The training history.
        
        Function:
            - Converts the boundary conditions and collocation points to TensorFlow tensors.
            - Extracts the input and output values from the boundary conditions.
            - Scales the input and output values using the `fit_scale` and `fit_scalex` methods.
            - Scales the collocation points using the `scalex` method.
            - Defines a function that computes the loss and gradients for a given set of parameters.
            - Trains the model using the L-BFGS optimizer and the defined function.
            - Returns the training history.
        '''
        
        bc = tf.convert_to_tensor(bc, tf.float32)
        cp = tf.convert_to_tensor(cp, tf.float32)
        
        x_bc = bc[:, :2]                        # This also needs to be changed to the values which are being compared at the boundary condition
        y_bc = bc[:, 2:]                        # This also needs to be changed to the values which are being compared at the boundary condition
        
        y_bc = self.fit_scale(y_bc)
        x_bc = self.fit_scalex(x_bc)
        
        cp = self.scalex(cp)
        bc = tf.concat([x_bc, y_bc], axis = 1)
        
        def func(params_1d):
            """
            A function that computes the loss and gradients for a given set of parameters.

            Parameters:
                params_1d (np.ndarray): The parameters for the neural network model.

            Returns:
                tuple: A tuple containing the loss and gradients for the given set of parameters.

            Function:
                - Assigns the given parameters to the neural network model.
                - Computes the loss and gradients for the current epoch using the `train_step` method.
                - Increments the epoch counter.
                - Appends the training history for the current epoch to the `hist` attribute.
                - Returns the loss and gradients for the given set of parameters.
            """
            self.sopt.assign_params(params_1d)
            tf.print('epoch:', self.epoch)
            loss, grads, hist = self.train_step(bc, cp)         # Calling of the train_step function here
            grads = tf.dynamic_stitch(self.sopt.idx, grads)
            self.epoch += 1
            self.hist.append(hist.numpy())
            return loss.numpy().astype(np.float64), grads.numpy().astype(np.float64)
        
        for epoch in range(self.epochs):
            tf.print('epoch:', self.epoch)
            loss, grads, hist = self.train_step(bc, cp)
            self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
            self.epoch += 1
            self.hist.append(hist.numpy())
            
            
        self.sopt.minimize(func)
            
        return np.array(self.hist)
    
    def predict(self, cp):
        """
        Predicts the output values for the given input values.

        Parameters:
            cp (np.ndarray): The input values.

        Returns:
            np.ndarray: The predicted output values.

        Function:
            - Converts the input values to a TensorFlow tensor.
            - Scales the input values using the `scalex` method.
            - Computes the predicted output values using the neural network model.
            - Reverses the scaling of the output values using the `scale_r` method.
            - Returns the predicted output values.
        """
        
        cp = tf.convert_to_tensor(cp, tf.float32)
        cp = self.scalex(cp)
        u_p = self.model(cp)
        u_p = self.scale_r(u_p)
        return u_p.numpy()