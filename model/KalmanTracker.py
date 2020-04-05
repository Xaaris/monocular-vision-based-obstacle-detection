import numpy as np
from filterpy.common import Q_discrete_white_noise, block_diag
from filterpy.kalman import KalmanFilter


class KalmanTracker:

    def __init__(self, initial_pos_x=0, initial_pos_y=0):
        self.kf = KalmanFilter(dim_x=4, dim_z=2)

        self.kf.x = np.array([[initial_pos_x, initial_pos_y, 0., 0.]], np.float32).T

        dt = 1.0
        self.kf.F = np.array([[1., 0., dt, 0.],  # pos.x, pos.y, vel.x, vel.y
                              [0., 1., 0., dt],
                              [0., 0., 1., 0.],
                              [0., 0., 0., 1.]])  # state transition matrix

        self.kf.H = np.array([[1., 0, 0, 0],  # pos.x, pos.y, vel.x, vel.y
                              [0., 1, 0, 0]])  # Measurement function

        self.kf.P *= 500.  # covariance matrix
        self.kf.R = np.eye(2) * 100  # measurement uncertainty
        q = Q_discrete_white_noise(2, dt=dt, var=.5, order_by_dim=False)
        self.kf.Q = block_diag(q, q)  # process uncertainty



    def update(self, center_or_none):
        self.kf.update(center_or_none)
        self.kf.predict()

    def predict_next_position(self):
        """ This function estimates the position of the object in the next time step"""
        return self.kf.x[0], self.kf.x[1]

    def get_uncertainty(self):
        """Returns a tuple with uncertainty in x and y direction"""
        covariance_matrix = self.kf.S
        covariances = np.diag(covariance_matrix)  # pos.x, pos.y, vel.x, vel.y
        return covariances[0], covariances[1]
