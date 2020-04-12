import numpy as np
from filterpy.common import Q_discrete_white_noise
from filterpy.kalman import KalmanFilter


class KalmanTracker:

    def __init__(self, initial_pos_x=0, initial_pos_y=0):
        self.kf = KalmanFilter(dim_x=6, dim_z=2)

        self.kf.x = np.array([[initial_pos_x, initial_pos_y, 0., 0., 0., 0.]], np.float32).T

        dt = 1.
        self.kf.F = np.array([[1., 0., dt, 0., dt ** 2 / 2, 0.],  # pos.x, pos.y, vel.x, vel.y, acc.x, acc.y
                              [0., 1., 0., dt, 0., dt ** 2 / 2],
                              [0., 0., 1., 0., dt, 0.],
                              [0., 0., 0., 1., 0., dt],
                              [0., 0., 0., 0., 1., 0.],
                              [0., 0., 0., 0., 0., 1.]])  # state transition matrix

        self.kf.H = np.array([[1., 0, 0, 0, 0, 0],  # pos.x, pos.y, vel.x, vel.y, acc.x, acc.y
                              [0., 1, 0, 0, 0, 0]])  # Measurement function

        self.kf.P *= 1000.  # covariance matrix
        self.kf.R = np.eye(2) * 50  # measurement uncertainty
        self.kf.Q = Q_discrete_white_noise(2, dt=dt, var=.1, block_size=3, order_by_dim=False)  # process uncertainty



    def update(self, center_or_none):
        if center_or_none:
            self.kf.update(center_or_none)
        self.kf.predict()

    def predict_next_position(self):
        """ This function estimates the position of the object in the next time step"""
        return self.kf.x[0], self.kf.x[1]

    def get_uncertainty(self):
        """Returns a tuple with uncertainty in x and y direction"""
        covariance_matrix = self.kf.P
        covariances = np.diag(covariance_matrix)  # pos.x, pos.y, vel.x, vel.y, acc.x, acc.y
        return covariances[0], covariances[1]
