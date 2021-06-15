
from quadrotor_model import QuadRotorModel
from quadrotor_optimizer import QuadOptimizer 
import numpy as np


if __name__ == '__main__':
    quad_model = QuadRotorModel()
    opt = QuadOptimizer(quad_model.model, quad_model.constraints, t_horizon=2., n_nodes=20)
    x_init = np.zeros(13)
    x_init = np.array([0.1, 0.0, 0.67, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
    opt.simulation(x0=x_init)

