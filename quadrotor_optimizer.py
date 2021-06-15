#!/usr/bin/env python
# coding: utf-8
'''
Author: Wei Luo
Date: 2021-03-14 22:01:33
LastEditors: Wei Luo
LastEditTime: 2021-03-20 00:01:03
Note: Note
'''

import os
import sys
# from utils.utils import safe_mkdir_recursive
from quadrotor_model import QuadRotorModel
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSimSolver
import casadi as ca
import scipy.linalg
import numpy as np
import time

import matplotlib.pyplot as plt

class QuadOptimizer:
    def __init__(self, quad_model, quad_constraints, t_horizon, n_nodes, ):
        model = quad_model
        self.constraints = quad_constraints
        self.g = 9.81
        self.T = t_horizon
        self.N = n_nodes
        Q_m_ = np.diag([10.0, 10.0, 10.0,
                        0.3, 0.3, 0.3,
                        1.0, 1.0, 0.03, 0.03,
                        0.5, 0.5, 0.5])  # position, velocity, load_position, load_velocity, roll, pitch, yaw

        P_m_ = np.diag([10.0, 10.0, 10.0,
                        0.05, 0.05, 0.05
                        # 0.1, 0.1, 0.1, 0.1,
                        # 0.1, 0.1, 0.1
                        ])  # only p and v
        # P_m_[0, 8] = 6.45
        # P_m_[8, 0] = 6.45
        # P_m_[1, 9] = 6.45
        # P_m_[9, 1] = 6.45
        # P_m_[2, 10] = 10.95
        # P_m_[10, 2] = 10.95
        R_m_ = np.diag([3.0, 3.0, 3.0, 1.0])
        # Ensure current working directory is current folder
        # os.chdir(os.path.dirname(os.path.realpath(__file__)))
        # self.acados_models_dir = './acados_models'
        # safe_mkdir_recursive(os.path.join(os.getcwd(), self.acados_models_dir))

        nx = model.x.size()[0]
        self.nx = nx
        nu = model.u.size()[0]
        self.nu = nu
        ny = nx + nu
        n_params = model.p.size()[0] if isinstance(model.p, ca.SX) else 0

        acados_source_path = os.environ['ACADOS_SOURCE_DIR']
        sys.path.insert(0, acados_source_path)

        # create OCP
        ocp = AcadosOcp()
        ocp.acados_include_path = acados_source_path + '/include'
        ocp.acados_lib_path = acados_source_path + '/lib'
        ocp.model = model
        ocp.dims.N = self.N
        ocp.solver_options.tf = t_horizon

        # initialize parameters
        ocp.dims.np = n_params
        ocp.parameter_values = np.zeros(n_params)

        # cost type
        ocp.cost.cost_type = 'LINEAR_LS'
        ocp.cost.cost_type_e = 'LINEAR_LS'
        ocp.cost.W = scipy.linalg.block_diag(Q_m_, R_m_)
        ocp.cost.W_e = P_m_ # np.zeros((nx-3, nx-3))

        ocp.cost.Vx = np.zeros((ny, nx))
        ocp.cost.Vx[:nx, :nx] = np.eye(nx)
        ocp.cost.Vu = np.zeros((ny, nu))
        ocp.cost.Vu[-nu:, -nu:] = np.eye(nu)
        ocp.cost.Vx_e = np.zeros((nx-7, nx)) # only consider p and v
        ocp.cost.Vx_e[:nx-7, :nx-7] = np.eye(nx-7)
        

        # initial reference trajectory_ref
        x_ref = np.zeros(nx)
        x_ref_e = np.zeros(nx-7)
        u_ref = np.zeros(nu)
        u_ref[-1] = self.g
        ocp.cost.yref = np.concatenate((x_ref, u_ref))
        ocp.cost.yref_e = x_ref_e

        # Set constraints
        ocp.constraints.lbu = np.array(
            [self.constraints.roll_min, self.constraints.pitch_min, self.constraints.yaw_min, self.constraints.thrust_min])
        ocp.constraints.ubu = np.array(
            [self.constraints.roll_max, self.constraints.pitch_max, self.constraints.yaw_max, self.constraints.thrust_max])
        ocp.constraints.idxbu = np.array([0, 1, 2, 3])

        # initial state
        ocp.constraints.x0 = x_ref

        # solver options
        ocp.solver_options.qp_solver = 'FULL_CONDENSING_HPIPM'
        ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
        # explicit Runge-Kutta integrator
        ocp.solver_options.integrator_type = 'ERK'
        ocp.solver_options.print_level = 0
        ocp.solver_options.nlp_solver_type = 'SQP' # 'SQP_RTI'

        # compile acados ocp
        json_file = os.path.join('./'+model.name+'_acados_ocp.json')
        self.solver = AcadosOcpSolver(ocp, json_file=json_file)
        self.integrator = AcadosSimSolver(ocp, json_file=json_file)

    def trajectory_generator(self, iter, current_state, current_trajectory):
        next_trajectories = current_trajectory[1:, :]
        next_trajectories = np.concatenate((next_trajectories,
        np.array([np.cos((iter)/30), np.sin((iter)/30), 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).reshape(1, -1)))
        return next_trajectories


    def simulation(self, x0):
        sim_time = 300 # s
        dt = 0.1 # s
        simX = np.zeros((int(sim_time/dt+1), self.nx))
        simD = np.zeros((int(sim_time/dt+1), self.nx))
        simU = np.zeros((int(sim_time/dt), self.nu))
        x_current = x0
        simX[0, :] = x0.reshape(1, -1)
        simD[0, :] = x0.reshape(1, -1)
        init_trajectory = np.array(
                [
                # [0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                # [0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                # [0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                # [0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                # [0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                # [0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                # [0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                # [0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                # [0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                # [0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                # [0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                # [0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                # [0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                # [0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                # [0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                # [0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                # [0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                # [0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                # [0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                # [0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                # [0.0, 0.0, 0.4, 0.0, 0.0, 0.01, 0.0, 0.0, 0.0],
                # [0.0, 0.0, 0.4, 0.0, 0.0, 0.01, 0.0, 0.0, 0.0],
                # [0.0, 0.0, 0.4, 0.0, 0.0, 0.01, 0.0, 0.0, 0.0],
                # [0.0, 0.0, 0.4, 0.0, 0.0, 0.01, 0.0, 0.0, 0.0],
                # [0.0, 0.0, 0.4, 0.0, 0.0, 0.01, 0.0, 0.0, 0.0],
                # [0.0, 0.0, 0.4, 0.0, 0.0, 0.01, 0.0, 0.0, 0.0],
                # [0.0, 0.0, 0.4, 0.0, 0.0, 0.01, 0.0, 0.0, 0.0],
                # [0.0, 0.0, 0.4, 0.0, 0.0, 0.01, 0.0, 0.0, 0.0],
                # [0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                # [0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                # [0.1, 0.0, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                # [0.1, 0.0, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                # [0.1, 0.0, 0.67, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.1, 0.0, 0.67, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.1, 0.0, 0.67, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.1, 0.0, 0.67, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.1, 0.0, 0.67, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.1, 0.0, 0.67, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.1, 0.0, 0.67, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.1, 0.0, 0.67, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.1, 0.0, 0.67, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.1, 0.0, 0.67, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.1, 0.0, 0.67, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.1, 0.0, 0.67, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.1, 0.0, 0.67, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.1, 0.0, 0.67, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.1, 0.0, 0.67, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.1, 0.0, 0.67, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.1, 0.0, 0.67, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.1, 0.0, 0.67, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.1, 0.0, 0.67, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.1, 0.0, 0.67, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.1, 0.0, 0.67, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.1, 0.0, 0.67, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                ]) # 20x13

        mpc_iter = 0
        current_trajectory = init_trajectory.copy()
        u_des = np.array([0.0, 0.0, 0.0, self.g])

        # current_trajectory = np.array([1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        t1 = time.time()
        while(mpc_iter < sim_time/dt and mpc_iter < 208):
            # define cost constraints
            self.solver.set(self.N, 'yref', current_trajectory[-1, :6])
            # self.solver.set(0, 'yref', np.concatenate([x_current, u_des]))
            for i in range(self.N):
                self.solver.set(i, 'yref', np.concatenate([current_trajectory[i], u_des]))

            self.solver.set(0, 'lbx', x_current)
            self.solver.set(0, 'ubx', x_current)
            status = self.solver.solve()
            if status != 0 :
                raise Exception('acados acados_ocp_solver returned status {}. Exiting.'.format(status))
            simU[mpc_iter, :] = self.solver.get(0, 'u')

            # print(current_trajectory)
            # print('-----')
            # for i in range(self.N):
            #     print(self.solver.get(i, 'x'))
            # print('-----')
            # print(simU[mpc_iter, :])
            # simulated system
            self.integrator.set('x', x_current)
            self.integrator.set('u', simU[mpc_iter, :])
            status = self.integrator.solve()
            if status != 0:
                raise Exception('acados integrator returned status {}. Exiting.'.format(status))
            # update
            x_current = self.integrator.get('x')
            # print(x_current)
            # print('-----')
            # print('x des {}'.format(current_trajectory[0]))
            simX[mpc_iter+1, :] = x_current
            simD[mpc_iter+1, :] = current_trajectory[0]
            # get new trajectory_ref
            current_trajectory = self.trajectory_generator(mpc_iter, x_current, current_trajectory)
            # next loop
            mpc_iter += 1

        print('average time is {}'.format((time.time()-t1)/mpc_iter))

        l = 1.
        hX = np.sqrt(l*l - np.square(simX[:mpc_iter,6]) - np.square(simX[:mpc_iter, 7]))
        hD = np.sqrt(l*l - np.square(simD[:mpc_iter,6]) - np.square(simD[:mpc_iter, 7]))
        loadX = np.array([simX[:mpc_iter, 0]+simX[:mpc_iter, 6], simX[:mpc_iter,1]+simX[:mpc_iter, 7], simX[:mpc_iter, 2]-hX])
        loadD = np.array([simD[:mpc_iter, 0]+simD[:mpc_iter, 6], simD[:mpc_iter,1]+simD[:mpc_iter, 7], simD[:mpc_iter, 2]-hD])

        alphaX = np.rad2deg(np.arcsin(simX[:mpc_iter, 6] / l))
        alphaD = np.rad2deg(np.arcsin(simD[:mpc_iter, 6] / l))

        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.plot(simX[:mpc_iter, 0], simX[:mpc_iter, 1], simX[:mpc_iter, 2], 'b')
        ax.plot(simD[:mpc_iter, 0], simD[:mpc_iter, 1], simD[:mpc_iter, 2], 'b--')
        ax.plot(loadX[0], loadX[1], loadX[2], 'r')
        ax.plot(loadD[0], loadD[1], loadD[2], 'r--')
        plt.show()
        fig = plt.figure()
        ax = fig.gca()
        ax.plot(range(mpc_iter), simX[:mpc_iter, 0], )
        ax.plot(range(mpc_iter), simD[:mpc_iter, 0], )
        plt.title("x coordinate of quadcopter")
        plt.show()
        fig = plt.figure()
        ax = fig.gca()
        ax.plot(range(mpc_iter), loadX[0], )
        ax.plot(range(mpc_iter), loadD[0], )
        plt.title("x coordinate of pendulum")
        plt.show()
        fig = plt.figure()
        ax = fig.gca()
        ax.plot(range(mpc_iter), alphaX, )
        ax.plot(range(mpc_iter), alphaD, )
        plt.title("alpha angle of pendulum")
        plt.show()

if __name__ == '__main__':
    quad_model = QuadRotorModel()
    opt = QuadOptimizer(quad_model.model, quad_model.constraints, t_horizon=3., n_nodes=10)
    opt.simulation(x0=np.zeros(13))