"""

LQR local path planning

author: Atsushi Sakai (@Atsushi_twi)

"""

import math
import random

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as la

class LQRPlanner:
    def __init__(self, A=None, B=None, show_animation=True):
        self.MAX_TIME = 100.0  # Maximum simulation time
        self.DT = 0.1  # Time tick
        self.GOAL_DIST = 0.0
        self.MAX_ITER = 150
        self.EPS = 0.00000000000000000000
        self.show_animation = show_animation
        self.A = np.array([[1.0, 0.0],
                           [0.0, 1.0]]) if A==None else A
        self.B = np.array([1.0, 1.0]).reshape(2, 1) if B==None else B
        # self.A = np.array([[1.0, 0.0],
        #                    [0.0, 1.0]]) if A==None else A
        # self.B = np.array([self.DT, self.DT]).reshape(2, 1) if B==None else B

    def lqr_planning(self, start, goal):
        sx, sy = start
        gx, gy = goal

        path = [start]
        K = []

        x = np.array([sx - gx, sy - gy]).reshape(2, 1)  # State vector

        # Linear system model
        A, B = self.get_system_model()

        found_path = False

        time = 0.0
        while time <= self.MAX_TIME:
            time += self.DT

            u, Kopt = self.lqr_control(A, B, x)
            K.append(Kopt)

            u = - Kopt @ x
            print(f"K: {Kopt}")
            print(f"distance x_t+1 - x_t: {la.norm(x - (A @ x + B @ u))}")
            x = A @ x + B @ u

            path.append([x[0, 0] + gx, x[1, 0] + gy])

            d = math.hypot(gx - path[-1][0], gy - path[-1][1])
            if d <= self.GOAL_DIST:
                found_path = True
                break

            # animation
            if self.show_animation:  # pragma: no cover
                # for stopping simulation with the esc key.
                plt.gcf().canvas.mpl_connect('key_release_event',
                        lambda event: [exit(0) if event.key == 'escape' else None])
                plt.plot(sx, sy, "or")
                plt.plot(gx, gy, "ob")
                # Extract x and y coordinates from the path list
                path_x = [p[0] for p in path]
                path_y = [p[1] for p in path]

                plt.plot(path_x, path_y, "-r")  # Correctly plot the x and y path coordinates
                plt.axis("equal")
                plt.pause(1.0)

        if not found_path:
            print("Cannot found path")
            return [], []

        return path, K

    def get_system_model(self):
        return self.A, self.B

    def lqr_control(self, A, B, x):

        Kopt, _, _ = self.dlqr(A, B, np.eye(2), np.eye(1))

        u = -Kopt @ x

        return u, Kopt
    
    def dlqr(self, A, B, Q, R):
        """Solve the discrete time lqr controller.
        x[k+1] = A x[k] + B u[k]
        cost = sum x[k].T*Q*x[k] + u[k].T*R*u[k]
        # ref Bertsekas, p.151
        """

        # first, try to solve the ricatti equation
        X = self.solve_dare(A, B, Q, R)

        # compute the LQR gain
        K = la.inv(B.T @ X @ B + R) @ (B.T @ X @ A)
        # K = B.T @ X

        eigValues = la.eigvals(A - B @ K)

        return K, X, eigValues
    
    def solve_dare(self, A, B, Q, R):
        """
        solve a discrete time_Algebraic Riccati equation (DARE)
        """
        X, Xn = Q, Q

        for _ in range(self.MAX_ITER):
            Xn = A.T * X * A \
                - A.T * X * B * la.inv(R + B.T * X * B) * B.T * X * A \
                + Q
            if (abs(Xn - X)).max() < self.EPS:
                break
            X = Xn

        return Xn

def main():
    print(__file__ + " start!!")

    ntest = 10  # number of goal
    area = 100.0  # sampling area

    lqr_planner = LQRPlanner()

    for i in range(ntest):
        sx = 6.0
        sy = 6.0
        gx = random.uniform(-area, area)
        gy = random.uniform(-area, area)

        path, _ = lqr_planner.lqr_planning([sx, sy], [gx, gy])

        rx, ry = [x for (x, y) in path], [y for (x, y) in path]

        if lqr_planner.show_animation:  # pragma: no cover
            plt.plot(sx, sy, "or")
            plt.plot(gx, gy, "ob")
            plt.plot(rx, ry, "-r")
            plt.axis("equal")
            plt.pause(1.0)


if __name__ == '__main__':
    main()