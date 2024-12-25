import math
import random
from itertools import count
import utils as utils
from rtree import index

import matplotlib.pyplot as plt
import numpy as np

class RRT:
    class Vertex:
        def __init__(self, state):
            self.state = np.array(state, dtype=float)
            self.path = [(tuple(self.state))]  # List of tuples storing path points
            self.parent = None

        def __eq__(self, other):
            if isinstance(other, self.__class__):
                return np.array_equal(self.state, other.state)
            return False

        def __hash__(self):
            return hash(tuple(self.state))

        def dimension(self):
            """Return dimension of the state space"""
            return len(self.state)

    def __init__(self, start, goal, obstacle, workspace, animation=True, eta=2.5,
                 goal_sample_rate=0.01):
        """
        start:Start Position [x,y]
        goal:Goal Position [x,y]
        obstacle:Coordinates of rectangle obstacle [left,right,bottom,top]
        workspace:Min/max coordinates of our square arena [min,max]
        """
        self.start = self.Vertex(start)
        self.goal = self.Vertex(goal)
        self.min_rand = workspace[0]
        self.max_rand = workspace[1]
        self.eta = eta
        self.goal_sample_rate = goal_sample_rate
        self.animation = animation
        self.obstacle = []
        self.rtree = index.Index()
        if isinstance(obstacle, tuple):
            left, right, bottom, top = o = utils.convert_to_rectangle(
                obstacle[0], obstacle[1], self.min_rand, self.max_rand)
            self.obstacle.append(o)
            self.rtree.insert(0, (left, bottom, right, top))
        elif isinstance(obstacle, list):
            for i, o in enumerate(obstacle):
                ref, length, orientation, translation = o
                left, right, bottom, top = self.generate_rectangle_from_reference(
                    ref, length, orientation, translation)
                self.obstacle.append((left, bottom, right, top))
                self.rtree.insert(i, (left, bottom, right, top))
        self.vertices = []
        self.iterations = 0
        self.num_vertices = 0

    def planning(self):
        self.vertices = [self.start]
        for self.iterations in count():
            if self.goal in self.vertices:
                break
            x_rand = self.Vertex(self.sample_random_vertex())
            v_nearest = self.get_nearest_vertex(x_rand.state, self.vertices)
            x_new = self.steer(v_nearest, x_rand)

            if self.is_edge_valid(v_nearest, x_new):
                self.vertices.append(x_new)

            if self.iterations % 3 == 0 and self.animation is True:
                self.update_graph(x_rand)

        return self.final_path(len(self.vertices) - 1)

    def steer(self, v_nearest, x_rand):
        x_new = self.Vertex(v_nearest.state)
        d, angle = utils.calc_distance_and_angle(x_new.state, x_rand.state)

        x_new.path = [x_new.state]
        if self.eta < d:
            x_new.state = [x_new.state[0] + self.eta * math.cos(angle),
                          x_new.state[1] + self.eta * math.sin(angle)]
        else:
            x_new.state = [x_new.state[0] + d * math.cos(angle),
                          x_new.state[1] + d * math.sin(angle)]
        
        x_new.path.append(x_new.state)
        x_new.parent = v_nearest

        return x_new

    def is_vertex_valid(self, vertex):
        if vertex is None:
            return False
        x, y = vertex[0], vertex[1]  # assuming vertex is [x, y]
        
        # query the r-tree directly
        possible_obstacles = list(self.rtree.intersection((x, y, x, y)))
        
        # if any obstacles match, the point is invalid
        return len(possible_obstacles) == 0

    def is_edge_valid(self, v_nearest, x_rand):
        path_resolution = 0.1
        x_new = self.Vertex(v_nearest.state)
        d, angle = utils.calc_distance_and_angle(x_new.state, x_rand.state)
        if not self.is_vertex_valid(x_rand.state):
            return False
        x_new.path = [x_new.state]

        if self.eta > d:
            n_steps = math.floor(d / path_resolution)
        else:
            n_steps = math.floor(self.eta / path_resolution)

        for _ in range(n_steps):
            x_new.state += path_resolution * np.array([math.cos(angle), math.sin(angle)])
            if not self.is_vertex_valid(x_new.state):
                return False
            x_new.path.append(x_new.state)

        d, _ = utils.calc_distance_and_angle(x_new.state, x_rand.state)
        if d <= path_resolution:
            x_new.path.append(x_rand.state)
            x_new.state = x_rand.state

        return True

    def update_graph(self, sampled_vec=None):
        plt.clf()
        # Plot the sampled vector as a black plus sign
        if sampled_vec is not None:
            plt.plot(sampled_vec.state[0], sampled_vec.state[1], "Pk")

        # Plot edges as yellow lines
        for vertex in self.vertices:
            if vertex.parent:
                path_x = [p[0] for p in vertex.path]
                path_y = [p[1] for p in vertex.path]
                plt.plot(path_x, path_y, "-y")

        for o in self.obstacle:
            # Plot the blue rectangle obstacle
            utils.plot_rectangle(o)

        # Plot the green start "S" and red goal "G"
        plt.plot(self.start.state[0], self.start.state[1],
                 c="g", marker=r"$\mathbb{S}$")
        plt.plot(self.goal.state[0], self.goal.state[1],
                 c="r", marker=r"$\mathbb{G}$")

        plt.axis("equal")
        plt.axis([self.min_rand, self.max_rand, self.min_rand, self.max_rand])
        plt.grid(True)
        plt.pause(0.01)

    def sample_random_vertex(self, d=2):
        if random.random() <= self.goal_sample_rate:
            sampled_vec = self.goal.state
        else:
            while True:
                sampled_vec = np.array([random.uniform(self.min_rand, self.max_rand),
                                               random.uniform(self.min_rand, self.max_rand)])
                if self.is_vertex_valid(sampled_vec) is True:
                    break
        return sampled_vec

    def get_nearest_vertex(self, x_rand, vertices):
        dlist = [np.linalg.norm(vertex.state - x_rand) for vertex in vertices]
        minind = dlist.index(min(dlist))
        return vertices[minind]

    def final_path(self, g_idx):
        path = [[self.goal.state[0], self.goal.state[1]]]
        vertex = self.vertices[g_idx]
        while vertex.parent is not None:
            path.append([vertex.state[0], vertex.state[1]])
            vertex = vertex.parent
        path.append([vertex.state[0], vertex.state[1]])
        self.num_vertices = len(self.vertices)
        return path