import random
from RandomTrees.RRT import RRT
from itertools import count

import matplotlib.pyplot as plt

class RRT_Connect(RRT):
    def __init__(self, start, goal, obstacle, workspace, animation=True, eta=2.5,
                 goal_sample_rate=0.01):
        super().__init__(start, goal, obstacle, workspace, animation, eta, goal_sample_rate)
        self.vertices_b = []

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

        for vertex in self.vertices_b:
            if vertex.parent:
                path_x = [p[0] for p in vertex.path]
                path_y = [p[1] for p in vertex.path]
                plt.plot(path_x, path_y, "-y")

        for o in self.obstacle:
            # Plot the blue rectangle obstacle
            self.plot_rectangle(o)

        # Plot the green start "S" and red goal "G"
        plt.plot(self.start.state[0], self.start.state[1], c="g", marker=r"$\mathbb{S}$")
        plt.plot(self.goal.state[0], self.goal.state[1], c="r", marker=r"$\mathbb{G}$")

        plt.axis("equal")
        plt.axis([self.min_rand, self.max_rand, self.min_rand, self.max_rand])
        plt.grid(True)
        plt.pause(0.01)

    def planning(self):
        self.vertices = [self.start]
        self.vertices_b = [self.goal]
        for counter in count():
            intersection = list(
                set(self.vertices).intersection(set(self.vertices_b)))
            if not len(intersection) == 0:
                self.iterations += counter
                break
            x_rand = self.sample_random_vertex()
            v_nearest = self.get_nearest_vertex(x_rand, self.vertices)
            x_new = self.steer(v_nearest, x_rand)

            if self.is_edge_valid(v_nearest, x_new):
                self.vertices.append(x_new)
                self.connect(x_new)

            if counter % 3 and self.animation is True == 0:
                self.update_graph(x_rand)

            self.vertices, self.vertices_b = self.vertices_b, self.vertices

        return self.final_paths(len(self.vertices) - 1, len(self.vertices_b) - 1)

    def connect(self, x_connect):
        v_nearest = self.get_nearest_vertex(x_connect, self.vertices_b)
        while True:
            x_step = self.steer(v_nearest, x_connect)
            if self.is_edge_valid(v_nearest, x_step):
                self.vertices_b.append(x_step)
                v_nearest = x_step

            self.iterations += 1
            if not (x_step != x_connect and self.is_edge_valid(v_nearest, x_step)):
                break
        return

    def final_paths(self, intersection_1, intersection_2):
        intersection = list(
            set(self.vertices).intersection(set(self.vertices_b)))
        if len(intersection) == 0:
            print("Error in breaking out of planning, the two sets do not share a vertex")

        path_a = [[intersection[0].state[0], intersection[0].state[1]]]
        path_b = [[intersection[0].state[0], intersection[0].state[1]]]

        vertex = self.vertices[intersection_1]
        while vertex.parent is not None:
            path_a.append(vertex.state)
            vertex = vertex.parent

        vertex_b = self.vertices_b[intersection_2]
        while vertex_b.parent is not None:
            path_b.append(vertex_b.state)
            vertex_b = vertex_b.parent

        path_a.append(vertex.state)
        path_b.append(vertex_b.state)
        path = path_a[1:][::-1] + path_b
        self.num_vertices = len(self.vertices) + len(self.vertices_b)

        return path

    def sample_random_vertex(self):
        while True:
            sampled_vec = self.Vertex([random.uniform(self.min_rand, self.max_rand),
                                 random.uniform(self.min_rand, self.max_rand)])
            if self.is_vertex_valid(sampled_vec):
                break
        return sampled_vec
