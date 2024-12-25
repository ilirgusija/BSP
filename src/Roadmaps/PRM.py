import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree
import utils as utils
from rtree import index

# parameter
N_SAMPLE = 100  # number of sample_points
MAX_EDGE_LEN = 30.0  # [m] Maximum edge length

class PRM:
    class Node:
        """
        Node class for dijkstra search
        """

        def __init__(self, state, cost, parent_index):
            self.state = state
            self.cost = cost
            self.parent_index = parent_index

        def __str__(self):
            return str(self.x) + "," + str(self.y) + "," +\
                str(self.cost) + "," + str(self.parent_index)

    def __init__(self, start, goal, obstacle, workspace, animation=True):
        """Initialize the PRM Planner

        Args:
            start (nparray): start state
            goal (np.array): goal state
            obstacle (list[np.array]): list of obstacles
            workspace (np.array): pair of min and max coordinates of the workspace
            animation (bool, optional): Determines whether we animate the algorithm or not. Defaults to True.
        """
        
        self.start = start
        self.goal = goal
        self.min_rand = workspace[0]
        self.max_rand = workspace[1]
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
                left, right, bottom, top = utils.generate_rectangle_from_reference(
                    ref, length, orientation, translation)
                self.obstacle.append((left, bottom, right, top))
                self.rtree.insert(i, (left, bottom, right, top))
        self.calculate_radius()

    def plot_samples(self, sample):
        if self.animation:
            print("Plotting samples and obstacles")
            utils.plot_rectangle(self.obstacle[0])
            plt.grid(True)
            plt.axis("equal")
            plt.plot(self.start[0], self.start[1],
                 c="g", marker=r"$\mathbb{S}$")
            plt.plot(self.goal[0], self.goal[1],
                 c="r", marker=r"$\mathbb{G}$")
                # Extract x and y from sample points
            sample_x = [p[0] for p in sample]
            sample_y = [p[1] for p in sample]
    
            # Plot sample points
            plt.plot(sample_x[:-2], sample_y[:-2], ".b")
            plt.pause(1)

    def plot_edges(self, road_map, sample):
        for i, neighbors in enumerate(road_map):
            for neighbor_idx in neighbors:
                # get the start and end points of the edge
                start_point = sample[i]
                end_point = sample[neighbor_idx]

                # extract x and y coordinates for plotting
                path_x = [start_point[0], end_point[0]]
                path_y = [start_point[1], end_point[1]]

                # plot the edge as a yellow line
                plt.plot(path_x, path_y, "-y")
        plt.pause(0.01)

    def plot_final_path(self, r):
        # Extract x and y coordinates from the path
        path_x = [point[0] for point in r]
        path_y = [point[1] for point in r]
        
        # Plot the final path as a red line
        plt.plot(path_x, path_y, "-r", linewidth=2, label="Final Path")
        plt.legend()
        plt.pause(0.001)
        plt.show()

    def planning(self, rng=None):
        """
        Run probabilistic road map planning

        :param start: start position
        :param goal: goal position
        :param obstacle: obstacle positions
        :param rng: (Optional) Random generator
        :return:
        """
        sample = self.sample_points(rng)
        if self.animation:
            self.plot_samples(sample)

        road_map = self.generate_road_map(sample)
        if self.animation:
            self.plot_edges(road_map, sample)

        r = self.dijkstra_planning(road_map, sample)
        
        assert r, 'Cannot find path'
        
        if self.animation:
            self.plot_final_path(r)

        return r

    def is_vertex_valid(self, vertex):
        if vertex is None:
            return False
        x, y = vertex[0], vertex[1]  # assuming vertex is [x, y]
        
        # query the r-tree directly
        possible_obstacles = list(self.rtree.intersection((x, y, x, y)))
        
        # if any obstacles match, the point is invalid
        return len(possible_obstacles) == 0
    
    def is_edge_valid(self, from_vertex, to_vertex):
        path_resolution = 0.1
        x_new = from_vertex
        d, angle = utils.calc_distance_and_angle(x_new, to_vertex)

        n_steps = math.floor(d / path_resolution)

        for _ in range(n_steps):
            x_new += path_resolution * np.array([math.cos(angle), math.sin(angle)])
            if not self.is_vertex_valid(x_new):
                return False

        return True

    def calculate_radius(self):
        d = len(self.start)
        n = N_SAMPLE

        # Calculate volume of unit ball in d dimensions
        match d:  # Source: https://en.wikipedia.org/wiki/Volume_of_an_n-ball
            case 2:
                zeta = 3.142
            case 3:
                zeta = 4.189
            case 4:
                zeta = 4.935
            case 5:
                zeta = 5.264
            case 6:
                zeta = 5.1677

        measure_tot = self.max_rand ** d  # Measure of workspace
        measure_obs = sum([(right-left) * (top-bottom) for left, right,
                          bottom, top in self.obstacle])  # Measure of obstacles
        measure_free = measure_tot - measure_obs  # Measure of free space

        # See ref [6] S. Karaman and E. Frazzoli, "Incremental Sampling-based \ 
        # Algorithms for Optimal Motion Planning" 2020
        self.radius = min((2 * (1 + 1/d) * (measure_free / zeta) * (math.log(n) / n))
                     ** (1 / d), MAX_EDGE_LEN)  # Radius based on dimensionality

    def generate_road_map(self, samples):
        """
        Road map generation

        sample: [m] positions of sampled points
        """
        print("Generating road map")
        road_map = []
        n_sample = len(samples)
        print(np.array(samples).shape)
        sample_kd_tree = KDTree(samples)

        for (_, vertex) in zip(range(n_sample), samples):
            indices, dist = sample_kd_tree.query_radius(
                np.array(vertex).reshape(1, -1), r=self.radius, return_distance=True
            )
            sorted_indices = np.argsort(dist)  # Sort indices based on distances
            indices = (indices[sorted_indices])[0]  # Sort indices accordingly
            
            edge_id = []
            for neighbor_idx in indices:
                neighbour = samples[neighbor_idx]
                if self.is_edge_valid(vertex, neighbour):
                    edge_id.append(neighbor_idx)

            road_map.append(edge_id)

        return road_map

    def dijkstra_planning(self, road_map, sample):
        """
        road_map: ??? [m]
        sample: [m]

        @return: Two lists of path coordinates ([x1, x2, ...], [y1, y2, ...]), empty list when no path was found
        """

        start_node = self.Node(self.start, 0.0, -1)
        goal_node = self.Node(self.goal, 0.0, -1)

        open_set, closed_set = dict(), dict()
        open_set[len(road_map) - 2] = start_node

        path_found = True

        while True:
            if not open_set:
                print("Cannot find path")
                path_found = False
                break

            c_id = min(open_set, key=lambda o: open_set[o].cost)
            current = open_set[c_id]

            if c_id == (len(road_map) - 1):
                print("goal is found!")
                goal_node.parent_index = current.parent_index
                goal_node.cost = current.cost
                break

            # Remove the item from the open set
            del open_set[c_id]
            # Add it to the closed set
            closed_set[c_id] = current

            # expand search grid based on motion model
            for i in range(len(road_map[c_id])):
                n_id = road_map[c_id][i]
                d = np.linalg.norm(sample[n_id] - np.array([current.state[0], current.state[1]]))
                node = self.Node(sample[n_id],
                            current.cost + d, c_id)

                if n_id in closed_set:
                    continue
                # Otherwise if it is already in the open set
                if n_id in open_set:
                    if open_set[n_id].cost > node.cost:
                        open_set[n_id].cost = node.cost
                        open_set[n_id].parent_index = c_id
                else:
                    open_set[n_id] = node

        if path_found is False:
            return [], []

        # generate final course
        r = [[goal_node.state[0], goal_node.state[1]]]
        parent_index = goal_node.parent_index
        while parent_index != -1:
            n = closed_set[parent_index]
            r.append([n.state[0], n.state[1]])
            parent_index = n.parent_index

        return r

    @staticmethod
    def plot_road_map(road_map, sample):  # pragma: no cover
        for i, _ in enumerate(road_map):
            for ii in range(len(road_map[i])):
                ind = road_map[i][ii]

                plt.plot([sample[i][0], sample[ind][0]],
                         [sample[i][1], sample[ind][1]], "-k")

    def sample_points(self, rng):
        samples = []

        if rng is None:
            rng = np.random.default_rng()

        while len(samples) <= N_SAMPLE:
            sample = [rng.random() * (self.max_rand - self.min_rand) + self.min_rand,
                      rng.random() * (self.max_rand - self.min_rand) + self.min_rand]
            
            if not self.is_vertex_valid(sample):
                continue

            samples.append(sample)

        samples.append(self.start)
        samples.append(self.goal)
        return samples