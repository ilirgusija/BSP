import math
import numpy as np
import numpy.linalg as la
from RandomTrees.RRT import RRT
from itertools import count
from sklearn.neighbors import KDTree
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
import utils.utils as utils
from Controller.LQRPlanner import LQRPlanner
import control as ct
import sys

LARGE_VALUE = 1e11  # Large number to represent high uncertainty
SMALL_VALUE = 0.001  # Small number to represent low uncertainty

class RRBT(RRT):
    class Vertex:
        def __init__(self, state):
            """
            Initialize a vertex in R^n with associated belief nodes

            Args:
                state: numpy array representing position in R^n
            """
            self.state = np.array(state, dtype=float)
            self.nodes = []  # List of BeliefNodes associated with this vertex
            self.path = []  # List to store path points

        def __eq__(self, other):
            if isinstance(other, self.__class__):
                return np.array_equal(self.state, other.state)
            return False

        def __hash__(self):
            return hash(tuple(self.state))

        # Define function for printing vertex
        def __repr__(self):
            return f"Vertex({self.state})"

        def dimension(self):
            """Return dimension of the state space"""
            return len(self.state)

        def add_belief_node(self, node):
            """Add a belief node to this vertex"""
            self.nodes.append(node)

        def get_best_node(self):
            """Get the belief node that is not dominated by any other node"""
            if not self.nodes:
                return None

            best_node = self.nodes[0]
            for node in self.nodes[1:]:
                if self.dominates(node, best_node):
                    best_node = node

            return best_node
        
        def get_best_cov(self):
            """Get the belief node with the smallest covariance matrix"""
            if not self.nodes:
                return None

            best_node = self.get_best_node()
            return best_node.sigma + best_node._lambda

        def dominates(self, node1, node2):
            """
            Implements the partial ordering criteria to determine if node1 dominates node2.
            A node dominates another if it has both:
            1. Lower cost-to-come
            2. Lower uncertainty (covariance matrix is "smaller")

            Args:
                node1: First BeliefNode to compare
                node2: Second BeliefNode to compare

            Returns:
                bool: True if node1 dominates node2, False otherwise
            """
            if node1 is None or node2 is None:
                return False

            # sigma_dominates = np.linalg.det(node1.sigma) <= np.linalg.det(node2.sigma)
            sigma_dominates = np.trace(node1.sigma) <= np.trace(node2.sigma)
            # lambda_dominates = np.linalg.det(node1._lambda) <= np.linalg.det(node2._lambda)
            lambda_dominates = np.trace(node1._lambda) <= np.trace(node2._lambda)
            cost_dominates = node1.cost <= node2.cost
            
            # sigma_dominates_i = np.linalg.det(node1.sigma) < np.linalg.det(node2.sigma)
            sigma_dominates_i = np.trace(node1.sigma) < np.trace(node2.sigma)
            # lambda_dominates_i = np.linalg.det(node1._lambda) < np.linalg.det(node2._lambda)
            lambda_dominates_i = np.trace(node1._lambda) < np.trace(node2._lambda)
            cost_dominates_i = node1.cost < node2.cost

            f_le_g = sigma_dominates and lambda_dominates and cost_dominates
            f_i_le_g_i = sigma_dominates_i or lambda_dominates_i or cost_dominates_i

            if f_le_g and f_i_le_g_i:
                return True
            else:
                return False

    class BeliefNode:
        def __init__(self, state, sigma=None, _lambda=None, cost=float('inf'), parent=None):
            """
            A node maintaining belief information for a state

            Args:
                state: The state this belief node represents
                sigma: Covariance matrix
                _lambda: Information matrix (defaults to small non-zero values for realistic uncertainty)
            """
            I = np.eye(2)
            self.state = state  # Store the state
            self.sigma = I if sigma is None else sigma
            self._lambda = np.zeros((2,2)) if _lambda is None else _lambda
            self.cost = cost  # Cost-to-come
            self.parent = parent  # Parent belief node

        def __eq__(self, other):
            if isinstance(other, self.__class__):
                return (np.array_equal(self.sigma, other.sigma) and
                        np.array_equal(self._lambda, other._lambda))
            return False
        
        def get_uncertainty(self):
            """Return the uncertainty of this belief node"""
            return np.trace(self.sigma+self._lambda)

    def __init__(self, start, goal_region, obstacle, workspace, measurement_zone, animation=True, eta=10, goal_sample_rate=0.01, delta=0.3, process_noise=0.01, iters=1000, debug=True):
        
        super().__init__(start=start[0], goal=goal_region[0], obstacle=obstacle, workspace=workspace, animation=animation, eta=eta, goal_sample_rate=goal_sample_rate)

        # Create initial belief nodes
        start_belief = self.BeliefNode(state=self.start.state, sigma=start[1], cost=0)  # Initial belief
        self.start.add_belief_node(start_belief)
        
        # Create goal region
        self.goal_region = self.convert_to_square(goal_region[0], goal_region[1], self.min_rand, self.max_rand)
        
        # Key: tuple(start_vertex, end_vertex)
        # Value: tuple(X_nominal, U_nominal, K_gains)
        self.edges = {}
        
        self.vertices = [self.start]
        
        # Process noise covariance
        I = np.eye(2)
        self.Q = process_noise * I
        self.measurement_zone = [measurement_zone]
        self.delta = delta
        self.init_radius_vals()
        self.max_iters = iters
        self.best_prob = 0.0
        self.best_path = False
        self.init_dynamics()
        self.debug = debug

    def planning(self):
        """Main RRBT algorithm"""
        goal_vertex = False
        for self.iterations in count():
            # Check if we've reached the goal region
            res = self.goal_reached()
            if res is not False:
                goal_vertex, p_goal = res
                self.best_path = self.final_path(goal_vertex)
                self.best_prob = p_goal
                goal_uncertainty = np.trace(goal_vertex.get_best_cov())
                if self.animation is True:
                    self.update_graph()
                # if self.debug:
                print("Goal reached with probability:", self.best_prob)
                print("Goal uncertainty:", goal_uncertainty)
                break
            
            x_rand = self.sample_random_vertex()  # Sample random belief state
            v_nearest = self.get_nearest_vertex(x_rand, self.vertices)  # Find nearest vertex in tree
            edge_new = self.Connect(v_nearest.state, x_rand)
            if edge_new is False or len(v_nearest.nodes) == 0:
                # Edge is invalid (e.g., same vertex or obstacle in the way)
                # or v_nearest has no belief nodes
                continue

            # Propagate belief along edge
            belief = self.Propagate(edge_new, v_nearest.get_best_node())
            if (v_nearest.get_best_node().state != v_nearest.state).all():
                raise ValueError(f"Error: v_nearest best node: {v_nearest.get_best_node().state} is not v_nearest: {v_nearest.state}") 

            # If belief is None, constraints were violated
            if belief is not False:
                queue = []
                # Add new vertex and edge to tree
                v_rand = self.Vertex(x_rand)
                v_rand.add_belief_node(belief)
                self.vertices.append(v_rand)
                self.add_edge(v_nearest, v_rand, edge_new)

                # Add reverse edge for bidirectional tree
                reverse_edge = self.Connect(x_rand, v_nearest.state)
                if reverse_edge is not False:
                    self.add_edge(v_rand, v_nearest, reverse_edge)

                # Although initially counter-intuitive, we would like to rewire the tree from the nearest vertex
                queue.append(v_nearest)

                # Find the vertices within some ball of our newly sampled vertex, v_rand
                nearby_vertices = self.Near(v_rand)

                # Connect v_rand with nearby vertices
                for v_near in nearby_vertices:
                    edge = self.Connect(v_near.state, x_rand)
                    if edge is False:
                        continue
                    self.add_edge(v_near, v_rand, edge)
                    reverse_edge = self.Connect(x_rand, v_near.state)
                    if reverse_edge is not False:
                        self.add_edge(v_rand, v_near, reverse_edge)
                    if v_near != v_nearest:
                        queue.append(v_near)

                # Rewire nearby vertices
                self.rewire(queue)
                if self.animation is True and self.iterations % 5 == 0:
                    self.update_graph(x_rand)

        return self.best_path, self.best_prob, goal_uncertainty
    
    def goal_reached(self):
        """Check if goal has been reached with acceptable uncertainty"""
        for vertex in self.vertices:
            # Compute probability of being inside goal using CDF
            p_goal = self.CDF(self.goal_region, vertex)
            
            # Check if probability is less than delta threshold
            if (1 - p_goal) < self.delta:
                return vertex, p_goal
        return False

    def Connect(self, from_state, to_state):
        """
        Connect two vertices with a nominal trajectory, control inputs, and stabilizing feedback gains
        according to equations (3)-(9).

        Args:
            from_vertex: Starting vertex (x^a)
            to_vertex: Target vertex (x^b)

        Returns:
            Tuple: (X̌^(a,b), Ǔ^(a,b), Ǩ^(a,b)) containing:
                - Nominal state trajectory
                - Nominal control inputs
                - Feedback gains
        """
        # Initialize trajectory lists
        X_nominal = [from_state]  # X̌^(a,b) = (x̃₀, x̃₁, ..., x̃ₜ)
        to_state = np.array(to_state)
        from_state = np.array(from_state)

        if (to_state == from_state).all():
            if self.debug:
                print(f"from_state: {from_state} and to_state: {to_state}")
            return False
        # Initial conditions (eq. 6)
        x_t = from_state
        
        # Generate trajectory using system dynamics (eq. 7)
        d = np.linalg.norm(to_state - from_state)
        n_steps = int(d / self.dt)  # Number of steps
        
        for i in range(n_steps+1):
            dt = self.dt
            # Generate nominal control input
            direction = to_state - x_t
            if i == n_steps:
                dt = np.linalg.norm(direction)
                if not (0 < dt < 0.1):
                    raise ValueError("Connect error: path resolution out of bounds.")
            u_t = dt * direction / np.linalg.norm(direction)
            # U_nominal.append(u_t)

            # Propagate state using nominal dynamics f(x̃ₜ₋₁, ũₜ₋₁, 0)
            x_ = x_t + u_t  # Simple integrator model 
            if not self.is_vertex_valid(x_):
                return False
            X_nominal.append(x_)
            x_t = x_
            
        if (X_nominal[-1] != to_state).all():
            raise ValueError(f"Error: X_nominal[-1]: {X_nominal[-1]} is not to_state: {to_state}")
        if (len(X_nominal)==1):
            raise ValueError(f"Error: X_nominal: {X_nominal} is not a path.")
        return np.array(X_nominal)

    def kalman_filter(self, t, parent, X):
        R = self.get_measurement_covariance(X[t])
        
        # covariance prediction (eq. 21, 33)
        sigma_bar = self.A @ parent.sigma @ self.A.T + self.Q  # (eq. 17)
        S_t = self.C @ sigma_bar @ self.C.T + R  # (eq. 18)
        L_t = (sigma_bar @ self.C.T) @ np.linalg.inv(S_t)  # (eq. 19)
        sigma = sigma_bar - (L_t @ self.C) @ sigma_bar  # (eq. 21)
        _lambda = (self.A - self.B @ self.K) @ parent._lambda @ (self.A - self.B @ self.K).T + L_t @ self.C @ sigma_bar  # (eq. 33)

        vertex = self.Vertex(X[t+1])
        node = self.BeliefNode(state=X[t+1], sigma=sigma, _lambda=_lambda, parent=parent)
        node.cost = parent.cost + self.edge_cost(parent.state, X[t+1])  # cost (eq. 11)
        vertex.add_belief_node(node)
        # chance constraint check (eq.13)
        if not self.obstacle_free(vertex):
            return None  # indicates failure due to constraints
        return node

    def Propagate(self, edge, n_start):
        """
        Propagate belief along an edge by computing covariance matrices and checking chance constraints.

        Args:
            edge: Tuple (X̌^(a,b), Ǔ^(a,b), Ǩ^(a,b)) containing nominal trajectory, controls, and gains
            n_start: Starting vertex with belief nodes

        Returns:
            BeliefNode: Final belief node if propagation succeeds, False if constraints violated
        """
        X = edge
        edge_len = len(X)
        if (n_start.state != X[0]).all():
            if self.debug:
                print(f"n_start: {n_start.state} is not X[0]: {X[0]}")
            raise ValueError("Propagate error: n_start is not X[0]")
        parent = n_start
        for t in range(1,edge_len-1):
            node = self.kalman_filter(t, parent, X)
            if node is None:
                return False
            parent = node

        if (node.state != X[-1]).all():
            if self.debug:
                print(f"Propagate error: node, {node.state} is not X[-1]: {X[-1]}")
            raise ValueError("Propagate error: node is not X[-1]")
        return node

    def Near(self, vertex):
        """Find nearby vertices within a certain radius"""
        radius = self.calculate_radius()
        kd_tree = KDTree(np.array([v.state for v in self.vertices]))
        # query neighbors within the radius
        indices, dist = kd_tree.query_radius(
            np.array(vertex.state).reshape(1, -1), r=radius, return_distance=True
        )
        sorted_indices = np.argsort(dist)  # Sort indices based on distances
        indices = (indices[sorted_indices])[0]  # Sort indices accordingly

        # filter neighbors based on the logic below ########
        nearby_vertices = [
            self.vertices[i]
            for i in indices
            if not np.array_equal(self.vertices[i].state, vertex.state)  # exclude the same vertex
        ]
        return nearby_vertices

    def rewire(self, queue):
        """Rewire nearby vertices through new vertex if it provides lower cost"""
        while len(queue) != 0:
            # Pop vertex from the queue
            v = queue.pop(0)
            # Get all possible "neighbors" of the vertex
            neighbours = self.Near(v)
            for v_neighbour in neighbours:
                # Check if they're actually neighbours
                e_neighbour = self.get_edge(v, v_neighbour)
                if e_neighbour is None:
                    continue
                # if (e_neighbour[-1] != v_neighbour.state).all():
                #     raise ValueError(f"Error: edge end: {e_neighbour[-1]} is not v_neighbour: {v_neighbour.state}")
                # If so, propagate the belief along the edge for each belief in the parent node
                for node in v.nodes:
                    node_new = self.Propagate(e_neighbour, node)  # returns n_end
                    # if node_new is valid AND has lower cost than all nodes in v_neighbour we add it to the belief nodes
                    if node_new is not False and self.AppendBelief(v_neighbour, node_new):
                        # if (node_new.state != e_neighbour[-1]).all():
                        #     raise ValueError(f"Error: node_new: {node_new.state} is not e_neighbour[-1]: {e_neighbour[-1]}")
                        vertex_idx = self.get_vertex_idx(node_new.state)
                        # if vertex_idx is None:
                        #     raise ValueError(f"Error: vertex: {node_new.state} not found")
                        self.vertices[vertex_idx].add_belief_node(node_new)
                        queue.append(self.vertices[vertex_idx])

    def update_graph(self, sampled_vec=None):
        plt.clf()
        # Plot the sampled vector as a black plus sign
        for m in self.measurement_zone:
            # Plot the red measurement zone
            utils.plot_rectangle(m, color="-g")
            
        for o in self.obstacle:
            # Plot the blue rectangle obstacle
            utils.plot_rectangle(o, color="-b")

        if sampled_vec is not None:
            plt.plot(sampled_vec[0], sampled_vec[1], "Pk")

        # Plot edges as yellow lines
        for edge in self.edges.values():
            path_x = [state[0] for state in edge]
            path_y = [state[1] for state in edge]
            plt.plot(path_x, path_y, "-y")

        # Plot the covariance of each vertex's best belief node
        for vertex in self.vertices:
            best_node = vertex.get_best_node()
            if best_node is not None:
                self.plot_covariance(vertex.state, best_node.sigma + best_node._lambda)
                
        if self.best_path is not False:
            """Plot the final path found by the algorithm"""
            plt.plot([x for (x, _) in self.best_path], [y for (_, y) in self.best_path], '-r')
            plt.grid(True)
            plt.pause(0.01)

        # Plot the green start "S" and red goal "G"
        utils.plot_rectangle(self.goal_region, color="-b")
        plt.plot(self.start.state[0], self.start.state[1],
                 c="g", marker=r"$\mathbb{S}$")
        plt.plot(self.goal.state[0], self.goal.state[1],
                 c="r", marker=r"$\mathbb{G}$")

        plt.axis("equal")
        plt.axis([self.min_rand, self.max_rand, self.min_rand, self.max_rand])
        plt.grid(True)
        plt.pause(0.01)

    def final_path(self, goal_vertex):
        vertex = goal_vertex
        best_node = vertex.get_best_node()
        path = [best_node.state]
        if self.debug:
            print(f"best_node: {best_node.state}")
            print(f"state: {best_node.state}, uncertainty: {best_node.get_uncertainty()}")
        while not np.array_equal(best_node.state, self.start.state):
            best_node = best_node.parent
            path.append(best_node.state)
            if self.debug:
                print(f"state: {best_node.state}, uncertainty: {best_node.get_uncertainty()}")
        
        self.num_vertices = len(self.vertices)
        return path

# Helper functions
    def AppendBelief(self, v, n_new):
        prev_belief = v.get_best_node()
        if v.dominates(n_new, prev_belief):
            v.add_belief_node(n_new)
            return True
        else:
            return False
 
    def edge_cost(self, from_vertex, to_vertex):
        """Calculate cost of an edge including distance and uncertainty"""
        # Euclidean distance cost
        cost = math.sqrt(np.linalg.norm(from_vertex - to_vertex))
        return cost 

    def get_measurement_covariance(self, state):
        """
        Determine measurement noise based on vertex location.

        Args:
            vertex: Vertex object with state and belief nodes.

        Returns:
            Measurement: Covariance matrix (2x2).
        """

        # Ensure the vertex is 2D
        d = len(state)
        if d != 2:
            raise ValueError("Vertex must be 2D for measurement noise calculation.")

        # Check if the vertex is inside a measurement zone
        for m in self.measurement_zone:
            left, right, bottom, top = m
            if left < state[0] < right and bottom < state[1] < top:
                return SMALL_VALUE * np.eye(d)  # Low measurement noise

        # Return high uncertainty if outside measurement zones
        return LARGE_VALUE * np.eye(d)

    def calculate_radius(self):
        n = len(self.vertices)

        # See ref [6] S. Karaman and E. Frazzoli, "Incremental Sampling-based \ 
        # Algorithms for Optimal Motion Planning" 2020
        radius = min((2 * (1 + 1/self.d) * (self.measure_free / self.zeta) * (math.log(n) / n))
                     ** (1 / self.d), self.eta)  # Radius based on dimensionality
        return radius

    def obstacle_free(self, vertex):
        """Check if probability of vertex being in obstacle is less than delta
        
        Args:
            vertex: Vertex object with state and belief nodes
            
        Returns:
            bool: True if probability of collision is less than delta
        """
        if vertex is None:
            return False
        
        for o in self.obstacle:
            # Compute probability of being inside obstacle using CDF
            p_collision = self.CDF(o, vertex)
            
            # Check if probability exceeds delta threshold
            if p_collision > self.delta:
                return False
                
        return True

    def add_edge(self, start_vertex, end_vertex, edge_data):
        """
        Add an edge to the graph

        Args:
            start_vertex: Starting vertex
            end_vertex: Ending vertex
            edge_data: Tuple of (X_nominal, U_nominal, K_gains)
        """
        if self.is_edge_valid(start_vertex.state, end_vertex.state):
            self.edges[(start_vertex, end_vertex)] = edge_data
        else:
            raise ValueError(f"Invalid edge: {start_vertex.state}, {end_vertex.state} violates constraints")
        if (start_vertex.state != edge_data[0]).all():
            raise ValueError(f"Error: start_vertex: {start_vertex.state} is not edge_data[0][0]: {edge_data[0]}")
        if (end_vertex.state != edge_data[-1]).all():
            raise ValueError(f"Error: end_vertex: {end_vertex.state} is not edge_data[0][-1]: {edge_data[-1]}")

    def get_edge(self, start_vertex, end_vertex):
        """
        Get edge data between two vertices

        Args:
            start_vertex: Starting vertex
            end_vertex: Ending vertex

        Returns:
            Tuple of (X_nominal, U_nominal, K_gains) or None if edge doesn't exist
        """
        return self.edges.get((start_vertex, end_vertex))

    def get_vertex_idx(self, state):
        """
        Find the vertex corresponding to the given state.

        Args:
            state: The state to search for

        Returns:
            Vertex: The vertex corresponding to the given state
        """
        for idx, vertex in enumerate(self.vertices):
            if np.array_equal(vertex.state, state):
                return idx
        return None

    def init_radius_vals(self):
        """Initialize radius values for each dimension"""
        self.d = len(self.start.state)
        
        # Calculate volume of unit ball in d dimensions
        match self.d:  # Source: https://en.wikipedia.org/wiki/Volume_of_an_n-ball
            case 2:
                self.zeta = 3.142
            case 3:
                self.zeta = 4.189
            case 4:
                self.zeta = 4.935
            case 5:
                self.zeta = 5.264
            case 6:
                self.zeta = 5.1677

        measure_tot = self.max_rand ** self.d  # Measure of workspace
        measure_obs = sum([(right-left) * (top-bottom) for left, right,
                          bottom, top in self.obstacle])  # Measure of obstacles
        self.measure_free = measure_tot - measure_obs  # Measure of free space

    def init_dynamics(self):
        self.dt = 0.1
        A = np.array([[0, 0], [0, 0]])
        B = 10*np.eye(2)
        C = np.eye(2)
        self.sys = ct.ss(A, B, C, 0)
        self.sys_dt = self.sys.sample(self.dt)
        self.A = self.sys_dt.A
        self.B = self.sys_dt.B
        self.C = self.sys_dt.C
        self.K, _, _ = ct.dlqr(self.sys_dt.A, self.sys_dt.B, np.eye(2), np.eye(2))
        
    @staticmethod
    def CDF(region, vertex):
        """Compute probability of vertex being inside a region"""
        left, right, bottom, top = region
        # Get best (lowest cost) belief node for this vertex
        belief = vertex.get_best_node()
        if belief is None:
            return 0.0
        
        # Total covariance is sum of information and covariance matrices
        total_cov = belief.sigma + belief._lambda
        mean = vertex.state
        
        # Create multivariate normal distribution
        mv_normal = multivariate_normal(mean=mean, cov=total_cov)
        
        # Compute probability of being inside region using CDF
        p_region = mv_normal.cdf(x=[right, top], lower_limit=[left, bottom])
        return p_region
    
    @staticmethod
    def convert_to_square(center, length, min_rand, max_rand):
        """Convert a rectangular region to a square region"""
        half_length = length / 2
        left = center[0] - half_length
        right = center[0] + half_length
        bottom = center[1] - half_length
        top = center[1] + half_length
        if top > max_rand or right > max_rand or bottom < min_rand or left < min_rand:
            raise ValueError("Invalid square dimensions: exceeds map bounds.")
        return left, right, bottom, top
    
# Plotting functions
    def plot_covariance(self, mean, cov):
        """
        Plot an ellipse representing the covariance matrix

        Args:
            mean: Mean (center) of the ellipse
            cov: Covariance matrix
        """
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        order = eigenvalues.argsort()[::-1]
        eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]

        # Compute the angle of the ellipse
        angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))

        # Compute width and height of the ellipse
        width, height = 2 * np.sqrt(eigenvalues)

        # Plot the ellipse
        ellipse = plt.matplotlib.patches.Ellipse(
            xy=mean, width=width, height=height, angle=angle, edgecolor='r', fc='None', lw=2)
        plt.gca().add_patch(ellipse)
