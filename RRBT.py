import math
from types import NoneType
import numpy as np
from traitlets import Instance
from RRT import RRT
from itertools import count
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt


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
            if isinstance(other, self.Vertex):
                return np.array_equal(self.state, other.state)
            return False

        def __hash__(self):
            return hash(tuple(self.state))

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
            self.state = state  # Store the state
            self.sigma = sigma if sigma is not None else np.eye(2)
            self._lambda = _lambda if _lambda is not None else 0.1 * np.eye(2)
            self.cost = cost  # Cost-to-come
            self.parent = parent  # Parent belief node

        def __eq__(self, other):
            if isinstance(other, self.__class__):
                return (np.array_equal(self.sigma, other.sigma) and
                        np.array_equal(self._lambda, other._lambda))
            return False

    def __init__(self, start, goal_region, obstacle, workspace, measurement_zone, animation=True, eta=2.5,
                 goal_sample_rate=0.005, delta=0.159, process_noise=0.01):
        
        super().__init__(start=start[0], goal=goal_region[0], obstacle=obstacle, workspace=workspace, animation=animation, eta=eta, goal_sample_rate=goal_sample_rate)

        # Create initial belief nodes
        start_belief = self.BeliefNode(self.start.state, start[1], cost=0)  # Initial belief
        self.start.add_belief_node(start_belief)
        
        # Create goal region
        self.goal_region = self.convert_to_square(goal_region[0], goal_region[1], self.min_rand, self.max_rand)
        
        # Key: tuple(start_vertex, end_vertex)
        # Value: tuple(X_nominal, U_nominal, K_gains)
        self.edges = {}
        
        self.vertices = [self.start]
        
        # Process noise covariance
        self.Q = process_noise * np.eye(self.start.dimension())
        self.measurement_zone = [measurement_zone]
        self.delta = delta

    def planning(self):
        """Main RRBT algorithm"""
        goal_vertex = False
        for self.iterations in count():
            # Check if we've reached the goal region
            goal_vertex = self.goal_reached()
            if goal_vertex is not False:
                print("Goal reached with probability:", self.CDF(self.goal_region, goal_vertex))
                break
            
            x_rand = self.sample_random_vertex()  # Sample random belief state
            v_nearest = self.get_nearest_vertex(x_rand, self.vertices)  # Find nearest vertex in tree
            edge_new = self.Connect(v_nearest.state, x_rand)
            if edge_new is False:
                continue

            # Propagate belief along edge
            for node in v_nearest.nodes:
                belief = self.Propagate(edge_new, node)

            # If belief is None, constraints were violated
            if len(v_nearest.nodes) > 0 and belief is not False:
                queue = []
                # Add new vertex and edge to tree
                v_rand =self.Vertex(x_rand)
                v_rand.add_belief_node(belief)
                self.vertices.append(v_rand)
                self.add_edge(v_nearest, v_rand, edge_new)

                # Add reverse edge for bidirectional tree
                reverse_edge = self.Connect(x_rand, v_nearest.state)
                self.add_edge(v_rand, v_nearest, reverse_edge)

                # Although initially counter-intuitive, we would like to rewire the tree from the nearest vertex
                queue.append(v_nearest)

                # Find the vertices within some ball of our newly sampled vertex, v_rand
                nearby_vertices = self.Near(v_rand)

                # Connect v_rand with nearby vertices
                for v_near in nearby_vertices:
                    edge = self.Connect(v_near.state, x_rand)
                    self.add_edge(v_near, v_rand, edge)
                    reverse_edge = self.Connect(x_rand, v_near.state)
                    self.add_edge(v_rand, v_near, reverse_edge)
                    queue.append(v_near)

                # Rewire nearby vertices
                self.rewire(queue, v_rand)

                if self.animation is True and self.iterations % 5 == 0:
                    self.update_graph(x_rand)

        return self.final_path(goal_vertex)
    
    def update_graph(self, sampled_vec=None):
        plt.clf()
        # Plot the sampled vector as a black plus sign
        if sampled_vec is not None:
            plt.plot(sampled_vec[0], sampled_vec[1], "Pk")

        # Plot edges as yellow lines
        for edge in self.edges.values():
            path_x = [state[0] for state in edge[0]]
            path_y = [state[1] for state in edge[0]]
            plt.plot(path_x, path_y, "-y")

        # Plot the covariance of each vertex's best belief node
        for vertex in self.vertices:
            best_node = vertex.get_best_node()
            if best_node is not None:
                self.plot_covariance(vertex.state, best_node.sigma + best_node._lambda)

        for o in self.obstacle:
            # Plot the blue rectangle obstacle
            self.plot_rectangle(o, color="-b")
            
        for m in self.measurement_zone:
            # Plot the red measurement zone
            self.plot_rectangle(m, color="-g")

        # Plot the green start "S" and red goal "G"
        self.plot_rectangle(self.goal_region, color="-b")
        plt.plot(self.start.state[0], self.start.state[1],
                 c="g", marker=r"$\mathbb{S}$")
        plt.plot(self.goal.state[0], self.goal.state[1],
                 c="r", marker=r"$\mathbb{G}$")

        plt.axis("equal")
        plt.axis([self.min_rand, self.max_rand, self.min_rand, self.max_rand])
        plt.grid(True)
        plt.pause(0.01)

    def final_path(self, goal_vertex):
        path = []
        vertex = goal_vertex
        best_node = vertex.get_best_node()
        print(f"best_node: {best_node.state}")
        
        while not np.array_equal(best_node.parent.state, self.start.state):
            path.append(best_node.state)
            best_node = best_node.parent
                
        self.num_vertices = len(self.vertices)
        return path

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

    def add_edge(self, start_vertex, end_vertex, edge_data):
        """
        Add an edge to the graph

        Args:
            start_vertex: Starting vertex
            end_vertex: Ending vertex
            edge_data: Tuple of (X_nominal, U_nominal, K_gains)
        """
        self.edges[(start_vertex, end_vertex)] = edge_data

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

    def get_vertex(self, state):
        """
        Find the vertex corresponding to the given state.

        Args:
            state: The state to search for

        Returns:
            Vertex: The vertex corresponding to the given state
        """
        for vertex in self.vertices:
            if np.array_equal(vertex.state, state):
                return vertex
            else:
                print(f"vertex.state: {vertex.state} is not equal to state: {state}")
        return None

    def get_measurement_covariance(self, state):
        """
        Determine measurement noise based on vertex location.

        Args:
            vertex: Vertex object with state and belief nodes.

        Returns:
            Measurement: Covariance matrix (2x2).
        """
        LARGE_VALUE = 1e6  # Large number to represent high uncertainty


        # Ensure the vertex is 2D
        d = len(state)
        if d != 2:
            raise ValueError("Vertex must be 2D for measurement noise calculation.")

        # Check if the vertex is inside a measurement zone
        for m in self.measurement_zone:
            left, right, bottom, top = m
            if left < state[0] < right and bottom < state[1] < top:
                return 0.01 * np.eye(d)  # Low measurement noise

        # Return high uncertainty if outside measurement zones
        return LARGE_VALUE * np.eye(d)

    def AppendBelief(self, v, n_new):
        prev_belief = v.get_best_node()
        if v.dominates(prev_belief, n_new):
            return False
        else:
            v.add_belief_node(n_new)
            return True

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
        path_resolution = 0.1
        # Initialize trajectory lists
        X_nominal = []  # X̌^(a,b) = (x̃₀, x̃₁, ..., x̃ₜ)
        U_nominal = []  # Ǔ^(a,b) = (ũ₀, ũ₁, ..., ũₜ)
        K_gains = []    # Ǩ^(a,b) = (K₀, K₁, ..., Kₜ)

        if (to_state == from_state).all():
            print(f"from_state: {from_state} and to_state: {to_state}")
            return False
        # Initial conditions (eq. 6)
        current_state = from_state
        X_nominal.append(current_state)
        
        # Generate trajectory using system dynamics (eq. 7)
        d = np.linalg.norm(to_state - from_state)
        
        n_steps = int(d / path_resolution)  # Number of steps
        
        for i in range(n_steps+1):
            # Generate nominal control input
            direction = to_state - current_state
            if i == n_steps:
                path_resolution = np.linalg.norm(direction)
                if not (0 < path_resolution < 0.1):
                    raise ValueError("Connect error: path resolution out of bounds.")
            u_t = path_resolution * direction / np.linalg.norm(direction)
            U_nominal.append(u_t)

            # Propagate state using nominal dynamics f(x̃ₜ₋₁, ũₜ₋₁, 0)
            next_state = current_state + u_t  # Simple integrator model
            if not self.is_vertex_valid(next_state):
                return False
            X_nominal.append(next_state)
            current_state = next_state

            # Compute Kalman gain for feedback control (eq. 8-9)
            # Using steady-state Kalman gain for simplicity
            R = self.get_measurement_covariance(current_state)
            P = self.Q  # State estimation covariance
            K_t = P @ np.linalg.inv(P + R)  # Kalman gain
            K_gains.append(K_t)
        

        return X_nominal, U_nominal, K_gains

    def Propagate(self, edge, n_start):
        """
        Propagate belief along an edge by computing covariance matrices and checking chance constraints.

        Args:
            edge: Tuple (X̌^(a,b), Ǔ^(a,b), Ǩ^(a,b)) containing nominal trajectory, controls, and gains
            n_start: Starting vertex with belief nodes

        Returns:
            BeliefNode: Final belief node if propagation succeeds, False if constraints violated
        """
        if not np.array_equal(n_start.state, edge[0][0]):
            print(f"n_start: {n_start.state} is not edge[0][0]: {edge[0][0]}")
            raise ValueError("Propagate error: n_start is not edge[0][0]")
        parent = n_start
        controller_len = len(edge[2])
        for t in range(controller_len):
            x_t = edge[0][t]
            x_tp1 = edge[0][t+1]
            K_t = edge[2][t]
            A = B = C = np.eye(2)
            R = self.get_measurement_covariance(x_t)

            # Covariance prediction (eq. 21, 33)
            sigma_bar = A @ parent.sigma @ A.T + self.Q  # (eq. 17)
            S_t = C @ sigma_bar @ C.T + R  # (eq. 18)
            L_t = sigma_bar @ C.T @ np.linalg.inv(S_t)  # (eq. 19)
            sigma = sigma_bar - (L_t @ C) @ sigma_bar  # (eq. 21)
            _lambda = (A - B @ K_t) @ parent._lambda @ (A - B @ K_t).T + L_t @ C @ sigma_bar  # (eq. 33)

            vertex = self.Vertex(x_tp1)
            node = self.BeliefNode(x_tp1, sigma, _lambda, parent=parent)
            node.cost = parent.cost + self.edge_cost(parent.state, x_tp1) # Cost (eq. 11)
            vertex.add_belief_node(node)
            # self.vertices.append(vertex)

            # Chance constraint check (eq.13)
            if not self.obstacle_free(vertex):
                return False
            parent = node

        if node.state is not edge[0][-1]:
            print(f"Propagate error: node, {node.state} is not edge[0][-1]: {edge[0][-1]}")
            raise ValueError("Propagate error: node is not edge[0][-1]")
        return node

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

    def Near(self, vertex):
        """Find nearby vertices within a certain radius"""
        nearby_vertices = []
        d = vertex.dimension()
        n = len(self.vertices)

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
        radius = min((2 * (1 + 1/d) * (measure_free / zeta) * (math.log(n) / n))
                     ** (1 / d), self.eta)  # Radius based on dimensionality

        for v in self.vertices:
            if np.linalg.norm(v.state - vertex.state) < radius and v.state is not vertex.state:
                nearby_vertices.append(v)
        return nearby_vertices

    def rewire(self, queue, v_rand):
        """Rewire nearby vertices through new vertex if it provides lower cost"""
        while len(queue) != 0:
            v_neighbour = queue.pop(0)
            e_neighbour = self.get_edge(v_rand, v_neighbour)
            if e_neighbour is None:
                continue


            for node in v_neighbour.nodes:
                node_new = self.Propagate(e_neighbour, v_rand.get_best_node() )  # returns n_end
                # returns True if node_new has lower cost than all nodes in v_neighbour
                if node_new is not False and self.AppendBelief(v_neighbour, node_new):
                    v_neighbour.add_belief_node(node_new)
                    queue.append(v_neighbour)

    def edge_cost(self, from_vertex, to_vertex):
        """Calculate cost of an edge including distance and uncertainty"""
        # Euclidean distance cost
        cost = math.sqrt(np.linalg.norm(from_vertex - to_vertex))
        return cost 

    def goal_reached(self):
        """Check if goal has been reached with acceptable uncertainty"""
        for vertex in self.vertices:
            # Compute probability of being inside goal using CDF
            p_goal = self.CDF(self.goal_region, vertex)
            
            # Check if probability is less than delta threshold
            if (1 - p_goal) < self.delta:
                print("Goal reached with probability:", p_goal)
                return vertex
        return False

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