import math
import numpy as np
from RRT import RRT
from itertools import count
from scipy.stats import multivariate_normal

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
        if isinstance(other, Vertex):
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

        sigma_dominates = np.linalg.det(
            node1.sigma) <= np.linalg.det(node2.sigma)
        lambda_dominates = np.linalg.det(
            node1._lambda) <= np.linalg.det(node2._lambda)
        cost_dominates = node1.cost <= node2.cost
        sigma_dominates_i = np.linalg.det(
            node1.sigma) < np.linalg.det(node2.sigma)
        lambda_dominates_i = np.linalg.det(
            node1._lambda) < np.linalg.det(node2._lambda)
        cost_dominates_i = node1.cost < node2.cost

        f_le_g = sigma_dominates and lambda_dominates and cost_dominates
        f_i_le_g_i = sigma_dominates_i or lambda_dominates_i or cost_dominates_i

        if f_le_g and f_i_le_g_i:
            return True
        else:
            return False

class BeliefNode:
    def __init__(self, sigma=None, _lambda=None, cost=float('inf'), parent=None):
        """
        A node maintaining belief information for a vertex

        Args:
            sigma: Covariance matrix
            _lambda: Information matrix (defaults to small non-zero values for realistic uncertainty)
        """
        # Initialize covariance matrices if not provided
        self.sigma = sigma if sigma is not None else np.eye(2)
        self._lambda = _lambda if _lambda is not None else 0.1 * np.eye(2)
        self.cost = cost  # Cost-to-come
        self.parent = parent  # Parent belief node

    def __eq__(self, other):
        if isinstance(other, BeliefNode):
            return (np.array_equal(self.sigma, other.sigma) and
                    np.array_equal(self._lambda, other._lambda))
        return False

class RRBT(RRT):
    def __init__(self, start, goal, obstacle, workspace, animation=True, eta=2.5,
                 goal_sample_rate=0.005, delta=0.159, process_noise=0.1, measurement_noise=0.1):
        super().__init__(start, goal, obstacle, workspace, animation, eta, goal_sample_rate)
        # Process noise covariance
        self.Q = process_noise * np.eye(len(start[0]))
        # Measurement noise covariance
        self.R = measurement_noise * np.eye(len(start[0]))
        self.delta = delta

        # Initialize vertices with their belief nodes
        self.start = Vertex(start[0])  # Just state
        self.goal = Vertex(goal[0])    # Just state

        # Create initial belief nodes
        start_belief = BeliefNode(start[1], cost=0)  # Initial belief
        self.start.add_belief_node(start_belief)

        goal_belief = BeliefNode(goal[1])    # Goal belief
        self.goal.add_belief_node(goal_belief)

        # Change edges from list to dictionary
        # Key: tuple(start_vertex, end_vertex)
        # Value: tuple(X_nominal, U_nominal, K_gains)
        self.edges = {}
        self.vertices = [self.start]

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

    def planning(self):
        """Main RRBT algorithm"""
        for self.iterations in count():
            # Check if we've reached the goal region
            if self.goal_reached():
                break
            x_rand = self.sample_random_vertex()  # Sample random belief state
            v_nearest = self.get_nearest_vertex(
                x_rand, self.vertices)  # Find nearest vertex in tree
            # Generate new vertex with uncertainty
            edge_new = self.Connect(v_nearest, v_rand)

            # Propagate belief along edge
            belief = self.Propagate(edge_new, v_nearest.nodes)

            # If belief is None, constraints were violated
            if len(v_nearest.nodes) > 0 and belief:
                queue = []
                # Add new vertex and edge to tree
                v_rand = Vertex(x_rand)
                v_rand.add_belief_node(belief)
                self.vertices.append(v_rand)
                self.add_edge(v_nearest, v_rand, edge_new)

                # Add reverse edge for bidirectional tree
                reverse_edge = self.Connect(v_rand, v_nearest)
                self.add_edge(v_rand, v_nearest, reverse_edge)

                # Although initially counter-intuitive, we would like to rewire the tree from the nearest vertex
                queue.append(v_nearest)

                # Find the vertices within some ball of our newly sampled vertex, v_rand
                nearby_vertices = self.Near(v_rand)

                # Connect v_rand with nearby vertices
                for v_near in nearby_vertices:
                    edge = self.Connect(v_near, v_rand)
                    self.add_edge(v_near, v_rand, edge)
                    reverse_edge = self.Connect(v_rand, v_near)
                    self.add_edge(v_rand, v_near, reverse_edge)
                    queue.append(v_near)

                # Rewire nearby vertices
                self.rewire(queue, v_rand)

                if self.iterations % 3 == 0 and self.animation:
                    self.update_graph(x_rand)

        return self.final_path(len(self.vertices) - 1)

    def AppendBelief(self, v, n_new):
        prev_belief = v.get_best_node()
        if v.dominates(prev_belief, n_new):
            return False
        else:
            v.add_belief_node(n_new)
            return True

    def Connect(self, from_vertex, to_vertex):
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
        X_nominal = []  # X̌^(a,b) = (x̃₀, x̃₁, ..., x̃ₜ)
        U_nominal = []  # Ǔ^(a,b) = (ũ₀, ũ₁, ..., ũₜ)
        K_gains = []    # Ǩ^(a,b) = (K₀, K₁, ..., Kₜ)

        # Initial conditions (eq. 6)
        current_state = from_vertex.state
        X_nominal.append(current_state)

        # Generate trajectory using system dynamics (eq. 7)
        T = int(np.linalg.norm(to_vertex.state - from_vertex.state) /
                self.eta)  # Number of steps

        for _ in range(T):
            # Generate nominal control input
            direction = to_vertex.state - current_state
            u_t = self.eta * direction / np.linalg.norm(direction)
            U_nominal.append(u_t)

            # Propagate state using nominal dynamics f(x̃ₜ₋₁, ũₜ₋₁, 0)
            next_state = current_state + u_t  # Simple integrator model
            X_nominal.append(next_state)
            current_state = next_state

            # Compute Kalman gain for feedback control (eq. 8-9)
            # Using steady-state Kalman gain for simplicity
            P = self.Q  # State estimation covariance
            K_t = P @ np.linalg.inv(P + self.R)  # Kalman gain
            K_gains.append(K_t)

            # Check if we've reached the target
            if np.linalg.norm(current_state - to_vertex.state) < self.eta:
                break

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
        if n_start is not edge[0][0]:
            return False
        parent = n_start.get_best_node()
        for t in range(len(edge)):
            x_nominal = edge[0][t]
            # u_nominal = edge[1][t]
            K_t = edge[2][t]
            A = B = np.eye(2)
            C = [1, 1].T

            # Covariance prediction (eq. 21, 33)
            sigma_bar = A@n_start.get_best_node().sigma@A.T + self.Q  # (eq. 17)
            L_t = sigma_bar @ C.T + self.R  # (eq. 19)
            sigma = sigma_bar - L_t @ C @ sigma_bar  # (eq. 21)
            _lambda = (A-B@K_t) @ n_start.get_best_node()._lambda @ (A -
                                                                     B@K_t).T + B @ self.R @ B.T  # (eq. 33)

            node = BeliefNode(sigma, _lambda, parent=parent)
            # Cost expectation evaluation (eq. 11)
            node.cost = parent.cost + self.edge_cost(parent, node)
            vertex = Vertex(x_nominal)
            vertex.add_belief_node(node)

            # Chance constraint check (eq.13)
            if not self.obstacle_free(vertex):
                return False
            parent = node

        if node is not edge[-1][0]:
            return False
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

        measure_tot = math.prod(self.workspace)  # Measure of workspace
        measure_obs = sum([(right-left) * (top-bottom) for left, right,
                          bottom, top in self.obstacle])  # Measure of obstacles
        measure_free = measure_tot - measure_obs  # Measure of free space

        # See ref [6] S. Karaman and E. Frazzoli, "Incremental Sampling-based \ 
        # Algorithms for Optimal Motion Planning" 2020
        radius = min((2 * (1 + 1/d) * (measure_free / zeta) * (math.log(n) / n))
                     ** (1 / d), self.eta)  # Radius based on dimensionality

        for v in self.vertices:
            if np.linalg.norm(v.state - vertex.state) < radius:
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
                node_new = self.Propagate(e_neighbour, node)  # returns n_end
                # returns True if node_new has lower cost than all nodes in v_neighbour
                if self.AppendBelief(v_neighbour, node_new):
                    v_neighbour.add_belief_node(node_new)
                    queue.append(v_neighbour)

    def edge_cost(self, from_vertex, to_vertex):
        """Calculate cost of an edge including distance and uncertainty"""
        # Euclidean distance cost
        distance_cost = math.sqrt(self.L2_norm(from_vertex, to_vertex))

        # Uncertainty cost (e.g., trace or determinant of covariance)
        uncertainty_cost = np.trace(to_vertex.covariance)

        return distance_cost + self.uncertainty_weight * uncertainty_cost

    def goal_reached(self):
        """Check if goal has been reached with acceptable uncertainty"""
        left, right, bottom, top = self.goal
        for vertex in self.vertices:
            # Compute probability of being inside goal using CDF
            p_goal = self.CDF(self.goal, vertex)
            
            # Check if probability is less than delta threshold
            if p_goal < self.delta:
                return True
        return False

    def CDF(self, region, vertex):
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