import math
import numpy as np
import matplotlib.pyplot as plt
import utils as utils
from scipy.stats import multivariate_normal
from sklearn.neighbors import KDTree
from Roadmaps.PRM import PRM

LARGE_VALUE = 1e6  # Large number to represent high uncertainty

class BRM(PRM):
    class BeliefNode:
        def __init__(self, state, sigma=None, parent=None, sigma_max=None):
            """
            A node maintaining belief information for a state

            Args:
                state: The state this belief node represents
                P_hat: Covariance matrix
                P_tilde: Information matrix (defaults to small non-zero values for realistic uncertainty)
            """
            self.state = state  # Store the state
            self.sigma = sigma # Covariance matrix
            self.parent = parent  # Parent belief node
            if sigma_max is None:
                self.max_covariance()  # Maximum covariance in the path
            else:
                self.sigma_max = sigma_max
            
        def visited(self, state):
            """Check if the state has been visited"""
            if self.parent is None:
                return False
            if np.array_equal(self.state, state):
                return True
            return self.parent.visited(state)
            
        def max_covariance(self):
            """Compute the maximum covariance in the path"""
            if self.parent is None:
                self.sigma_max = LARGE_VALUE * np.eye(2)
            else:
                self.sigma_max = max(self.sigma, self.parent.sigma_max, key=np.trace)

        def __eq__(self, other):
            if isinstance(other, self.__class__):
                return (np.array_equal(self.P_hat, other.P_hat) and
                        np.array_equal(self.P_tilde, other.P_tilde))
            return False
            if isinstance(other, self.__class__):
                return (np.array_equal(self.P_hat, other.P_hat) and
                        np.array_equal(self.P_tilde, other.P_tilde))
            return False
    
    def __init__(self, start, goal, obstacle, workspace, measurement_zone, process_noise=0.001, animation=True):
        super().__init__(start[0], goal, obstacle, workspace, animation)
        self.W_t = process_noise * np.eye(2)
        self.measurement_zone = [measurement_zone]
        self.start_belief = self.BeliefNode(self.start, sigma=start[1])
        self.start_belief.sigma_max = start[1]
          
    def planning(self, rng=None):
        """
        Run probabilistic road map planning

        :param start: start position
        :param goal: goal position
        :param obstacle: obstacle positions
        :param rng: (Optional) Random generator
        :return:
        """
        samples, beliefs = self.sample_points(rng)
        if self.animation:
            self.plot_samples(samples)

        road_map = self.generate_road_map(samples)
        if self.animation:
            self.plot_edges(road_map, samples)
            
        r = self.bfs(road_map, samples, beliefs)
                
        assert r, 'Cannot find path'
        
        if self.animation:
            self.plot_final_path(r)

        return r
     
# Sample process
    def sample_points(self, rng=None):
        """
        Generate sample points and corresponding belief nodes.

        Args:
            rng: Random number generator.

        Returns:
            samples: List of coordinates (for KDTree).
            belief_nodes: List of BeliefNode objects (for metadata).
        """
        samples = super().sample_points(rng)  # Call PRM's sample_points method
        
        # Create belief nodes for each sample
        belief_nodes = [
            self.BeliefNode(state=np.array(sample))
            for sample in samples
        ]
        belief_nodes[len(belief_nodes) - 2] = self.start_belief
        
        return samples, belief_nodes

# Roadmap generation
    def generate_road_map(self, samples):
        """
        Road map generation

        sample: [m] positions of sampled points
        """
        print("Generating road map")
        road_map = []
        n_sample = len(samples)
        sample_kd_tree = KDTree(samples)

        for (v_idx, vertex) in zip(range(n_sample), samples):
            indices, dist = sample_kd_tree.query_radius(
                np.array(vertex).reshape(1, -1), r=self.radius, return_distance=True
            )
            sorted_indices = np.argsort(dist)  # Sort indices based on distances
            indices = (indices[sorted_indices])[0]  # Sort indices accordingly
            
            
            edge_id = []
            zetas = []
            for neighbor_idx in indices:
                neighbour = samples[neighbor_idx] # Neighbour or neighbor? Canada or US?
                if neighbour == vertex:
                    continue
                if self.is_edge_valid(vertex, neighbour):
                    edge_id.append(neighbor_idx)
                    trajectory = self.Connect(vertex, neighbour)
                    zetas.append(self.zeta(trajectory))
            road_map.append((edge_id, zetas))
        print("Road map generated")
        return road_map
        
    def Connect(self, from_state, to_state):
        """
        Connect two vertices with a nominal trajectory, control inputs, and stabilizing feedback gains
        Args:
            from_vertex: Starting vertex (x^a)
            to_vertex: Target vertex (x^b)
        Returns:
            List: X̌^(a,b) containing Nominal state trajectory
        """
        path_resolution = 0.1
        X_nominal = []
        to_state = np.array(to_state)
        from_state = np.array(from_state)

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

            # Propagate state using nominal dynamics f(x̃ₜ₋₁, ũₜ₋₁, 0)
            next_state = current_state + u_t  # Simple integrator model
            if not self.is_vertex_valid(next_state):
                return False
            X_nominal.append(next_state)
            current_state = next_state

        return X_nominal

    def zeta(self, trajectory):
        """
        Compute the aggregate scattering matrix for a trajectory segment.

        Args:
            trajectory: List of states along the trajectory.
            motion_model: Function for the motion model.
            measurement_model: Function for the measurement model.
            Q_t: Process noise covariance matrix (assumed constant for simplicity).
            R_t: Measurement noise covariance matrix (assumed constant for simplicity).

        Returns:
            S_ij: The aggregate scattering matrix for the trajectory.
        """
        n = len(trajectory[0])  # State dimension
        S_ij = np.zeros((2 * n, 2 * n))  # Initialize the aggregate scattering matrix
        for t in range(len(trajectory) - 1):
            # Get current state and next state
            x_t = trajectory[t]

            # Compute Jacobians
            H_t = G_t = V_t = np.eye(2)  # Jacobian of the measurement and motion model
            Q_t = self.get_measurement_covariance(x_t)   # Measurement noise covariance
            R_t = V_t @ self.W_t @ np.linalg.inv(V_t)  # Process noise covariance

            # Compute M_t = H_t.T @ Q_t @ H_t
            M_t = H_t.T @ np.linalg.inv(Q_t) @ H_t

            S_t = np.block([
                [G_t, R_t],
                [-M_t, G_t.T]
            ])
            if t > 0:
                # Combine with the current aggregate matrix using the Redheffer star product
                S_ij = self.redheffer_star_product(S_ij, S_t)
            else: 
                S_ij = S_t

        return S_ij      

# Search Process
    def bfs_minmax(self, road_map, samples, beliefs):
        """
        Perform BFS to explore the roadmap.

        Args:
            road_map: List of roadmap edges and their associated zeta values.
            samples: List of sampled nodes.
            sigma_0: Initial covariance matrix for the belief state.

        Returns:
            goal_node: The node at the goal if reachable, else None.
        """
        # Map each node to its index for efficient querying
        node_index = {tuple(node): idx for idx, node in enumerate(samples)}

        # Initialize the queue with the start node and its initial covariance        
        queue = [self.start_belief]
        n = len(samples[0])

        while queue:
            belief = queue.pop(0)

            # Check if we've reached the goal
            if np.array_equal(belief.state, self.goal):
                print("Goal reached!")
                continue

            # Retrieve the current node's index
            current_idx = node_index[tuple(belief.state)]

            # Explore neighbors
            neighbors, zetas = road_map[current_idx]
            for neighbor_idx, zeta_ij in zip(neighbors, zetas):
                neighbor = samples[neighbor_idx]
                if belief.visited(neighbor):
                    continue
                
                # Compute the updated covariance using the transfer function
                zeta_nn_prime = self.transfer_func(belief, zeta_ij)
                Psi = np.block([
                    [belief.sigma],
                    [np.eye(len(belief.sigma))]
                ])
                Psi_prime = zeta_nn_prime @ Psi
                Psi_11 = Psi_prime[:n , :]
                Psi_21 = Psi_prime[n:, :]
                
                if(np.linalg.det(Psi_21) == 0):
                    Psi_21 = 0.01 * np.eye(2)
                sigma_prime = Psi_11 @ np.linalg.inv(Psi_21)
                
                if max(np.trace(sigma_prime), np.trace(belief.sigma_max)) < np.trace(beliefs[neighbor_idx].sigma_max):
                    # Add the neighbor to the queue
                    beliefs[neighbor_idx].sigma = max(sigma_prime, belief.sigma_max, key=np.trace)
                    beliefs[neighbor_idx].parent = belief
                    beliefs[neighbor_idx].max_covariance()
                    for i in range(len(queue)):
                        if queue[i].state == neighbor:
                            queue[i] = beliefs[neighbor_idx]
                            break
                    else:
                        queue.append(beliefs[neighbor_idx])

        print("Goal not reachable.")
        return None

    def bfs(self, road_map, samples, beliefs):
        """
        Perform BFS to explore the roadmap.

        Args:
            road_map: List of roadmap edges and their associated zeta values.
            samples: List of sampled nodes.
            sigma_0: Initial covariance matrix for the belief state.

        Returns:
            goal_node: The node at the goal if reachable, else None.
        """
        # Map each node to its index for efficient querying
        node_index = {tuple(node): idx for idx, node in enumerate(samples)}

        # Initialize the queue with the start node and its initial covariance
        queue = [self.start_belief]
        n = len(samples[0])

        while queue:
            belief = queue.pop(0)

            # Check if we've reached the goal
            if np.array_equal(belief.state, self.goal):
                print("Goal reached!")
                return belief

            # Retrieve the current node's indexbelief.state
            current_idx = node_index[tuple(belief.state)]

            print(f"Current state: {belief.state}, uncertainty: {np.trace(belief.sigma)}")
            # Explore neighbors
            neighbors, zetas = road_map[current_idx]
            for neighbor_idx, zeta_ij in zip(neighbors, zetas):
                print(f"zetaj:")
                print(f"{zeta_ij} to {samples[neighbor_idx]}")
                neighbor = samples[neighbor_idx]
                
                if belief.visited(neighbor):
                    continue

                # Compute the updated covariance using the transfer function
                sigma_prime = self.transfer_func(belief, zeta_ij)
                
                if beliefs[neighbor_idx].sigma is None or np.trace(sigma_prime) < np.trace(beliefs[neighbor_idx].sigma):
                    # Add the neighbor to the queue
                    beliefs[neighbor_idx].sigma = sigma_prime  
                    beliefs[neighbor_idx].parent = belief
                    for i in range(len(queue)):
                        if (queue[i].state == neighbor).all():
                            queue[i] = beliefs[neighbor_idx]
                            break
                    else:
                        queue.append(beliefs[neighbor_idx])

        print("Goal not reachable.")
        return None

    def transfer_func(self, from_vertex, S_1toT):
        """
        Compute the one-step transfer function between two vertices using scattering matrices.

        Args:
            from_vertex: The starting vertex (numpy array).
            to_vertex: The ending vertex (numpy array).
            trajectory: The trajectory segment from from_vertex to to_vertex.

        Returns:
            mean: Updated mean (numpy array).
            covariance: Updated covariance matrix (numpy array).
        """
        # Motion and measurement models
        # Compute the aggregate scattering matrix

        # Get the dimension of the state space
        dim = len(from_vertex.state)

        # Apply the scattering matrix to the prior belief state
        init_block = np.block([
            [np.eye(dim), from_vertex.sigma],
            [np.zeros(shape=(dim,dim)), np.eye(dim)]
        ])
        output_block = self.redheffer_star_product(init_block, S_1toT) # eq (74)
        # Extract the top right block as the updated covariance matrix
        return output_block[:dim, dim:]
        
# Helper functions
    @staticmethod
    def redheffer_star_product(S1, S0):
        """
        Compute the Redheffer star product of two scattering matrices.

        Args:
            S1: First scattering matrix (2n x 2n).
            S0: Second scattering matrix (2n x 2n).

        Returns:
            S_star: The resulting scattering matrix.
        """
        n = S1.shape[0] // 2  # Dimension of the state space
        
        split = lambda array, n: array.reshape(array.shape[1] // n, n, -1, n).swapaxes(1, 2).reshape(-1, n, n)  

        # Partition the matrices into sub-blocks
        A, B, C, D = split(S1, n)
        W, X, Y, Z = split(S0, n)
        
        I = np.eye(n)

        # Compute the star product
        A_star = W @ np.linalg.inv(I - B @ Y) @ A
        B_star = X + W @ np.linalg.inv(I - B @ Y) @ B @ Z
        C_star = C + D @ np.linalg.inv(I - Y @ B) @ Y @ A
        D_star = D @ np.linalg.inv(I - Y @ B) @ Z

        # Combine into a single matrix
        S_star = np.block([
            [A_star, B_star],
            [C_star, D_star]
        ])
        return S_star
     
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
                return 0.001 * np.eye(d)  # Low measurement noise

        # Return high uncertainty if outside measurement zones
        return LARGE_VALUE * np.eye(d)
 
    @staticmethod
    def edge_cost(from_vertex, to_vertex):
        """Calculate cost of an edge including distance and uncertainty"""
        # Euclidean distance cost
        cost = math.sqrt(np.linalg.norm(from_vertex - to_vertex))
        return cost 
  
    def obstacle_free(self, vertex):
        """Check if probability of vertex being in obstacle is less than delta
        
        Args:
            vertex: Vertex object with state and belief nodes
            
        Returns:
            bool: True if probability of collision is less than delta
        """
        for o in self.obstacle:
            # Compute probability of being inside obstacle using CDF
            p_collision = self.CDF(o, vertex)
            
            # Check if probability exceeds delta threshold
            if p_collision > self.delta:
                return False
                
        return True

# Plotting functions
    def plot_samples(self, sample):
        if self.animation:
            print("Plotting samples and obstacles")
            for o in self.obstacle:
                # Plot the blue rectangle obstacle
                utils.plot_rectangle(o, color="-b")
            for m in self.measurement_zone:
                # Plot the red measurement zone
                utils.plot_rectangle(m, color="-g")
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

    def plot_edges(self, road_map, sample):
        for v_idx, (neighbors, _) in enumerate(road_map):
            for neighbor_idx in neighbors:
                # get the start and end points of the edge
                start_point = sample[v_idx]
                end_point = sample[neighbor_idx]

                # extract x and y coordinates for plotting
                path_x = [start_point[0], end_point[0]]
                path_y = [start_point[1], end_point[1]]

                # plot the edge as a yellow line
                plt.plot(path_x, path_y, "-y")
        plt.pause(0.001)

    def plot_final_path(self, goal_belief):
        # Extract x and y coordinates from the path
        path_x = []
        path_y = []
        parent = goal_belief
        while parent is not None:
            path_x.append(parent.state[0])
            path_y.append(parent.state[1])
            print(f"state: {parent.state}, uncertainty: {np.trace(parent.sigma)}")
            self.plot_covariance(parent.state, parent.sigma)
            parent = parent.parent
        
        # Plot the final path as a red line
        plt.plot(path_x, path_y, "-r", linewidth=2, label="Final Path")
        plt.legend()
        plt.pause(0.001)
        plt.show()
