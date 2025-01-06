import numpy as np
from Roadmaps.BRM import BRM

class BRM_minmax(BRM):
    def __init__(self, start, goal, obstacle, workspace, measurement_zone, process_noise=0.1, num_samples=1000, animation=True, debug=True):
        super().__init__(start, goal, obstacle, workspace, measurement_zone, process_noise, num_samples, animation, debug)
        
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
                sigma_prime = self.transfer_func(belief, zeta_ij)
                
                neighbor_belief = beliefs[neighbor_idx]
                if neighbor_belief.sigma is None or max(np.trace(sigma_prime), np.trace(belief.sigma_max)) < np.trace(neighbor_belief.sigma_max):
                    # Add the neighbor to the queue
                    beliefs[neighbor_idx].sigma = max(sigma_prime, belief.sigma_max, key=np.trace)
                    beliefs[neighbor_idx].parent = belief
                    beliefs[neighbor_idx].max_covariance()
                    for i in range(len(queue)):
                        if (queue[i].state == neighbor).all():
                            queue[i] = beliefs[neighbor_idx]
                            break
                    else:
                        queue.append(beliefs[neighbor_idx])

        print("Goal not reachable.")
        return beliefs[-1]
