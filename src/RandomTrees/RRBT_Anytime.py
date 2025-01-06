from itertools import count

import numpy as np
from RandomTrees.RRBT import RRBT
class RRBT_Anytime(RRBT):
    def __init__(self, start, goal_region, obstacle, workspace, measurement_zone, animation=True, eta=10, goal_sample_rate=0.01, delta=0.159, process_noise=0.01, iters=1000):
        super().__init__(start, goal_region, obstacle, workspace, measurement_zone, animation, eta, goal_sample_rate, delta, process_noise, iters)
    
    def planning(self):
        """Main RRBT algorithm"""
        goal_vertex = False
        for self.iterations in count():
            if self.iterations >= self.max_iters:
                break
            res = self.goal_reached()
            if res is not False:
                goal_vertex, p_goal = res
                if p_goal > self.best_prob:
                    self.best_path = self.final_path(goal_vertex)
                    self.best_prob = p_goal
            # Check if we've reached the goal region
            x_rand = self.sample_random_vertex()  # Sample random belief state
            v_nearest = self.get_nearest_vertex(x_rand, self.vertices)  # Find nearest vertex in tree
            edge_new = self.Connect(v_nearest.state, x_rand)
            if edge_new is False:
                # Edge is invalid (e.g., same vertex or obstacle in the way)
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
                    if edge is False:
                        continue
                    self.add_edge(v_near, v_rand, edge)
                    reverse_edge = self.Connect(x_rand, v_near.state)
                    self.add_edge(v_rand, v_near, reverse_edge)
                    queue.append(v_near)

                # Rewire nearby vertices
                self.rewire(queue)

                if self.animation is True and self.iterations % 5 == 0:
                    self.update_graph(x_rand, goal_vertex)
                
        if goal_vertex is not False:
            goal_uncertainty = np.trace(goal_vertex.get_best_cov())
            print("Goal reached with probability:", self.final_prob)
            print("Goal uncertainty:", goal_uncertainty)
            return self.best_path, self.best_prob, goal_uncertainty
        else:
            return None, None, None