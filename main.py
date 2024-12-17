import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from RRT import RRT
from RRT_Connect import RRT_Connect
from RRBT import RRBT
import math
import inquirer  # You'll need to: pip install inquirer

def get_rrbt_params(goal):
    """Interactive prompt for RRBT-specific parameters using inquirer"""
    # Show all default values first
    defaults = {
        'process_noise': 0.01,
        'delta': 0.159,
        'goal_region': (goal, 15),
        'sigma': 0.01,
        'measurement_zone': (0, 100, 100, 75)
        # 'measurement_zone': (30, 70, 100, 90)
    }
    
    print("\nDefault RRBT parameters:")
    for key, value in defaults.items():
        print(f"{key}: {value}")
    
    # Ask if defaults are okay using inquirer
    questions = [
        inquirer.List('use_defaults',
                     message='Use these default values?',
                     choices=['Yes', 'No'],
                     default='Yes')
    ]
    
    if inquirer.prompt(questions)['use_defaults'] == 'Yes':
        return {
            'process_noise': defaults['process_noise'],
            'delta': defaults['delta'],
            'goal_region': defaults['goal_region'],
            'start_uncertainty': defaults['sigma'] * np.eye(2),
            'measurement_zone': defaults['measurement_zone']
        }
    
    params = defaults.copy()    
    
    # Create list of parameters that can be changed
    param_choices = [
        ('Process noise', 'process_noise'),
        ('Delta (collision probability)', 'delta'),
        ('Goal region', 'goal_region'),
        ('Initial state uncertainty', 'sigma'),
        ('Measurement zone', 'measurement_zone')
    ]
    
    questions = [
        inquirer.Checkbox('params_to_change',
                         message='Select parameters to modify (use spacebar to select, enter to confirm)',
                         choices=param_choices)
    ]
    to_change = inquirer.prompt(questions)['params_to_change']
    
    # For each selected parameter, prompt for new value
    for param_name, param_key in param_choices:
        if param_key in to_change:
            if param_key == 'goal_region':
                questions = [
                    inquirer.Text('length',
                                message='Enter goal region length',
                                default=str(params[param_key][1]))
                ]
                length = float(inquirer.prompt(questions)['length'])
                params[param_key] = (goal, length)  # Keep center point, update length
            if param_key in ['process_noise', 'delta', 'sigma']:
                questions = [
                    inquirer.Text(param_key,
                                message=f'Enter new value for {param_name}',
                                default=str(params[param_key]))
                ]
                params[param_key] = float(inquirer.prompt(questions)[param_key])
            else:  # goal_region or measurement_zone
                print(f"\nEnter new bounds for {param_name}:")
                questions = [
                    inquirer.Text('left', message='Left bound', default=str(params[param_key][0])),
                    inquirer.Text('right', message='Right bound', default=str(params[param_key][1])),
                    inquirer.Text('bottom', message='Bottom bound', default=str(params[param_key][2])),
                    inquirer.Text('top', message='Top bound', default=str(params[param_key][3]))
                ]
                answers = inquirer.prompt(questions)
                params[param_key] = (
                    float(answers['left']),
                    float(answers['right']),
                    float(answers['bottom']),
                    float(answers['top'])
                )
    
    return {
        'process_noise': params['process_noise'],
        'delta': params['delta'],
        'goal_region': params['goal_region'],
        'start_uncertainty': params['sigma'] * np.eye(2),
        'measurement_zone': params['measurement_zone']
    }
    
def main(obs, alg, animation, debug_mode=True):
    # Get start and goal configurations
    start, goal = get_start_goal_config()
    
    # Get obstacle configuration
    obstacle = get_obstacle_config(obs, start, goal)
    
    if alg == "RRBT":
        # Get RRBT-specific parameters through interactive prompt
        rrbt_params = get_rrbt_params(goal)
        
        # Modify start to include uncertainty
        start = (start, rrbt_params['start_uncertainty'])
        
        # Initialize planner with RRBT parameters
        planner = RRBT(
            start=start,
            goal_region=rrbt_params['goal_region'],
            obstacle=obstacle,
            workspace=[0, 100],
            animation=animation,
            measurement_zone=rrbt_params['measurement_zone'],
            process_noise=rrbt_params['process_noise'],
            delta=rrbt_params['delta']
        )
    else:
        # Initialize RRT or RRT_Connect as before
        planner = select_planner(alg, start, goal, obstacle, animation)
    
    if debug_mode:
        run_debug(planner)
    else:
        run_testing(planner, alg, obs)

def get_start_goal_config():
    """Get default start and goal configurations
    
    Returns:
        tuple: (start_pos, goal_pos) where each is [x, y]
    """
    start = [25.0, 50.0]
    goal = [75.0, 50.0]
    return start, goal

def get_obstacle_config(obs_index, start, goal):
    """Get obstacle configuration based on index
    
    Args:
        obs_index: Integer 1-5 selecting obstacle configuration
        start: Start position [x, y]
        goal: Goal position [x, y]
        
    Returns:
        List of obstacle specifications
    """
    # Define obstacles
    bug_trap_start = [[start, 30, "vertical", (12.5, 0)],
                      [start, 20, "horizontal", (0, -12.5)],
                      [start, 20, "horizontal", (0, 12.5)],
                      [start, 14, "vertical", (-12.5, 8)],
                      [start, 14, "vertical", (-12.5, -8)]]
    
    bug_trap_goal = [[goal, 30, "vertical", (-12.5, 0)],
                     [goal, 20, "horizontal", (0, -12.5)],
                     [goal, 20, "horizontal", (0, 12.5)],
                     [goal, 14, "vertical", (12.5, 8)],
                     [goal, 14, "vertical", (12.5, -8)]]
    
    obstacles = [(25, 10), (4, 10), bug_trap_start, bug_trap_goal,
                 bug_trap_start + bug_trap_goal]
                 
    return obstacles[obs_index-1]

def compute_path_length(path):
    if len(path) < 2:
        return 0

    total_length = 0.0
    for i in range(1, len(path)):
        x1, y1 = path[i-1]
        x2, y2 = path[i]
        distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        total_length += distance

    return total_length

def select_planner(alg, start, goal, obstacle, animation):
    rrt = None
    if alg == "RRT":
        rrt = RRT(
            start=start,
            goal=goal,
            obstacle=obstacle,
            workspace=[0, 100],
            animation=animation
        )
    elif alg == "RRT_Connect":
        rrt = RRT_Connect(
            start=start,
            goal=goal,
            obstacle=obstacle,
            workspace=[0, 100],
            animation=animation
        )
    return rrt

def histograms(alg, obs, iterations, num_verts):
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.hist(iterations, bins=10, color='blue', alpha=0.7, edgecolor='black')
    plt.title("Iterations Histogram")
    plt.xlabel("Iterations")
    plt.ylabel("Frequency")

    plt.subplot(1, 2, 2)
    plt.hist(num_verts, bins=10, color='green', alpha=0.7, edgecolor='black')
    plt.title("Number of Vertices Histogram")
    plt.xlabel("Number of Vertices")
    plt.ylabel("Frequency")

    plt.tight_layout()
    plt.savefig(f"figures/hist_{alg}_{obs}.png", format="png", dpi=300)

def save_medians_to_csv(iterations, num_verts, paths, alg, obs, csv_file="results.csv"):
    median_iterations = np.median(iterations)
    median_num_verts = np.median(num_verts)
    median_paths = np.median(paths)

    column_name = f"{alg}_{obs}"

    try:
        df = pd.read_csv(csv_file, index_col=0)
    except FileNotFoundError:
        df = pd.DataFrame(index=["iterations", "num_verts", "paths"])

    df[column_name] = [median_iterations, median_num_verts, median_paths]

    df.to_csv(csv_file)

def run_debug(planner, save_plot=False):
    """Single run mode for debugging the algorithm
    
    Args:
        planner: Initialized planner object
        save_plot: Whether to save the final plot
    """
    print("Running debug mode (single iteration)...")
    path = planner.planning()
    
    if planner.animation:
        planner.update_graph()
        plt.plot([x for (x, _) in path], [y for (_, y) in path], '-r')
        plt.grid(True)
        plt.pause(0.01)
        if save_plot:
            plt.savefig(f"figures/debug_path.png", format="png", dpi=300)
        else:
            plt.show()
    
    return path

def run_testing(planner, alg, obs, num_iters=100):
    """Multiple run mode for testing algorithm performance
    
    Args:
        planner: Initialized planner object
        alg: Algorithm name for saving results
        obs: Obstacle configuration number
        num_iters: Number of iterations to run
    """
    print(f"Running testing mode ({num_iters} iterations)...")
    paths = []
    planners = []
    
    for i in range(num_iters):
        print(f"Iteration: {i}")
        # Only show animation on last iteration
        planner.animation = (i == num_iters-1)
        planners.append(planner)
        paths.append(planner.planning())
        
        if planner.animation:
            planner.update_graph()
            plt.plot([x for (x, _) in paths[-1]], [y for (_, y) in paths[-1]], '-r')
            plt.grid(True)
            plt.pause(0.01)
            plt.savefig(f"figures/final_path_{alg}_{obs}.png", format="png", dpi=300)
    
    # Generate statistics
    iterations = [p.iterations for p in planners]
    num_verts = [p.num_vertices for p in planners]
    histograms(alg, obs, iterations, num_verts)
    save_medians_to_csv(iterations, num_verts, 
                        [compute_path_length(p) for p in paths], alg, obs)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Run specific RRT algorithms with selected obstacles.")
    parser.add_argument(
        "--animation",
        type=str,
        choices=["t", "f"],
        default="t",
        help="Select True or False for animation"
    )
    parser.add_argument(
        "--obstacle",
        type=int,
        choices=[1, 2, 3, 4, 5],
        default=0,
        help="Select the obstacle index (1-5)."
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        choices=["RRT", "RRT_Connect", "RRBT"],
        default="RRT",
        help="Select the RRT algorithm to run."
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["debug", "test"],
        default="debug",
        help="Select running mode: debug (single run) or test (multiple runs)"
    )
    args = parser.parse_args()
    animation = True if args.animation == "t" else False
    debug_mode = True if args.mode == "debug" else False
    main(args.obstacle, args.algorithm, animation, debug_mode)
