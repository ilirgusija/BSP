import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from RandomTrees.RRT import RRT
from RandomTrees.RRT_Connect import RRT_Connect
from RandomTrees.RRBT import RRBT
from RandomTrees.RRBT_Anytime import RRBT_Anytime
from Roadmaps.PRM import PRM
from Roadmaps.BRM import BRM
from Roadmaps.BRM_minmax import BRM_minmax
import math
import inquirer
import time
import os

def main(obs, alg, animation, debug_mode=True):
    # Get start and goal configurations
    print("Entering main function...")
    start, goal = get_start_goal_config()
    obstacle = get_obstacle_config(obs, start, goal)
    
    meas = [(30, 70, 90, 100), (10, 90, 75, 100)]
    # meas = [(30, 70, 90, 100)]
    # meas = [(10, 90, 75, 100)]
    if debug_mode:
        planner = select_planner(alg, start, goal, obstacle, animation, meas[1], True)
        run_debug(planner)
    else:
        for m in meas:
            print("Entering testing mode...")
            run_testing(alg, m, start, goal, obs, animation)

# Configurations
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

# Get params
def get_params(goal):
    """Interactive prompt for planner-specific parameters using inquirer"""
    # Show all default values first
    defaults = {
        'process_noise': 0.01,
        'delta': 0.159,
        'goal_region': (goal, 10),
        'sigma': 2,
    }
    
    print("\nDefault Planner parameters:")
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
        }
    
    params = defaults.copy()    
    
    # Create list of parameters that can be changed
    param_choices = [
        ('Process noise', 'process_noise'),
        ('Delta (collision probability)', 'delta'),
        ('Goal region', 'goal_region'),
        ('Initial state uncertainty', 'sigma'),
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
    }
    
# Planner selection
def select_planner(alg, start, goal, obstacle, animation, m, debug):
    planner = None
    if debug:
        params = get_params(goal)
    else:
        defaults = {
        'process_noise': 0.01,
        'delta': 0.4,
        'goal_region': (goal, 15),
        'sigma': 2,
        }
        params = {
            'process_noise': defaults['process_noise'],
            'delta': defaults['delta'],
            'goal_region': defaults['goal_region'],
            'start_uncertainty': defaults['sigma'] * np.eye(2),
        }
    if alg == "RRBT":
        # Get RRBT-specific parameters through interactive prompt
        start = (start, params['start_uncertainty'])
        planner = RRBT(
            start=start,
            goal_region=params['goal_region'],
            obstacle=obstacle,
            workspace=[0, 100],
            measurement_zone=m,
            animation=animation,
            delta=params['delta'],
            process_noise=params['process_noise'],
            debug=debug
        )
    elif alg == "RRBT_Anytime":
        # Get RRBT-specific parameters through interactive prompt
        start = (start, params['start_uncertainty'])
        planner = RRBT_Anytime(
            start=start,
            goal_region=params['goal_region'],
            obstacle=obstacle,
            workspace=[0, 100],
            measurement_zone=m,
            animation=animation,
            delta=params['delta'],
            process_noise=params['process_noise'],
            debug=debug
        )
    elif alg == "BRM":
        start = (start, params['start_uncertainty'])
        planner = BRM(
            start=start,
            goal=goal,
            obstacle=obstacle,
            workspace=[0, 100],
            measurement_zone=m,
            process_noise=params['process_noise'],
            animation=animation,
            debug=debug
        )
    elif alg == "BRM_minmax":
        start = (start, params['start_uncertainty'])
        planner = BRM_minmax(
            start=start,
            goal=goal,
            obstacle=obstacle,
            workspace=[0, 100],
            measurement_zone=m,
            process_noise=params['process_noise'],
            animation=animation,
            debug=debug
        )
    elif alg == "RRT":
        planner = RRT(
            start=start,
            goal=goal,
            obstacle=obstacle,
            workspace=[0, 100],
            animation=animation
        )
    elif alg == "RRT_Connect":
        planner = RRT_Connect(
            start=start,
            goal=goal,
            obstacle=obstacle,
            workspace=[0, 100],
            animation=animation
        )
    elif alg == "PRM":
        planner = PRM(
            start=start,
            goal=goal,
            obstacle=obstacle,
            workspace=[0, 100],
            animation=animation
        )
    return planner

# Running modes
def run_debug(planner, save_plot=True):
    """Single run mode for debugging the algorithm
    
    Args:
        planner: Initialized planner object
        save_plot: Whether to save the final plot
    """
    print("Running debug mode (single iteration)...")
    path = planner.planning()
    
    if planner.animation:
        if save_plot:
            if isinstance(planner, BRM):
                planner_s = "BRM_minmax" if isinstance(planner, BRM_minmax) else "BRM"
            elif isinstance(planner, PRM):
                planner_s = "PRM"
            elif isinstance(planner, RRBT):
                planner_s = "RRBT"
            else:
                planner_s = "RRT" if isinstance(planner, RRT) else "RRT_Connect"
            save_plot_f(planner_s)
        else:
            if isinstance(planner,RRBT) or isinstance(planner,RRBT_Anytime) or isinstance(planner, BRM) or isinstance(planner, BRM_minmax):
                path, _, _ = path
            elif isinstance(planner, RRT):
                planner.update_graph()
                plt.plot([x for (x, _) in path], [y for (_, y) in path], '-r')
                plt.grid(True)
                plt.pause(0.01)
            plt.show()
    return path

def run_testing(alg, m, start, goal, obs, animation, num_iters=100):
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
    times = []
    path_lengths = []
    goal_uncertainties = []
    tings = []
    fails=0
    obstacle=get_obstacle_config(obs, start, goal)
    
    for i in range(num_iters):
        print(f"Iteration: {i}")
        # Only show animation on last iteration
        planner = select_planner(alg, start, goal, obstacle, animation, m, False)
        planner.animation = (i == num_iters-1)
        planners.append(planner)
        
        # time the planning process
        try:
            start_time = time.time()
            run = planner.planning()
            elapsed_time = time.time() - start_time
        except Exception as e:
            print(f"Error in iteration {i}: {e}")
            i=i-1
            fails+=1
            continue
        if run is False:
            print("No path found.")
            fails+=1
            continue
        path, ting, uncertainty = run 
            
        paths.append(path)
        times.append(elapsed_time)
        path_lengths.append(compute_path_length(path))
        tings.append(ting)
        goal_uncertainties.append(uncertainty)
        
        if planner.animation:
            if isinstance(planner,BRM):
                planner_s = "BRM_minmax" if isinstance(planner, BRM_minmax) else "BRM"
            elif isinstance(planner, PRM):
                planner_s = "PRM"
            elif isinstance(planner, RRBT):
                planner_s = "RRBT"
            else:
                planner_s = "RRT_Connect" if isinstance(planner, RRT_Connect) else "RRT"
            save_plot_f(planner_s, False)
    
    # Generate statistics
    if alg =="RRBT" or alg == "RRBT_Anytime" or alg=="BRM" or alg=="BRM_minmax":
        print("Timing statistics for all iterations:")
        print(f"Mean time: {np.mean(times):.4f} seconds")
        print(f"Median time: {np.median(times):.4f} seconds")
        print(f"Max time: {np.max(times):.4f} seconds")
        print(f"Min time: {np.min(times):.4f} seconds")
        print(f"Total fails: {fails}")
        save_BSP_results_to_csv(times, path_lengths, tings, goal_uncertainties, alg, obs, m)

# Testing Helpers
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

def save_BSP_results_to_csv(times, path_lengths, tings, goal_uncertainties, alg, obs, meas, csv_file="results.csv"):
    """
    Save the results of the BSP algorithm to a CSV file.

    Args:
        times (list): List of times for each run.
        path_lengths (list): List of path lengths for each run.
        goal_uncertainties (list): List of goal uncertainties for each run.
        alg (str): Algorithm name.
        obs (int): Obstacle configuration number.
        meas (str): Measurement zone description.
        csv_file (str): Default CSV file name.
    """
    # Construct the save directory
    project_root = get_project_root()
    save_dir = os.path.join(project_root, "results")
    os.makedirs(save_dir, exist_ok=True)
    print(f"goal_uncertainties:")
    print(goal_uncertainties)

    # Construct the file name
    m = 1 if meas == (30, 70, 90, 100) else 2 if meas == (10, 90, 75, 100) else 3
    file_name = f"{alg}_{obs}_{m}_{csv_file}"
    save_path = os.path.join(save_dir, file_name)

    # Create a DataFrame
    if alg =="BRM_minmax":
        df = pd.DataFrame({
            "time": times,
            "path_length": path_lengths,
            "goal_uncertainty": goal_uncertainties,
            "max_goal_uncertainty": tings
        })
    elif alg =="RRBT" or alg == "RRBT_Anytime":
        df = pd.DataFrame({
            "time": times,
            "path_length": path_lengths,
            "goal_uncertainty": goal_uncertainties,
            "p_goal": tings
        })
    else:
        df = pd.DataFrame({
            "time": times,
            "path_length": path_lengths,
            "goal_uncertainty": goal_uncertainties
        })
    
     # Add row numbering
    df.index = range(1, len(df) + 1)  # Start row numbering at 1
    df.index.name = "row_number"  # Name the index column

    # Save to CSV
    df.to_csv(save_path, index=False)
    print(f"Results saved to {save_path}")

def get_project_root():
    """Fetch the root of the project directory."""
    current_dir = os.path.abspath(__file__)
    while not os.path.exists(os.path.join(current_dir, ".git")) and not current_dir.endswith("BSP"):
        current_dir = os.path.dirname(current_dir)
    return current_dir

def save_plot_f(planner_s, debug=True):
    current_time = time.strftime("%Y%m%d_%H%M%S")
    string = planner_s + f"_{current_time}"
    
    # determine project root and construct save path
    run_type = "debug" if debug else "test"
    save_dir = os.path.join(get_project_root(), f"figures/{run_type}/{planner_s}")
    os.makedirs(save_dir, exist_ok=True)
    
    # save plot
    save_path = os.path.join(save_dir, f"{string}.png")
    plt.savefig(save_path, format="png", dpi=300)
    print(f"Saved plot to: {save_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Run specific planning algorithms with selected obstacles.")
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
        choices=["PRM", "RRT", "RRT_Connect", "RRBT", "BRM", "RRBT_Anytime", "BRM_minmax"],
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
