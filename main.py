import utils
from SaTPPO import main
from NS2CO_env import env
import numpy as np

if __name__ == '__main__':
    """
    Entry point of the script.
    
    This block initializes the necessary parameters, sets up the environment for reproducibility, 
    creates directories for saving results, saves the configuration, and starts the main training process.
    """
    # Import the argument parser module and initialize the arguments.
    from arg_PPO import arg_PPO
    args = arg_PPO()
    
    # Set the algorithm name to "SaTPPO".
    args.alg = "SaTPPO"
    
    
    # Import the environment class and create an environment instance
    my_env = env(args=args)
    # Set the dimensions of state and action spaces
    args.action_n = my_env.action_n
    args.state_n = my_env.state_n
    args.S_action_n = my_env.S_action_n
    args.U_action_n = my_env.U_action_n
    
    # Set the random seed for reproducibility.
    seed = 1
    seed = np.random.randint(1, 10000)
    utils.seed_everything(seed)
    
    # Create a directory to save experiment results and models.
    utils.make_savedir(args)
    
    # Save the current configuration to the specified path.
    utils.save_args(args, args.full_save_path)
    
    # Initialize a log file to record the training process.
    with open(args.full_save_path + "/" + "log.txt", 'w') as f:
        f.write('start' + '\n')
    
    # Start the main training function with the initialized arguments.
    main(args, my_env)
