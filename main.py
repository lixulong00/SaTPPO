import utils
from SaTPPO import main
from NS2CO_env1 import env

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
    args.state_dim = my_env.state_dim
    args.action_S_dim = my_env.action_S_dim
    args.action_U_dim = my_env.action_U_dim
    args.action_dim = my_env.action_S_dim + my_env.action_U_dim
    
    # Set the random seed for reproducibility.
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
