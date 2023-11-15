import numpy as np
import sys
import argparse
import time

# Import local code
import FCM_functions as FCM

if __name__ == '__main__':
  # Get command line arguments
  parser = argparse.ArgumentParser(description='Run a multi-body simulation and save trajectory.')
  parser.add_argument('--input-file', dest='input_file', type=str, default='data.main', help='name of the input file')
  parser.add_argument('--print-residual', action='store_true', help='print gmres and lanczos residuals')
  args=parser.parse_args()
  input_file = args.input_file

  # Read input file
  read = FCM.read_input(input_file)

  # Set some parameters
  dt = float(read.get('dt')) 
  n_save = int(read.get('n_save') if read.get('n_save') else 1)
  n_steps = int(read.get('n_steps'))  
  initial_step = int(read.get('initial_step') if read.get('initial_step') else 0)    
  eta = float(read.get('eta'))
  kT = float(read.get('kT') if read.get('kT') else 0)
    
  # Read particles configuration
  x = FCM.read_config(read.get('structure'))
  
  # Open files
  if True:
    name = read.get('output_name') + '.config'
    config_file = open(name, 'w')
      
  # Time loop
  start_time = time.time()
  for step in range(initial_step, n_steps):

    # Save data
    if (step % n_save) == 0 and step >= 0:    
      print('step = ', step, ', time = ', time.time() - start_time)
      
      # Save configuration
      config_file.write(str(x.shape[0]) + '\n')
      np.savetxt(config_file, x)
      
    # Advance time step
    

    
  # Save last configuration if necessary
  if ((step+1) % n_save) == 0 and step+1 >= 0:    
    print('step = ', step+1, ', time = ', time.time() - start_time)

    # Save configuration
    config_file.write(str(x.shape[0]) + '\n')
    np.savetxt(config_file, x)
  
  
