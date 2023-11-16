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
  parameters = FCM.read_input(input_file)

  # Some input variables should be converted to integers or set to default values if they do not exist
  parameters['n_steps'] = int(parameters['n_steps']) if 'n_steps' in parameters else 1
  parameters['n_save'] = int(parameters['n_save']) if 'n_save' in parameters else 1
  parameters['initial_step'] = int(parameters['initial_step']) if 'initial_step' in parameters else 0
  parameters['M_system'] = np.array(parameters['M_system'], dtype=int) if 'M_system' in parameters else np.zeros(3)
  
  # Some input variables should be set to default values if they do not exist
  parameters['kT'] = parameters['n_steps'] if 'kT' in parameters else 0  
    
  # Read particles configuration
  x = FCM.read_config(parameters.get('structure'))
  
  # Open files
  if True:
    name = parameters.get('output_name') + '.config'
    config_file = open(name, 'w')
      
  # Time loop
  start_time = time.time()
  for step in range(parameters.get('initial_step'), parameters.get('n_steps')):

    # Save data
    if (step % parameters.get('n_save')) == 0 and step >= 0:    
      print('step = ', step, ', time = ', time.time() - start_time)
      
      # Save configuration
      config_file.write(str(x.shape[0]) + '\n')
      np.savetxt(config_file, x)
      
      # Advance time step
      FCM.advance_time_step(parameters.get('dt'), parameters.get('scheme'), step, parameters)

    
  # Save last configuration if necessary
  if ((step+1) % parameters.get('n_save')) == 0 and step+1 >= 0:    
    print('step = ', step+1, ', time = ', time.time() - start_time)

    # Save configuration
    config_file.write(str(x.shape[0]) + '\n')
    np.savetxt(config_file, x)
  
  
