import numpy as np
import sys
import argparse

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
  n_save = int(read.get('n_save'))
  eta = float(read.get('eta'))

  # Read particles configuration
  x = FCM.read_config(read.get('structure'))
  
  # Open files

  # Time loop

  # Save configuration

  # Advance time step

  # Save last configuration if necessary

  
