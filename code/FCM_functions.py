import numpy as np
import sys
import argparse


def is_number(s):
  '''
  Check if a string is a number or not.
  '''
  try:
    float(s)
    return True
  except ValueError:
    return False

def read_input(name):
  '''
  Build a dictionary from an input file.
  The symbol # marks comments.
  '''
  
  # Comment symbol and empty dictionary
  comment_symbols = ['#']
  options = {}

  # Read input file
  with open(name, 'r') as f:
    # Loop over lines
    for line in f:
      # Strip comments
      if comment_symbols[0] in line:
        line, comment = line.split(comment_symbols[0], 1)

      # Save options to dictionary, Value may be more than one word
      line = line.strip()
      if line != '':
        option, value = line.split(None, 1)

        # Save as a number, numpy array or string
        if is_number(value.split(None)[0]):
          value = np.fromstring(value, sep=' ')
          options[option] = value[0] if value.size == 1 else value
        else:
          options[option] = value
   
  return options


def read_config(name):
  '''
  Read config and store in an array of shape (num_bodies, 3).
  '''
  # Read config
  N = 0
  x = []
  with open(name, 'r') as f_handle:
    for k, line in enumerate(f_handle):
      if k == 0:
        N = int(line)
      else:
        data = np.fromstring(line, sep=' ')
        x.append(data)
      if k >= N:
        break
  x = np.array(x)

  # Return config
  return x


def advance_time_step(dt, scheme, step, parameters):
  '''
  Advance time step with integrator self.scheme
  '''
  if scheme == 'deterministic_forward_Euler':
    return deterministic_forward_Euler(dt, scheme, step, parameters)
  else:
    print('Scheme: ', scheme, ' is not implemented.')
  return


def deterministic_forward_Euler(dt, scheme, step, parameters):
  '''
  Forward Euler scheme.
  '''
  # Compute force between particles

  # Spread force

  # Solve Stokes equations

  # Interpolate velocity

  # Advance particle positions



  
  


