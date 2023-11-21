import numpy as np
import sys


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
        options[option] = value

  return options
  

def read_config(name):
  '''
  Read config and store in an array of shape (num_frames, num_bodies, 7).
  '''

  # Read number of lines and bodies
  N = 0
  try:
    with open(name, 'r') as f_handle:
      num_lines = 0
      N = 0
      for line in f_handle:
        if num_lines == 0:
          N = int(line)
        num_lines += 1
  except OSError:
    return np.array([])

  # Set array 
  num_frames = num_lines // (N + 1) 
  x = np.zeros((num_frames, N, 3)) 

  # Read config
  with open(name, 'r') as f_handle:
    for k, line in enumerate(f_handle):
      if (k % (N+1)) == 0:
        continue
      else:
        data = np.fromstring(line, sep=' ')
        frame = k // (N+1)      
        i = k - 1 - (k // (N+1)) * (N+1)
        if frame >= num_frames:
          break
        x[frame, i] = np.copy(data)

  # Return config
  return x
