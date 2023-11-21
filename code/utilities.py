import numpy as np
import sys
import scipy.spatial as scsp
from numba import njit, prange

# Import local code
import FCM_functions as FCM


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


@njit(parallel=False, fastmath=True)
def gr_numba(r_vectors, L, list_of_neighbors, offsets, rcut, nbins, Nblobs_body):
  '''
  This function compute the gr for one snapshot.
  '''
  N = r_vectors.size // 2
  r_vectors = r_vectors.reshape((N, 2))
  dbin = rcut / nbins
  gr = np.zeros((nbins, 2))

  # Copy arrays
  rx_vec = np.copy(r_vectors[:,0])
  ry_vec = np.copy(r_vectors[:,1])
  Lx = L[0]
  Ly = L[1]

  for i in prange(N):
    i_body = i // Nblobs_body
    for k in range(offsets[i+1] - offsets[i]):
      j = list_of_neighbors[offsets[i] + k]
      j_body = j // Nblobs_body
      if (i >= j) or (i_body == j_body):
        continue
      rx = rx_vec[j] - rx_vec[i]
      ry = ry_vec[j] - ry_vec[i]

      # Use distance with PBC
      if Lx > 0:
        rx = rx - int(rx / Lx + 0.5 * (int(rx>0) - int(rx<0))) * Lx
      if Ly > 0:
        ry = ry - int(ry / Ly + 0.5 * (int(ry>0) - int(ry<0))) * Ly

      # Compute distance
      r_norm = np.sqrt(rx*rx + ry*ry)
      xbin = int(r_norm / dbin)
      if xbin < nbins:
        gr[xbin, :] += 2

  return gr


def radial_distribution_function(x, num_frames, rcut=1.0, nbins=100, r_vectors=None, L=np.ones(3), name=None, header=''):
  '''
  Compute radial distribution function between bodies or blobs.
  It assumes all bodies are the same.

  '''
  # Prepare variables
  M = x.shape[0] if x.shape[0] < num_frames else num_frames
  Nblobs_body = 1
  N = x.shape[1] 
  dbin = rcut / nbins
  gr = np.zeros((nbins, 3))
  gr[:,0] = np.linspace(0, rcut, num=nbins+1)[:-1] + dbin / 2

  # Loop over frames
  for i, xi in enumerate(x[0:M]):
    z = xi[:,0:2]
    
    # Project to PBC
    z = FCM.project_to_periodic_image(np.copy(z), L)

    # Set box dimensions for PBC
    if L[0] > 0 or L[1] > 0:
      boxsize = np.zeros(2)
      for j in range(2):
        if L[j] > 0:
          boxsize[j] = L[j]
        else:
          boxsize[j] = (np.max(z[:,j]) - np.min(z[:,j])) + rcut * 10
    else:
      boxsize = None   

    # Build tree 
    tree = scsp.cKDTree(z, boxsize=boxsize)
    pairs = tree.query_ball_tree(tree, rcut)
    offsets = np.zeros(len(pairs)+1, dtype=int)
    for j in range(len(pairs)):
      offsets[j+1] = offsets[j] + len(pairs[j])
    list_of_neighbors = np.concatenate(pairs).ravel()

    gri = gr_numba(z, L, list_of_neighbors, offsets, rcut, nbins, Nblobs_body)
    gr[:,1:3] += gri[:]
      
  # Normalize gr
  factor = np.pi * ((gr[:,0] + dbin / 2)**2 - (gr[:,0] - dbin / 2)**2) * M * N**2 / (L[0] * L[1])
  gr[:,1] = gr[:,1] / factor

  # Save gr
  if name is not None:
    if len(header) == 0:
      header='Columns: r, gr density, gr count number'
      np.savetxt(name, gr, header=header)
      
  return gr
