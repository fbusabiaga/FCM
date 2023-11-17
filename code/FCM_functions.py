import numpy as np
import sys
import argparse
from numba import njit, prange


def is_number(s):
  '''
  Check if a string is a number or not.
  '''
  try:
    float(s)
    return True
  except ValueError:
    return False


def project_to_periodic_image(r, L):
  '''
  Project a vector r to the minimal image representation of size L=(Lx, Ly) and with a corner at (0,0).
  If any dimension of L is equal or smaller than zero the box is assumed to be infinite in that direction.
    
  If one dimension is not periodic shift all coordinates by min(r[:,i]) value.
  '''
  r_PBC = np.empty_like(r)
  for i in range(2):
    r_PBC[:,i] = r[:,i] - (r[:,i] // L[i]) * L[i]
  return r_PBC


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


def create_mesh(parameters):
  '''
  Create 2d mesh.
  '''
  L = parameters.get('L_system')
  M = parameters.get('M_system')
  num_points = M[0] * M[1]
  
  # Set grid coordinates along axes
  dx = L / M
  grid_x = np.array([0 + dx[0] * (x+0.5) for x in range(M[0])])
  grid_y = np.array([0 + dx[1] * (x+0.5) for x in range(M[1])])

  # Create mesh
  xx, yy = np.meshgrid(grid_y, grid_x, indexing = 'ij')
  r_mesh = np.zeros((num_points, 2))
  r_mesh[:,0] = np.reshape(xx, xx.size)
  r_mesh[:,1] = np.reshape(yy, yy.size)

  # Create velocity field
  velocity_mesh = np.zeros((num_points, 2))
  vx_mesh = np.zeros((M[0], M[1]))
  vy_mesh = np.zeros((M[0], M[1]))  
  
  return r_mesh, velocity_mesh, vx_mesh, vy_mesh


def Gaussian(x, sigma):
  return np.exp(-np.linalg.norm(x)**2 / (2 * sigma**2)) / (2 * np.pi * sigma**2)


@njit(parallel=True, fastmath=True)
def interpolate(x, r_mesh, vx_mesh, vy_mesh, sigma_u, sigma_w, L):
  '''
  Interpolate fluid velocity.
  '''
  velocity_particles = np.zeros_like(x)
  strain_rate_xx = np.zeros(r_mesh.size // 2)
  strain_rate_xy = np.zeros(r_mesh.size // 2)
  strain_rate_yx = np.zeros(r_mesh.size // 2)
  strain_rate_yy = np.zeros(r_mesh.size // 2)

  # Prepare some variables
  Lx = L[0]
  Ly = L[1]
  dx = L[0] / r_mesh.shape[0]
  dy = L[1] / r_mesh.shape[1]  
  N = 4 * np.sqrt(np.pi) * sigma_u / min(dx, dy)
  sigma_u2 = sigma_u**2
  factor_gaussian = 1.0 / (2 * np.pi * sigma_u2)
  x_disp = np.zeros(2)

  # Get vectors in the minimal image representation of size L=(Lx, Ly) and with a corner at (0,0).
  x_PBC = np.empty_like(x)
  for axis in range(2):
    x_PBC[:,axis] = x[:,axis] - (x[:,axis] // L[axis]) * L[axis]  

  # Loop over particles
  for n in prange(x.shape[0]):
    kx = int(x_PBC[n,0] / Lx * r_mesh.shape[0])
    ky = int(x_PBC[n,1] / Ly * r_mesh.shape[1])

    # Loop over neighboring cells
    for cellx in range(-N, N, 1):
      kx_PBC = kx + cellx - ((kx + cellx) // r_mesh.shape[0]) * r_mesh.shape[0]
      x_disp[0] = r_mesh[kx, 0, 0] - x_PBC[n,0] + cellx * dx
      
      for celly in range(-N, N, 1):
        ky_PBC = ky + celly - ((ky + celly) // r_mesh.shape[1]) * r_mesh.shape[1]
        x_disp[1] = r_mesh[0,ky, 1] - x_PBC[n,1] + celly * dy
        factor = np.exp(-np.linalg.norm(x_disp)**2 / (2 * sigma_u2)) / (2 * np.pi * sigma_u2) * dx * dy
        velocity_particles[n,0] += vx_mesh[kx_PBC, ky_PBC] * factor
        velocity_particles[n,1] += vy_mesh[kx_PBC, ky_PBC] * factor
    
  return velocity_particles, 2


@njit(parallel=False, fastmath=True)
def spread(x, force_torque, r_mesh, sigma_u, sigma_w, L):
  '''
  Interpolate fluid velocity.
  '''
  fx_mesh = np.zeros((r_mesh.shape[0], r_mesh.shape[1]))
  fy_mesh = np.zeros((r_mesh.shape[0], r_mesh.shape[1]))  

  # Prepare some variables
  Lx = L[0]
  Ly = L[1]
  dx = L[0] / r_mesh.shape[0]
  dy = L[1] / r_mesh.shape[1]  
  N = 4 * np.sqrt(np.pi) * sigma_u / min(dx, dy)
  sigma_u2 = sigma_u**2
  factor_gaussian = 1.0 / (2 * np.pi * sigma_u2)
  x_disp = np.zeros(2)

  # Get vectors in the minimal image representation of size L=(Lx, Ly) and with a corner at (0,0).
  x_PBC = np.empty_like(x)
  for axis in range(2):
    x_PBC[:,axis] = x[:,axis] - (x[:,axis] // L[axis]) * L[axis]  

  # Loop over particles
  for n in range(x.shape[0]):
    kx = int(x_PBC[n,0] / Lx * r_mesh.shape[0])
    ky = int(x_PBC[n,1] / Ly * r_mesh.shape[1])

    # Loop over neighboring cells
    for cellx in range(-N, N, 1):
      kx_PBC = kx + cellx - ((kx + cellx) // r_mesh.shape[0]) * r_mesh.shape[0]
      x_disp[0] = r_mesh[kx, 0, 0] - x_PBC[n,0] + cellx * dx
      
      for celly in range(-N, N, 1):
        ky_PBC = ky + celly - ((ky + celly) // r_mesh.shape[1]) * r_mesh.shape[1]
        x_disp[1] = r_mesh[0,ky, 1] - x_PBC[n,1] + celly * dy
        factor = np.exp(-np.linalg.norm(x_disp)**2 / (2 * sigma_u2)) / (2 * np.pi * sigma_u2)

        # Spread force
        fx_mesh[kx_PBC, ky_PBC] += factor * force_torque[n,0]
        fy_mesh[kx_PBC, ky_PBC] += factor * force_torque[n,1]
    
  return fx_mesh, fy_mesh

  

def advance_time_step(dt, scheme, step, x, r_mesh, velocity_mesh, vx_mesh, vy_mesh, parameters):
  '''
  Advance time step with integrator self.scheme
  '''
  if scheme == 'deterministic_forward_Euler_no_stresslet':
    return deterministic_forward_Euler_no_stresslet(dt, scheme, step, x, r_mesh, velocity_mesh, vx_mesh, vy_mesh, parameters)     
  elif scheme == 'deterministic_forward_Euler':
    return deterministic_forward_Euler(dt, scheme, step, r_mesh, velocity_mesh, parameters)
  else:
    print('Scheme: ', scheme, ' is not implemented.')
  return


def deterministic_forward_Euler_no_stresslet(dt, scheme, step, x, r_mesh, velocity_mesh, vx_mesh, vy_mesh, parameters):
  '''
  Forward Euler scheme without including stresslet.
  '''
  # Get parameters
  M = parameters.get('M_system')
  L = parameters.get('L_system')
  r_mesh = r_mesh.reshape((M[0], M[1], 2))
  
  # Compute force between particles
  force_torque = np.zeros((x.shape[0], 3))
  force_torque[0, 0] = 1
  
  # Spread force
  fx_mesh, fy_mesh = spread(x, force_torque, r_mesh, parameters.get('sigma_u'), parameters.get('sigma_w'), L)
  
  # Solve Stokes equations
  # vx_mesh[:,:] = 1.0
  vx_mesh = fx_mesh

  
  # Interpolate velocity
  velocity_particles, strain_rate = interpolate(x, r_mesh, vx_mesh, vy_mesh, parameters.get('sigma_u'), parameters.get('sigma_w'), L)

  print('velocity_particles = \n', velocity_particles)
  
  # Advance particle positions
  x += dt * velocity_particles
  print('x = ', x)

  


