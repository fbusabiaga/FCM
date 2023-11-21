import numpy as np
import sys
import argparse
import scipy.spatial as scsp
from numba import njit, prange


# Try to import the visit_writer (boost implementation)
# sys.path.append('../../RigidMultiblobsWall/')
sys.path.append('../')
try:
  # import visit.visit_writer as visit_writer
  from visit import visit_writer as visit_writer
except ImportError as e:
  print(e)
  pass


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
  xx, yy = np.meshgrid(grid_x, grid_y, indexing = 'ij')
  r_mesh = np.zeros((num_points, 2))
  r_mesh[:,0] = np.reshape(xx, xx.size)
  r_mesh[:,1] = np.reshape(yy, yy.size)
  
  # Create velocity field
  velocity_mesh = np.zeros((num_points, 2))
  vx_mesh = np.zeros((M[0], M[1]))
  vy_mesh = np.zeros((M[0], M[1]))  
  
  return r_mesh, velocity_mesh, vx_mesh, vy_mesh


def plot_velocity_field(L, M, vx_mesh, vy_mesh, output):
  '''
  This function plots the velocity field to a grid using boost visit writer
  '''
  # Set grid coordinates along axes
  dx = L / M
  grid_x = np.array([0 + dx[0] * (x+0.5) for x in range(M[0])])
  grid_y = np.array([0 + dx[1] * (x+0.5) for x in range(M[1])])
  
  # Prepare grid values
  velocity = np.zeros((M[0] * M[1], 3))
  for ix in range(M[0]):
    for iy in range(M[1]):
      velocity[iy * M[0] + ix, 0] = vx_mesh[ix, iy]
      velocity[iy * M[0] + ix, 1] = vy_mesh[ix, iy]
  
  # Prepare data for VTK writer 
  variables = [np.reshape(velocity, velocity.size)] 
  dims = np.array([M[0]+1, M[1]+1, 1], dtype=np.int32) 
  nvars = 1
  vardims = np.array([3])
  centering = np.array([0])
  varnames = ['velocity\0']
  name = output + '.velocity_field.vtk'
  grid_x = np.array([dx[0] * x for x in range(M[0] + 1)])
  grid_y = np.array([dx[1] * x for x in range(M[1] + 1)])  
  grid_z = np.zeros(1) 
  
  # Write velocity field
  visit_writer.boost_write_rectilinear_mesh(name,      # File's name
                                            0,         # 0=ASCII,  1=Binary
                                            dims,      # {mx, my, mz}
                                            grid_x,    # xmesh
                                            grid_y,    # ymesh
                                            grid_z,    # zmesh
                                            nvars,     # Number of variables
                                            vardims,   # Size of each variable, 1=scalar, velocity=3*scalars
                                            centering, # Write to cell centers of corners
                                            varnames,  # Variables' names
                                            variables) # Variables
  return


def Gaussian(x, sigma):
  return np.exp(-np.linalg.norm(x)**2 / (2 * sigma**2)) / (2 * np.pi * sigma**2)


@njit(parallel=True, fastmath=True)
def interpolate(x, vx_mesh, vy_mesh, sigma_u, sigma_w, L, M):
  '''
  Interpolate fluid velocity.
  '''
  velocity_particles = np.zeros_like(x)

  # Prepare some variables
  dx = L[0] / M[0]
  dy = L[1] / M[1]
  N = 3 * np.sqrt(np.pi) * sigma_u / min(dx, dy)
  sigma_u2 = sigma_u**2
  sigma_w2 = sigma_w**2  

  # Get vectors in the minimal image representation of size L=(Lx, Ly) and with a corner at (0,0).
  x_PBC = np.empty_like(x[:,0:2])
  for axis in range(2):
    x_PBC[:,axis] = x[:,axis] - (x[:,axis] // L[axis]) * L[axis]

  # Loop over particles
  for n in prange(x.shape[0]):
    x_disp = np.zeros(2)
    kx = int(x_PBC[n,0]  / L[0] * M[0])
    ky = int(x_PBC[n,1]  / L[1] * M[1])

    # Loop over neighboring cells
    for cellx in range(-N, N, 1):
      kx_PBC = kx + cellx - ((kx + cellx) // M[0]) * M[0]
      x_disp[0] = dx * (kx + cellx + 0.5) - x_PBC[n,0]
      
      for celly in range(-N, N, 1):
        ky_PBC = ky + celly - ((ky + celly) // M[1]) * M[1]
        x_disp[1] = dy * (ky + celly + 0.5) - x_PBC[n,1]
        factor = np.exp(-np.linalg.norm(x_disp)**2 / (2 * sigma_u2)) / (2 * np.pi * sigma_u2) * dx * dy
        velocity_particles[n,0] += vx_mesh[kx_PBC, ky_PBC] * factor
        velocity_particles[n,1] += vy_mesh[kx_PBC, ky_PBC] * factor
    
  return velocity_particles


@njit(parallel=False, fastmath=True)
def spread(x, force_torque, sigma_u, sigma_w, L, M):
  '''
  Interpolate fluid velocity.
  '''
  fx_mesh = np.zeros((M[0], M[1]))
  fy_mesh = np.zeros((M[0], M[1]))

  # Prepare some variables
  dx = L[0] / M[0]
  dy = L[1] / M[1]
  N = 3 * np.sqrt(np.pi) * sigma_u / min(dx, dy)
  sigma_u2 = sigma_u**2
  factor_gaussian = 1.0 / (2 * np.pi * sigma_u2)
  x_disp = np.zeros(2)

  # Get vectors in the minimal image representation of size L=(Lx, Ly) and with a corner at (0,0).
  x_PBC = np.empty_like(x[:,0:2])
  for axis in range(2):
    x_PBC[:,axis] = x[:,axis] - (x[:,axis] // L[axis]) * L[axis]  

  # Loop over particles
  for n in range(x.shape[0]):
    kx = int(x_PBC[n,0] / L[0] * M[0])
    ky = int(x_PBC[n,1] / L[1] * M[1])

    # Loop over neighboring cells
    for cellx in range(-N, N, 1):
      kx_PBC = kx + cellx - ((kx + cellx) // M[0]) * M[0]
      x_disp[0] = dx * (kx + cellx + 0.5) - x_PBC[n,0]
      
      for celly in range(-N, N, 1):
        ky_PBC = ky + celly - ((ky + celly) // M[1]) * M[1]
        x_disp[1] = dy * (ky + celly + 0.5) - x_PBC[n,1]
        factor = np.exp(-np.linalg.norm(x_disp)**2 / (2 * sigma_u2)) / (2 * np.pi * sigma_u2)

        # Spread force
        fx_mesh[kx_PBC, ky_PBC] += factor * force_torque[n,0]
        fy_mesh[kx_PBC, ky_PBC] += factor * force_torque[n,1]
    
  return fx_mesh, fy_mesh


@njit(parallel=False, fastmath=True)
def solve_Stokes_Fourier_spectral(fx_Fourier, fy_Fourier, gradKx, gradKy, LKx, LKy, expKx, expKy, eta, L, M):
  '''
  The solution is v = (1 - G*D/L) * force / (eta * L)

  with
  G = Gradient
  D = Divergence = Gradient^T
  L = Laplacian = D * G  
  '''
  vx_Fourier = np.zeros_like(fx_Fourier)
  vy_Fourier = np.zeros_like(fy_Fourier)  
  
  for kx in range(gradKx.shape[0]):        
    for ky in range(gradKy.shape[0]):
      # Get physical wave number
      wx_indx = -M[0]+kx if kx > M[0] // 2 else kx
      wy_indx = -M[1]+ky if ky > M[1] // 2 else ky

      xHalf = 1 if (M[0] % 2) == 0 and (wx_indx == M[0] // 2) else 0
      yHalf = 1 if (M[1] % 2) == 0 and (wy_indx == M[1] // 2) else 0

      wx = -2 * np.pi * wx_indx / L[0] if (xHalf and wy_indx < 0) else 2 * np.pi * wx_indx / L[0]
      wy = -2 * np.pi * wy_indx / L[1] if (yHalf and wx_indx < 0) else 2 * np.pi * wy_indx / L[1]

      wx = wx * 1.0j
      wy = wy * 1.0j
      
      # Compute Laplacian
      Lap = -wx.imag**2 - wy.imag**2
      if abs(Lap) < 1e-12:
        continue

      # Shift mode
      fx_Fourier[kx, ky] = fx_Fourier[kx, ky] * np.conjugate(expKx[kx]) * np.conjugate(expKy[ky]) 
      fy_Fourier[kx, ky] = fy_Fourier[kx, ky] * np.conjugate(expKx[kx]) * np.conjugate(expKy[ky]) 
               
      # Compute divergence of the force 
      div = wx * fx_Fourier[kx, ky] + wy * fy_Fourier[kx, ky]
        
      # Compute velocity 
      vx_Fourier[kx, ky] = -(fx_Fourier[kx, ky] - wx * div / Lap) / (eta * Lap) 
      vy_Fourier[kx, ky] = -(fy_Fourier[kx, ky] - wy * div / Lap) / (eta * Lap)

      # Shift mode
      vx_Fourier[kx, ky] = vx_Fourier[kx, ky] * expKx[kx] * expKy[ky] 
      vy_Fourier[kx, ky] = vy_Fourier[kx, ky] * expKx[kx] * expKy[ky]

  return vx_Fourier, vy_Fourier

@njit(parallel=False, fastmath=True)
def solve_Stokes_Fourier(fx_Fourier, fy_Fourier, gradKx, gradKy, LKx, LKy, expKx, expKy, eta):
  '''
  The solution is v = (1 - G*D/L) * force / (eta * L)

  with
  G = Gradient
  D = Divergence = Gradient^T
  L = Laplacian = D * G  
  '''
  vx_Fourier = np.zeros_like(fx_Fourier)
  vy_Fourier = np.zeros_like(fy_Fourier)  
  
  for kx in range(gradKx.shape[0]):
    for ky in range(gradKy.shape[0]):
      # Compute Laplacian
      L = -LKx[kx]**2 - LKy[ky]**2
      DG = -gradKx[kx].imag**2 - gradKy[ky].imag**2
      if abs(DG) < 1e-12:
        continue

      # Shift mode
      fx_Fourier[kx, ky] = fx_Fourier[kx, ky] * np.conjugate(expKx[kx]) * np.conjugate(expKy[ky]) 
      fy_Fourier[kx, ky] = fy_Fourier[kx, ky] * np.conjugate(expKx[kx]) * np.conjugate(expKy[ky]) 
               
      # Compute divergence of the force 
      div = gradKx[kx] * fx_Fourier[kx, ky] + gradKy[ky] * fy_Fourier[kx, ky]
        
      # Compute velocity 
      vx_Fourier[kx, ky] = -(fx_Fourier[kx, ky] - gradKx[kx] * div / DG) / (eta * L) 
      vy_Fourier[kx, ky] = -(fy_Fourier[kx, ky] - gradKy[ky] * div / DG) / (eta * L)

      # Shift mode
      vx_Fourier[kx, ky] = vx_Fourier[kx, ky] * expKx[kx] * expKy[ky] 
      vy_Fourier[kx, ky] = vy_Fourier[kx, ky] * expKx[kx] * expKy[ky]

  return vx_Fourier, vy_Fourier

  
def solve_Stokes(fx_mesh, fy_mesh, eta, kT, L, M, discretization='spectral'):
  '''
  Solve Stokes equation with PBC.  
  '''
  # Prepare variables
  dx = L[0] / M[0]
  dy = L[1] / M[1]

  # Prepare gradient arrays
  gradKx = (1 / dx) * np.sin(2 * np.pi * np.arange(M[0]) / M[0]) * 1.0j
  gradKy = (1 / dy) * np.sin(2 * np.pi * np.arange(M[1]) / M[1]) * 1.0j
  LKx =    (2 / dx) * np.sin(    np.pi * np.arange(M[0]) / M[0])
  LKy =    (2 / dy) * np.sin(    np.pi * np.arange(M[1]) / M[1])   
  expKx = np.cos(np.pi * np.arange(M[0]) / M[0]) + 1.0j * np.sin(np.pi * np.arange(M[0]) / M[0])
  expKy = np.cos(np.pi * np.arange(M[1]) / M[1]) + 1.0j * np.sin(np.pi * np.arange(M[1]) / M[1])
        
  # Transform fields to Fourier space
  fx_Fourier = np.fft.fft2(fx_mesh)
  fy_Fourier = np.fft.fft2(fy_mesh)

  # Solve in Fourier space
  if discretization == 'finite_volumes':
    vx_Fourier, vy_Fourier = solve_Stokes_Fourier(fx_Fourier, fy_Fourier, gradKx, gradKy, LKx, LKy, expKx, expKy, eta)
  elif discretization == 'spectral':
    vx_Fourier, vy_Fourier = solve_Stokes_Fourier_spectral(fx_Fourier, fy_Fourier, gradKx, gradKy, LKx, LKy, expKx, expKy, eta, L, M)

  
  # Transform velocities to real space
  vx_mesh = np.fft.ifft2(vx_Fourier)
  vy_mesh = np.fft.ifft2(vy_Fourier)

  return vx_mesh.real, vy_mesh.real


@njit(parallel=True, fastmath=True)
def force_torque_tree_numba(r_vectors, L, eps, b, a, list_of_neighbors, offsets):
  '''
  This function compute the force between two blobs
  with vector between blob centers r.

  In this example the torque=0 and the force is derived from the potential
  
  U(r) = U0 + U0 * (2*a-r)/b   if z<2*a
  U(r) = U0 * exp(-(r-2*a)/b)  iz z>=2*a
  
  with
  eps = potential strength
  r_norm = distance between blobs
  b = Debye length
  a = blob_radius
  '''
  N = r_vectors.size // 2
  r_vectors = r_vectors.reshape((N, 2))
  force_torque = np.zeros((N, 3))
  
  # Copy arrays
  rx_vec = np.copy(r_vectors[:,0])
  ry_vec = np.copy(r_vectors[:,1])
  Lx = L[0]
  Ly = L[1]

  for i in prange(N):
    for k in range(offsets[i+1] - offsets[i]):
      j = list_of_neighbors[offsets[i] + k]
      if i == j:
        continue
      rx = rx_vec[j] - rx_vec[i]
      ry = ry_vec[j] - ry_vec[i]

      # Use distance with PBC
      if Lx > 0:
        rx = rx - int(rx / Lx + 0.5 * (int(rx>0) - int(rx<0))) * Lx
      if Ly > 0:
        ry = ry - int(ry / Ly + 0.5 * (int(ry>0) - int(ry<0))) * Ly

      # Compute force
      r_norm = np.sqrt(rx*rx + ry*ry)
      if r_norm > 2*a:
        f0 = -((eps / b) * np.exp(-(r_norm - 2.0*a) / b) / r_norm)
      else:
        f0 = -((eps / b) / np.maximum(r_norm, 1e-25))
      force_torque[i, 0] += f0 * rx
      force_torque[i, 1] += f0 * ry
  return force_torque


def force_torque_pair_wise(x, L, parameters):
  '''
  This function computes the forces and torques between colloids.
  '''     
  # Get parameters from arguments
  eps = parameters.get('repulsion_strength')
  b = parameters.get('debye_length')
  a = parameters.get('particle_radius')
  d_max = 2 * a + 30 * b # Cutoff distance for interactions

  # Project to PBC, this is necessary here to build the Kd-tree with scipy.
  # Copy is necessary because we don't want to modify the original vector here
  r_vectors = project_to_periodic_image(np.copy(x[:,0:2]), L)

  # Set box dimensions for PBC
  if L[0] > 0 or L[1] > 0:
    boxsize = np.zeros(2)
    for i in range(2):
      if L[i] > 0:
        boxsize[i] = L[i]
      else:
        boxsize[i] = (np.max(r_vectors[:,i]) - np.min(r_vectors[:,i])) + d_max * 10
  else:
    boxsize = None   

  # Build tree
  tree = scsp.cKDTree(r_vectors, boxsize=boxsize)
  pairs = tree.query_ball_tree(tree, d_max)
  offsets = np.zeros(len(pairs)+1, dtype=int)
  for i in range(len(pairs)):
    offsets[i+1] = offsets[i] + len(pairs[i])
  list_of_neighbors = np.concatenate(pairs).ravel()
  
  # Compute forces and torques
  force_torque = force_torque_tree_numba(r_vectors, L, eps, b, a, list_of_neighbors, offsets)
  return force_torque


def force_torque_single(x, L, parameters):
  '''
  Compute the external force torque on the particles.
  '''

  return
  

def advance_time_step(dt, scheme, step, x, vel, parameters):
  '''
  Advance time step with integrator self.scheme
  '''
  if scheme == 'deterministic_forward_Euler_no_stresslet':
    return deterministic_forward_Euler_no_stresslet(dt, scheme, step, x, vel, parameters) 
  elif scheme == 'deterministic_forward_Euler':
    return deterministic_forward_Euler(dt, scheme, step, x, vel, parameters)
  else:
    print('Scheme: ', scheme, ' is not implemented.')
  return


def deterministic_forward_Euler_no_stresslet(dt, scheme, step, x, vel, parameters):
  '''
  Forward Euler scheme without including stresslet.
  '''
  # Get parameters
  eta = parameters.get('eta')  
  M = parameters.get('M_system')
  L = parameters.get('L_system')
  discretization = parameters.get('discretization')

  # Compute force between particles
  force_torque = force_torque_pair_wise(x, L, parameters)
  # force_torque += force_torque_single(x, L, parameters)
  print('force_torque = \n', force_torque)

  # Spread force
  fx_mesh, fy_mesh = spread(x, force_torque, parameters.get('sigma_u'), parameters.get('sigma_w'), L, M)
  print('Fx = ', np.sum(fx_mesh) * L[0] / M[0] * L[1] / M[1])
  print('Fy = ', np.sum(fy_mesh) * L[0] / M[0] * L[1] / M[1])
   
  # Solve Stokes equations
  vx_mesh, vy_mesh = solve_Stokes(fx_mesh, fy_mesh, eta, 0, L, M, discretization=discretization)

  # Plot fluid velocity field
  if parameters.get('plot_velocity_field') == 'True':
    plot_velocity_field(L, M, vx_mesh, vy_mesh, parameters.get('output_name')) 
  
  # Interpolate velocity
  velocity_particles = interpolate(x, vx_mesh, vy_mesh, parameters.get('sigma_u'), parameters.get('sigma_w'), L, M)
  vel[:,:] = velocity_particles
 
  # Advance particle positions
  x += dt * velocity_particles
  
  

