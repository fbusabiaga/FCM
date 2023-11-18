import numpy as np
import sys
import argparse
from numba import njit, prange

# Try to import the visit_writer (boost implementation)
sys.path.append('../../RigidMultiblobsWall/')
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
  # strain_rate_xx = np.zeros(r_mesh.size // 2)
  # strain_rate_xy = np.zeros(r_mesh.size // 2)
  # strain_rate_yx = np.zeros(r_mesh.size // 2)
  # strain_rate_yy = np.zeros(r_mesh.size // 2)

  # Prepare some variables
  Lx = L[0]
  Ly = L[1]
  dx = L[0] / M[0]
  dy = L[1] / M[1]
  N = 3 * np.sqrt(np.pi) * sigma_u / min(dx, dy)
  sigma_u2 = sigma_u**2
  factor_gaussian = 1.0 / (2 * np.pi * sigma_u2)
  x_disp = np.zeros(2)

  # Get vectors in the minimal image representation of size L=(Lx, Ly) and with a corner at (0,0).
  x_PBC = np.empty_like(x)
  for axis in range(2):
    x_PBC[:,axis] = x[:,axis] - (x[:,axis] // L[axis]) * L[axis]

  # Loop over particles
  for n in prange(x.shape[0]):
    kx = int(x_PBC[n,0]  / Lx * M[0])
    ky = int(x_PBC[n,1]  / Ly * M[1])

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
    
  return velocity_particles, 2


@njit(parallel=False, fastmath=True)
def spread(x, force_torque, sigma_u, sigma_w, L, M):
  '''
  Interpolate fluid velocity.
  '''
  fx_mesh = np.zeros((M[0], M[1]))
  fy_mesh = np.zeros((M[0], M[1]))

  # Prepare some variables
  Lx = L[0]
  Ly = L[1]
  dx = L[0] / M[0]
  dy = L[1] / M[1]
  N = 3 * np.sqrt(np.pi) * sigma_u / min(dx, dy)
  sigma_u2 = sigma_u**2
  factor_gaussian = 1.0 / (2 * np.pi * sigma_u2)
  x_disp = np.zeros(2)

  # Get vectors in the minimal image representation of size L=(Lx, Ly) and with a corner at (0,0).
  x_PBC = np.empty_like(x)
  for axis in range(2):
    x_PBC[:,axis] = x[:,axis] - (x[:,axis] // L[axis]) * L[axis]  

  # Loop over particles
  for n in range(x.shape[0]):
    kx = int(x_PBC[n,0] / Lx * M[0])
    ky = int(x_PBC[n,1] / Ly * M[1])

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

  
def solve_Stokes(fx_mesh, fy_mesh, eta, kT, L, M):
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
  vx_Fourier, vy_Fourier = solve_Stokes_Fourier(fx_Fourier, fy_Fourier, gradKx, gradKy, LKx, LKy, expKx, expKy, eta)

  # Transform velocities to real space
  vx_mesh = np.fft.ifft2(vx_Fourier)
  vy_mesh = np.fft.ifft2(vy_Fourier)

  print(' ')
  print('vx_mesh.imag = ', np.linalg.norm(vx_mesh.imag))
  print('vy_mesh.imag = ', np.linalg.norm(vy_mesh.imag))
  print('vx_mesh.real = ', np.linalg.norm(vx_mesh.real))
  print('vy_mesh.real = ', np.linalg.norm(vy_mesh.real))
  print(' ')
  
  return vx_mesh.real, vy_mesh.real


def advance_time_step(dt, scheme, step, x, parameters):
  '''
  Advance time step with integrator self.scheme
  '''
  if scheme == 'deterministic_forward_Euler_no_stresslet':
    return deterministic_forward_Euler_no_stresslet(dt, scheme, step, x, parameters) 
  elif scheme == 'deterministic_forward_Euler':
    return deterministic_forward_Euler(dt, scheme, step, parameters)
  else:
    print('Scheme: ', scheme, ' is not implemented.')
  return


def deterministic_forward_Euler_no_stresslet(dt, scheme, step, x, parameters):
  '''
  Forward Euler scheme without including stresslet.
  '''
  # Get parameters
  eta = parameters.get('eta')  
  M = parameters.get('M_system')
  L = parameters.get('L_system')

  # Compute force between particles
  force_torque = np.zeros((x.shape[0], 3))
  force_torque[0, 0] = 1
  
  # Spread force
  fx_mesh, fy_mesh = spread(x, force_torque, parameters.get('sigma_u'), parameters.get('sigma_w'), L, M)
  print('Fx = ', np.sum(fx_mesh) * L[0] / M[0] * L[1] / M[1])
  print('Fy = ', np.sum(fy_mesh) * L[0] / M[0] * L[1] / M[1])
   
  # Solve Stokes equations
  vx_mesh, vy_mesh = solve_Stokes(fx_mesh, fy_mesh, eta, 0, L, M)
  
  # Interpolate velocity
  #vx_mesh = np.zeros((M[0], M[1]))
  #vy_mesh = np.zeros((M[0], M[1]))
  #vx_mesh[:,:] = fx_mesh
  #vy_mesh[:,:] = fy_mesh
  velocity_particles, strain_rate = interpolate(x, vx_mesh, vy_mesh, parameters.get('sigma_u'), parameters.get('sigma_w'), L, M)
  print('velocity_particles = \n', velocity_particles, '\n\n')

  if parameters.get('plot_velocity_field') == 'True':
    plot_velocity_field(L, M, vx_mesh, vy_mesh, parameters.get('output_name')) 
  
  # Advance particle positions
  x += dt * velocity_particles
  print('x = ', x)

  


