import numpy as np
import sys
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import utilities

# Import local code
import FCM_functions as FCM


if __name__ == '__main__':

  # Input names
  input_file = '../data/run1.inputfile'
  name_config = '../data/run1.config'

  # Read input file
  parameters = FCM.read_input(input_file)
  
  # Read config
  x = utilities.read_config(name_config)
  

  # Set some parameters
  L = parameters.get('L_system')

  # Prepare plot
  fig, ax = plt.subplots()
  plt.gca().set_aspect('equal')
  scat = ax.scatter(x[0,:,0], x[0,:,1], c="b", s=2000, clip_on=False)
  ax.set(xlim=[0, L[0]], ylim=[0, L[1]])

  def update(frame):
    # for each frame, update the data stored on each artist.
    xi = FCM.project_to_periodic_image(np.copy(x[frame,:,0:2]), L)
    xf = xi[:,0]
    yf = xi[:,1]
    
    # update the scatter plot:
    data = np.stack([xf, yf]).T
    scat.set_offsets(data)
    return scat

  ani = animation.FuncAnimation(fig=fig, func=update, frames=x.shape[0], interval=1)
  plt.show()
