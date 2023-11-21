import numpy as np
import sys
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import utilities

# Import local code
import FCM_functions as FCM

import matplotlib.pyplot as plt
import matplotlib as mpl

plt.rcParams.update({
  "text.usetex": True,
  "font.family": "DeJavu serif",
  "font.serif": ["Times New Roman"],
  "text.latex.preamble": r"\usepackage{amsmath}",
  "text.latex.preamble": r"\usepackage{siunitx}"
})

if True:   
  # Set some global options
  fontsize = 22
  mpl.rcParams['axes.linewidth'] = 1.5
  mpl.rcParams['xtick.direction'] = 'in'
  mpl.rcParams['xtick.major.size'] = 4
  mpl.rcParams['xtick.major.width'] = 1.5
  mpl.rcParams['xtick.minor.size'] = 4
  mpl.rcParams['xtick.minor.width'] = 1
  mpl.rcParams['ytick.direction'] = 'in'
  mpl.rcParams['ytick.major.size'] = 4
  mpl.rcParams['ytick.major.width'] = 1.5
  mpl.rcParams['ytick.minor.size'] = 4
  mpl.rcParams['ytick.minor.width'] = 1

  
if __name__ == '__main__':

  # Input names
  input_file = '../data/run1.inputfile'
  name_config = '../data/run1.config'

  # Set parameters
  num_frames_skip = 0
  num_frames = 1000
  nbins = 100
  dim = '2d'

  # Read input file
  parameters = FCM.read_input(input_file)
  file_prefix = parameters.get('output_name')
  rcut = min(parameters.get('L_system')) / 2.0
  
  # Read config
  x = utilities.read_config(name_config)

  # Set additional parameters
  L = parameters.get('L_system')

  # Get number of particles
  N = x.shape[1]

  # Call gr
  name = file_prefix + '.gr.dat'
  gr = utilities.radial_distribution_function(x[num_frames_skip:], num_frames, rcut=rcut, nbins=nbins, r_vectors=None, L=L, name=name)

  # Create two panels
  fig, axes = plt.subplots(1, 1, figsize=(5,5))
      
  # Plot panel one
  axes.plot(gr[:,0], gr[:,1], linewidth=2, color='r', label=r'$g(r)$')

  # Set axes
  axes.set_xlabel(r'$r$', fontsize=fontsize)
  axes.set_ylabel(r'$g(r)$', fontsize=fontsize)
  axes.tick_params(axis='both', which='major', labelsize=fontsize)
  axes.yaxis.offsetText.set_fontsize(fontsize)

  # Show legend
  axes.legend(fontsize=fontsize * 2 // 3)

  # Adjust distance between subplots
  fig.tight_layout()
  # fig.subplots_adjust(left=0.13, top=0.95, right=0.9, bottom=0.17, wspace=0.0, hspace=0.0)
  
  # Save to pdf and png
  name = file_prefix + '.gr.pdf'
  plt.savefig(name, format='pdf') 
  plt.show()


  
