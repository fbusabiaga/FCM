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
  MSD_steps = 100
  
  # Read input file
  parameters = FCM.read_input(input_file)
  file_prefix = parameters.get('output_name')
  
  # Read config
  x = utilities.read_config(name_config)

  # Set additional parameters
  dt = parameters.get('dt')
  n_save = parameters.get('n_save')
  dt_sample = dt * n_save

  # Call gr
  name = file_prefix + '.msd.dat'
  msd, msd_std = utilities.msd(x, dt_sample, MSD_steps=MSD_steps, output_name=name)

  # Create two panels
  fig, axes = plt.subplots(1, 1, figsize=(5,5))
  
  # Plot panel one
  axes.errorbar(np.arange(msd.shape[0]) * dt_sample, np.sqrt(msd[:,0]**2 + msd[:,3]**2), yerr=np.sqrt(msd_std[:,0]**2 + msd_std[:,3]**2), linewidth=2, color='r', label=r'$MSD(r)$')

  # Set axes
  axes.set_xlabel(r'$t$', fontsize=fontsize)
  axes.set_ylabel(r'$MSD(t)$', fontsize=fontsize)
  axes.tick_params(axis='both', which='major', labelsize=fontsize)
  axes.yaxis.offsetText.set_fontsize(fontsize)

  # Show legend
  axes.legend(fontsize=fontsize * 2 // 3)

  # Adjust distance between subplots
  fig.tight_layout()
  # fig.subplots_adjust(left=0.13, top=0.95, right=0.9, bottom=0.17, wspace=0.0, hspace=0.0)
  
  # Save to pdf and png
  name = file_prefix + '.msd.pdf'
  plt.savefig(name, format='pdf') 
  plt.show()


  
