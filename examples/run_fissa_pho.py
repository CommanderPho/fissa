"""
Basic FISSA usage example.

This file contains a step-by-step example workflow for using the FISSA toolbox.

An example notebook is provided here:
https://github.com/rochefort-lab/fissa/blob/master/examples/Basic%20usage.ipynb
"""

import fissa
# suite2p toolbox
import suite2p.run_s2p

# numpy toolbox
import numpy as np
from pathlib import Path

# Custom Datahandler import:
import datahandler_pho_suite2p as active_custom_datahandler_obj


# On Windows, it is necessary to wrap the script in a __name__ check, so
# that multiprocessing works correctly. Multiprocessing is triggered by the
# experiment.separate() step.
if __name__ == '__main__':
    # Define the data to extract
    root_folder_path = Path(r'E:\PhoHaleScratchFolder\202001_17-20-24_PassiveStim_Registered\suite2p\plane0')
    images_path = root_folder_path.joinpath('reg_tif')
    output_folder_path =  root_folder_path.joinpath('fissa_suite2p_example')

    images_path_str = str(images_path.resolve())
    output_folder_path_str = str(output_folder_path.resolve())

    # Load the detected regions of interest
    stat = np.load(root_folder_path.joinpath('stat.npy'), allow_pickle=True)  # cell stats
    ops = np.load(root_folder_path.joinpath('ops.npy'), allow_pickle=True).item()
    iscell = np.load(root_folder_path.joinpath('iscell.npy'), allow_pickle=True)[:, 0]

    # Get image size
    Lx = ops['Lx']
    Ly = ops['Ly']

    # Get the cell ids
    ncells = len(stat)
    cell_ids = np.arange(ncells)  # assign each cell an ID, starting from 0.
    cell_ids = cell_ids[iscell==1]  # only take the ROIs that are actually cells.
    num_rois = len(cell_ids)

    print('ncells: {}\nnum_rois: {}\n\ncell_ids: {}\n'.format(ncells, num_rois, cell_ids))
    print('image size: {}x{}\n'.format(Lx, Ly))

    # Generate ROI masks in a format usable by FISSA (in this case, a list of masks)
    rois = [np.zeros((Ly, Lx), dtype=bool) for n in range(num_rois)]

    for i, n in enumerate(cell_ids):
        # i is the position in cell_ids, and n is the actual cell number
        ypix = stat[n]['ypix'][~stat[n]['overlap']]
        xpix = stat[n]['xpix'][~stat[n]['overlap']]
        rois[i][ypix, xpix] = 1
    
#     rois = 'exampleData/20150429.zip'
#     images = 'exampleData/20150529'

    # Define the name of the experiment extraction location

    # Make sure you use a different output path for each experiment you run.
    #     output_dir = root_folder_path.joinpath('suite2p/plane0/fissa_example')
    

    # Instantiate a fissa experiment object
    #     experiment = fissa.Experiment(images, rois, output_dir)
    # experiment = fissa.Experiment(str(images.resolve()), [rois[:ncells]], output_dir)


    # exp = fissa.Experiment(images_path, [rois[:ncells]], output_folder) # Wrong shape
    experiment = fissa.Experiment(images_path_str, [rois], output_folder_path_str, datahandler_custom = active_custom_datahandler_obj, ncores_preparation = 1) # TypeError: Wrong ROIs input format: expected a list or sequence, but got a <class 'numpy.ndarray'>



    # Run the FISSA separation algorithm
    experiment.separate()

    # Export to a .mat file which can be opened with MATLAB (optional)
    experiment.save_to_matlab()
