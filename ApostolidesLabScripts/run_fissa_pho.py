"""
Basic FISSA Neuropil Removal Script.

Pho Hale, Pierre Lab 12-17-2020

"""

import pickle # For persisting outputs
import fissa
# suite2p toolbox
import suite2p.run_s2p

# numpy toolbox
import numpy as np
from pathlib import Path

# Custom Datahandler import:
import datahandler_pho_suite2p as active_custom_datahandler_obj
import ShowResults as pho_show_results
import pho_temp_test as pho_tt

# out_masks = pho_tt.reformat_polygons_to_masks(experiment.roi_polys)
# def load_experiment_object(path):
#     filehandler = open(path, 'r')
#     object = pickle.load(filehandler)
#     return object
#
# def save_experiment_object(exp, path):
#     file_out = open(path, 'w')
#     pickle.dump(exp, file_out)

def load_suite2p_results(root_folder_path):
    print(
        'Loading Suite2p Output Results from:\n \t\t {}\n \t\t {}\n \t\t {}\n\n'.format(
            root_folder_path.joinpath('stat.npy'), root_folder_path.joinpath('ops.npy'),
            root_folder_path.joinpath('iscell.npy')))

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

    return rois


def pho_post_load(experiment, output_folder_path):
    out_masks = pho_tt.reformat_polygons_to_masks(experiment.roi_polys)
    pho_tt.export_roi_masks(experiment.roi_polys, out_masks, output_folder_path, custom_filename_prefix='experiment')


# On Windows, it is necessary to wrap the script in a __name__ check, so
# that multiprocessing works correctly. Multiprocessing is triggered by the
# experiment.separate() step.
if __name__ == '__main__':
    # Define the data to extract
    root_folder_path = Path(r'E:\PhoHaleScratchFolder\202001_17-20-24_PassiveStim_Registered\suite2p\plane0')
    imaging_frequency_Hz = 30
    rois = load_suite2p_results(root_folder_path)

    images_path = root_folder_path.joinpath('reg_tif')
    output_folder_path =  root_folder_path.joinpath('fissa_suite2p_example')
    # output_file_experiment_path = output_folder_path.joinpath('experiment.npy')
    # output_file_experiment_path = output_folder_path.joinpath('experiment.obj')
    # output_file_experiment_path = output_folder_path.joinpath('experiment.npz')

    images_path_str = str(images_path.resolve())
    output_folder_path_str = str(output_folder_path.resolve())

    # Instantiate a fissa experiment object
    # experiment = fissa.Experiment(images_path_str, [rois], output_folder_path_str, datahandler_custom = active_custom_datahandler_obj, ncores_preparation = 1)
    experiment = fissa.Experiment(images_path_str, [rois], output_folder_path_str,
                                  datahandler_custom=active_custom_datahandler_obj)

    # Run the FISSA separation algorithm
    print('Starting FISSA separation processing. This may take several hours...\n \t Results will be written to:\n \t\t {}\n \t\t {}\n\n'.format(output_folder_path.joinpath('matlab.mat'), output_folder_path.joinpath('preparation.npy'), output_folder_path.joinpath('separated.npy')))
    experiment.separate()

    print('Calculating deltaf (this should take a few minutes)...\n')
    experiment.calc_deltaf(imaging_frequency_Hz)

    # Export to a .mat file which can be opened with MATLAB (optional)
    print('Saving separation results as .mat file at {}\n'.format(output_folder_path.joinpath('matlab.mat')))
    experiment.save_to_matlab()

    pho_post_load(experiment, output_folder_path)

    # print('Saving full experiment results out to disk at {}\n'.format(output_file_experiment_path))
    # # Save the full experiment out to disk:
    # np.save(output_file_experiment_path, experiment)


    print('Saving updated .mat file results at {}\n'.format(output_folder_path.joinpath('matlab_deltaf.mat')))
    # experiment.save_to_matlab('matlab_deltaf.mat')



    # out_fig = pho_show_results.generate_plots(experiment)
    # save_experiment_object(experiment, output_file_experiment_path)