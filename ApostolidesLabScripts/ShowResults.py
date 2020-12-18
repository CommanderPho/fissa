## After running run_fissa_pho.py, this can be used to visualize the results.
# output_folder_path_str


# FISSA toolbox
import fissa

from pathlib import Path

# numpy toolbox
import numpy as np

# Plotting toolbox, with notebook embedding options
import holoviews as hv
%load_ext holoviews.ipython
%output widgets='embed'


# def export_experiment_npz(experiment_obj, save_path):
#     print('Saving experiment npz to {}...\n'.format(save_path))
#     np.savez_compressed(save_path, roi_polys=experiment_obj.roi_polys, raw=experiment_obj.raw, result=experiment_obj.result, info=experiment_obj.info,
#                                 means=experiment_obj.means, sep=experiment_obj.sep, deltaf_raw=experiment_obj.deltaf_raw, deltaf_result=experiment_obj.deltaf_result)
#     print('done.')


def export_experiment_npz(experiment_obj, save_parent_path, should_save_separate_npzs=False):
    # Note: Currently skips experiment.raw and experiment.sep, as they are non nparray lists, so they can't be directly saved with np.savez_compressed without conversion
    if should_save_separate_npzs:
        print('Saving separate experiment npz to path {}...\n'.format(save_parent_path))
        print('\t saving npz to {}...\n'.format(save_parent_path.joinpath('experiment_main.npz')))
        np.savez_compressed(save_parent_path.joinpath('experiment_main.npz'), roi_polys=experiment_obj.roi_polys,
                            result=experiment_obj.result, info=experiment_obj.info,
                            means=experiment_obj.means)
        if experiment_obj.deltaf_result is not None:
            print('\t saving npz to {}...\n'.format(save_parent_path.joinpath('experiment_deltaf.npz')))
            np.savez_compressed(save_parent_path.joinpath('experiment_deltaf.npz'), deltaf_raw=experiment_obj.deltaf_raw,
                                deltaf_result=experiment_obj.deltaf_result)
    else:
        print('Saving experiment npz to {}...\n'.format(save_parent_path.joinpath('experiment.npz')))
        if experiment_obj.deltaf_result is not None:
            np.savez_compressed(save_parent_path.joinpath('experiment.npz'), roi_polys=experiment_obj.roi_polys,
                                result=experiment_obj.result, info=experiment_obj.info,
                                means=experiment_obj.means, deltaf_raw=experiment_obj.deltaf_raw,
                                deltaf_result=experiment_obj.deltaf_result)
        else:
            np.savez_compressed(save_parent_path.joinpath('experiment.npz'), roi_polys=experiment_obj.roi_polys,
                                result=experiment_obj.result, info=experiment_obj.info,
                                means=experiment_obj.means)

    print('done.')





# updated_output_folder_path =  root_folder_path.joinpath('fissa_suite2p_updated')
# export_experiment_npz(experiment, updated_output_folder_path, should_save_separate_npzs=False)
# output_file_updated_experiment_path = updated_output_folder_path.joinpath('experiment.npy')


# export_experiment_npz(experiment, output_file_experiment_path)

# np.savez_compressed(output_folder_path.joinpath('experiment_deltaf.npz'), deltaf_raw=experiment.deltaf_raw,
#                         deltaf_result=experiment.deltaf_result)

# np.savez_compressed(output_folder_path.joinpath('experiment.npz'), roi_polys=experiment.roi_polys, result=experiment.result, info=experiment.info,
#                                  means=experiment.means)

# np.savez_compressed(output_folder_path.joinpath('experiment_main.npz'), roi_polys=experiment.roi_polys, raw=experiment.raw, result=experiment.result)

def plot_cell_regions(roi_polys, plot_neuropil=False):
    '''
    Plot a single cell region, using holoviews.
    '''
    out = hv.Overlay()

    if plot_neuropil:
        # Plot the neuropil as well as the ROI
        n_region = len(roi_polys)
    else:
        # Just plot the ROI, not the neuropil
        n_region = 1

    for i_region in range(n_region):
        for part in roi_polys[i_region]:
            x = part[:, 1]
            y = part[:, 0]
            out *= hv.Curve(zip(x, y)).opts(color='w')

    return out

def generate_plots_from_variables(roi_polys, raw, result, means):
    i_trial = 0
    nCell = np.shape(roi_polys)[0]

    # Generate plots for all detected regions
    region_plots = {
        i_cell: plot_cell_regions(roi_polys[i_cell][i_trial])
        for i_cell in range(nCell)
    }

    # Generate plots for raw extracts and neuropil removed
    traces_plots = {
        i_cell: hv.Curve(raw[i_cell][i_trial][0, :], label='suite2p') *
                hv.Curve(result[i_cell][i_trial][0, :], label='FISSA')
        for i_cell in range(nCell)
    }

    # Generate average image
    avg_img = hv.Raster(means[i_trial])

    # Generate overlay plot showing each cell location
    cell_locs = hv.Overlay()
    for c in range(nCell):
        roi_poly = roi_polys[c][i_trial][0][0]
        x = roi_poly[:, 1]
        y = roi_poly[:, 0]
        cell_locs *= hv.Curve(zip(x, y))

    # Render holoviews
    avg_img * cell_locs * hv.HoloMap(region_plots, kdims=['Cell']) + hv.HoloMap(traces_plots, kdims=['Cell'])

def generate_plots(experiment):
    generate_plots_from_variables(experiment.roi_polys, experiment.raw, experiment.result, experiment.means)

def generate_plots_from_loaded_experiment_npz(loaded_npz_data):
    generate_plots_from_variables(loaded_npz_data['roi_polys'], loaded_npz_data['raw'], loaded_npz_data['result'], loaded_npz_data['means'])


if __name__ == '__main__':
    # Specify and load the output data:
    root_folder_path = Path(r'E:\PhoHaleScratchFolder\202001_17-20-24_PassiveStim_Registered\suite2p\plane0')
    images_path = root_folder_path.joinpath('reg_tif')
    output_folder_path =  root_folder_path.joinpath('fissa_suite2p_example')

    output_file_preparation_path = output_folder_path.joinpath('preparation.npy')
    output_file_separated_path = output_folder_path.joinpath('separated.npy')

    output_file_experiment_path = output_folder_path.joinpath('experiment.npy')

    # Load the output data:
    preparation = np.load(output_file_preparation_path, allow_pickle=True)
    separated = np.load(output_file_separated_path, allow_pickle=True)  # experiment output data
    experiment = np.load(output_file_experiment_path, allow_pickle=True)  # experiment output data

    generate_plots(experiment)



