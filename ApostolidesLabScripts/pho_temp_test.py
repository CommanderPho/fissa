# import pho_temp_test as active_custom_test
# FISSA toolbox
import fissa

from pathlib import Path

# numpy toolbox
import numpy as np



def printExperiment(exp):
    print('experiment: {}'.format(exp))


def reformat_dict_for_npz(orig_dict):
    element_shape = np.shape(orig_dict[0][0])
    nCell = len(orig_dict)
    nTrials = len(orig_dict[0])
    out_npz_array = np.empty([nCell, nTrials, element_shape[0], element_shape[1]])
    # loop over cells and trial
    for cell in range(nCell):
        for trial in range(nTrials):
            curr_element_shape = np.shape(orig_dict[cell][trial])
            if (curr_element_shape[1] < element_shape[1]):
                # needed_padding = ((element_shape[0] - curr_element_shape[0]), (element_shape[1] - curr_element_shape[1]))
                needed_padding = [(0, (element_shape[0] - curr_element_shape[0])),
                                  (0, (element_shape[1] - curr_element_shape[1]))]
                # needed_padding = (, (element_shape[1] - curr_element_shape[1])
                # print('curr_element_shape: {}\n needed_padding: {}\n'.format(curr_element_shape, needed_padding))
                temp_padded_array = np.pad(orig_dict[cell][trial], needed_padding, mode='constant',
                                           constant_values=(-1, -1))
                # print('temp_padded_array shape: {}\n'.format(np.shape(temp_padded_array)))
                out_npz_array[cell][trial] = temp_padded_array

            else:
                out_npz_array[cell][trial] = orig_dict[cell][trial]

    return out_npz_array


def export_difficult_variables(experiment_obj, save_parent_path, custom_filename_prefix='experiment',
                               save_compressed=True):
    # Note: Currently skips experiment.raw and experiment.sep, as they are non nparray lists, so they can't be directly saved with np.savez_compressed without conversion
    raw_np = reformat_dict_for_npz(experiment_obj.raw)
    sep_np = reformat_dict_for_npz(experiment_obj.sep)
    print('\t saving npz containing the difficult variables (raw, sep) to {}...\n'.format(
        save_parent_path.joinpath('{}_difficult.npz'.format(custom_filename_prefix))))
    if save_compressed:
        np.savez_compressed(save_parent_path.joinpath('{}_difficult.npz'.format(custom_filename_prefix)),
                            raw=raw_np,
                            sep=sep_np)
    else:
        np.savez(save_parent_path.joinpath('{}_difficult.npz'.format(custom_filename_prefix)),
                 raw=raw_np,
                 sep=sep_np)


def export_experiment_npz_uncompressed(experiment_obj, save_parent_path, custom_filename_prefix='experiment',
                                       should_save_separate_npzs=False):
    # Note: Currently skips experiment.raw and experiment.sep, as they are non nparray lists, so they can't be directly saved with np.savez_compressed without conversion
    if should_save_separate_npzs:
        print('Saving separate experiment npz to path {} with prefix {}...\n'.format(save_parent_path,
                                                                                     custom_filename_prefix))
        print(
            '\t saving npz to {}...\n'.format(save_parent_path.joinpath('{}_main.npz'.format(custom_filename_prefix))))
        np.savez(save_parent_path.joinpath('{}_main.npz'.format(custom_filename_prefix)),
                 roi_polys=experiment_obj.roi_polys,
                 result=experiment_obj.result, info=experiment_obj.info,
                 means=experiment_obj.means)
        if ((getattr(experiment_obj, 'deltaf_raw', None) is not None) and (
                getattr(experiment_obj, 'deltaf_result', None) is not None)):
            print('\t saving npz to {}...\n'.format(
                save_parent_path.joinpath('{}_deltaf.npz'.format(custom_filename_prefix))))
            np.savez(save_parent_path.joinpath('{}_deltaf.npz'.format(custom_filename_prefix)),
                     deltaf_raw=experiment_obj.deltaf_raw,
                     deltaf_result=experiment_obj.deltaf_result)

    else:
        print('Saving experiment npz to {}...\n'.format(
            save_parent_path.joinpath('{}.npz'.format(custom_filename_prefix))))
        if ((getattr(experiment_obj, 'deltaf_raw', None) is not None) and (
                getattr(experiment_obj, 'deltaf_result', None) is not None)):

            np.savez(save_parent_path.joinpath('{}.npz'.format(custom_filename_prefix)),
                     roi_polys=experiment_obj.roi_polys,
                     result=experiment_obj.result, info=experiment_obj.info,
                     means=experiment_obj.means, deltaf_raw=experiment_obj.deltaf_raw,
                     deltaf_result=experiment_obj.deltaf_result)


        else:
            np.savez(save_parent_path.joinpath('{}.npz'.format(custom_filename_prefix)),
                     roi_polys=experiment_obj.roi_polys,
                     result=experiment_obj.result, info=experiment_obj.info,
                     means=experiment_obj.means)

    export_difficult_variables(experiment_obj, save_parent_path, custom_filename_prefix=custom_filename_prefix,
                               save_compressed=False)


def export_experiment_npz(experiment_obj, save_parent_path, custom_filename_prefix='experiment',
                          should_save_separate_npzs=False):
    # Note: Currently skips experiment.raw and experiment.sep, as they are non nparray lists, so they can't be directly saved with np.savez_compressed without conversion
    if should_save_separate_npzs:
        print('Saving separate experiment npz to path {} with prefix {}...\n'.format(save_parent_path,
                                                                                     custom_filename_prefix))
        print(
            '\t saving npz to {}...\n'.format(save_parent_path.joinpath('{}_main.npz'.format(custom_filename_prefix))))
        np.savez_compressed(save_parent_path.joinpath('{}_main.npz'.format(custom_filename_prefix)),
                            roi_polys=experiment_obj.roi_polys,
                            result=experiment_obj.result, info=experiment_obj.info,
                            means=experiment_obj.means)
        if ((getattr(experiment_obj, 'deltaf_raw', None) is not None) and (
                getattr(experiment_obj, 'deltaf_result', None) is not None)):
            print('\t saving npz to {}...\n'.format(
                save_parent_path.joinpath('{}_deltaf.npz'.format(custom_filename_prefix))))
            np.savez_compressed(save_parent_path.joinpath('{}_deltaf.npz'.format(custom_filename_prefix)),
                                deltaf_raw=experiment_obj.deltaf_raw,
                                deltaf_result=experiment_obj.deltaf_result)

    else:
        print('Saving experiment npz to {}...\n'.format(
            save_parent_path.joinpath('{}.npz'.format(custom_filename_prefix))))
        if ((getattr(experiment_obj, 'deltaf_raw', None) is not None) and (
                getattr(experiment_obj, 'deltaf_result', None) is not None)):

            np.savez_compressed(save_parent_path.joinpath('{}.npz'.format(custom_filename_prefix)),
                                roi_polys=experiment_obj.roi_polys,
                                result=experiment_obj.result, info=experiment_obj.info,
                                means=experiment_obj.means, deltaf_raw=experiment_obj.deltaf_raw,
                                deltaf_result=experiment_obj.deltaf_result)

        else:
            np.savez_compressed(save_parent_path.joinpath('{}.npz'.format(custom_filename_prefix)),
                                roi_polys=experiment_obj.roi_polys,
                                result=experiment_obj.result, info=experiment_obj.info,
                                means=experiment_obj.means)

    export_difficult_variables(experiment_obj, save_parent_path, custom_filename_prefix=custom_filename_prefix,
                               save_compressed=True)


def export_roi_masks(roi_polys, roi_masks, save_parent_path, custom_filename_prefix='experiment'):
    np.savez_compressed(save_parent_path.joinpath('{}_roi_masks.npz'.format(custom_filename_prefix)),
                        roi_polys=roi_polys,
                        roi_masks=roi_masks)


def import_experiment_npzs(save_directory):
    output_file_experiment_path = save_directory.joinpath('experiment.npy')
    output_file_difficult_experiment_path = save_directory.joinpath('experiment_difficult.npy')

    main = np.load(output_file_experiment_path, allow_pickle=True)
    difficult = np.load(output_file_difficult_experiment_path, allow_pickle=True)

    main_fields = ['roi_polys','result','info','means']
    dff_fields = ['deltaf_raw','deltaf_result']
    difficult_fields = ['raw','sep']

    all_fields = main_fields + dff_fields + difficult_fields
    output_dict = {**main, **difficult}

    return output_dict

def getmasks(rois, shpe):
    '''Get the masks for the specified rois.

    Parameters
    ----------
    rois : list
        list of roi coordinates. Each roi coordinate should be a 2d-array
        or equivalent list. I.e.:
        roi = [[0,0], [0,1], [1,1], [1,0]]
        or
        roi = np.array([[0,0], [0,1], [1,1], [1,0]])
        I.e. a n by 2 array, where n is the number of coordinates.
        If a 2 by n array is given, this will be transposed.
    shpe : array/list
        shape of underlying image [width,height]

    Returns
    -------
    list of numpy.ndarray
        List of masks for each roi in the rois list
    '''
    from fissa.ROI import poly2mask
    # get number of rois
    nrois = len(rois)

    # start empty mask list
    masks = [''] * nrois

    for i in range(nrois):
        # transpose if array of 2 by n
        if np.asarray(rois[i]).shape[0] == 2:
            rois[i] = np.asarray(rois[i]).T

        # transform current roi to mask
        mask = poly2mask(rois[i], shpe)
        # store in list
        masks[i] = np.array(mask[0].todense())

    return masks

def reformat_polygons_to_masks(polys):
    output_masks = getmasks(polys, (512, 512))
    return output_masks


if __name__ == '__main__':
    export_experiment_npz(experiment, save_path, should_save_separate_npzs=False)

    # export_roi_masks(experiment.roi_polys, out_masks, save_parent_path, custom_filename_prefix='experiment')

    print('done.')