# -*- coding: utf-8 -*-

"""
ms_plotting.py
Methods called by ms2molecules to plot MS output
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from lignin_wrangler.process_ms_input import trim_close_mz_vals, print_high_inten_short
from matplotlib.ticker import FormatStrFormatter
from common_wrangler.common import (create_out_fname, warning)

plt.style.use('seaborn-whitegrid')
NREL_COLORS = ["#282D30", "#00A4E4", "#FFC423", "#8CC63F", "#D9531E", "#9757DF"]


def x_value_warning(data_max_x, default_x_max):
    warning(f"The default maximum x-axis value ({default_x_max}) is less than the maximum x-axis value in the "
            f"data ({data_max_x}). Not all data will be shown.")


def find_pos_plot_limit(data_max_val):
    """
    This method will find a hopefully a reasonable positive integer maximum for an axis value
    Some adaption would be needed for negative numbers; only do so if needed
    :param data_max_val: float, actual maximum value from the data
    :return: float, a larger value estimated to be a reasonable maximum for the axis value
    """
    int_max_val = int(np.ceil(data_max_val))
    max_value_str = str(int_max_val)
    if len(max_value_str) < 3:
        return int_max_val
    power_ten_of_intensity = len(max_value_str) - 1
    first_digit = float(max_value_str[0])
    # could first check
    decimal_add = float(max_value_str[2])/10.
    second_digit = np.ceil(float(max_value_str[1]) + decimal_add)
    order_of_mag = np.power(10., power_ten_of_intensity)
    max_val = first_digit * order_of_mag + second_digit * (order_of_mag / 10)
    return max_val


def make_vlines_plot(title, x_label, y_label, label_data_dict, plot_fname, num_decimals_x_axis,
                     x_max, y_max, loc="best"):
    fig, (ax) = plt.subplots(1, 1, figsize=(6, 4))
    ax.set_xlim([0, x_max])
    ax.set_ylim([0, y_max])
    ax.xaxis.set_major_formatter(FormatStrFormatter(f'%.{num_decimals_x_axis}f'))
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    i = 0
    colors = NREL_COLORS
    while len(colors) < len(label_data_dict):
        colors += NREL_COLORS
    for label, data_array in label_data_dict.items():
        x_array = data_array[:, 0]
        y_array = data_array[:, 1]
        ax.vlines(x_array, [0], y_array, label=label, colors=colors[i])
        i += 1
    if len(label_data_dict) > 1:
        ax.legend(loc=loc)

    plt.savefig(plot_fname, bbox_inches='tight', transparent=True)
    print(f"Wrote file: {os.path.relpath(plot_fname)}")
    plt.close()


def plot_total_intensity_v_ret_time(fname, ms_level, data_array, num_decimals_ret_time_accuracy, out_dir):
    """
    Plot total intensity versus retention times (combines retention times in this method; calls plotting function)
    :param fname: str, name of the file where the data originated
    :param ms_level: str, used to distinguish between different MS output of the same input file (no overwriting)
    :param data_array: ndarray (n x 3) with M/Z, intensity, and retention times
    :param num_decimals_ret_time_accuracy: number of decimal points in retention time accuracy, for rounding
    :param out_dir: None or str, provides location where new file should be saved (None for current directory)
    :return: ndarray, (m x 2), were m is the number of unique retention times, in first column. Second column is
        total intensity for that retention time.
    """
    default_x_max = 16.
    x_index = 2

    # in case not already rounded and sorted...
    data_array[:, x_index] = np.around(data_array[:, x_index], num_decimals_ret_time_accuracy)
    # the intensity and mz order does not matter, only ret time
    data_array = data_array[data_array[:, x_index].argsort()]
    unique_ret_times = np.unique(data_array[:, x_index])
    total_intensities = np.full((len(unique_ret_times)), np.nan)

    for ret_index, ret_time in enumerate(unique_ret_times):
        unique_ret_time_data_array = data_array[data_array[:, x_index] == ret_time]
        total_intensities[ret_index] = np.sum(unique_ret_time_data_array[:, 1])

    data_max_x = np.max(unique_ret_times)
    min_y_max = np.max(total_intensities)
    if data_max_x > default_x_max:
        x_value_warning(data_max_x, default_x_max)
    y_max = find_pos_plot_limit(min_y_max)

    title = f"Total Intensity Plot"
    x_label = "Retention time (min)"
    y_label = "Total intensity (unscaled)"
    suffix = "_tot_int"
    if "_ms" not in fname.lower():
        suffix = f"_ms{ms_level}" + suffix
    plot_fname = create_out_fname(fname, suffix=suffix, ext='png', base_dir=out_dir)

    # # Uncomment below if want both vlines and not
    # make_fig(plot_fname, unique_ret_times, total_intensities, x_label=x_label, y_label=y_label,
    #          loc=0, title=title)
    # print(f"Wrote file: {os.path.relpath(plot_fname)}")
    # plot_fname = create_out_fname(base_fname, suffix="_tot_int_vlines", ext='png', base_dir=out_dir)

    ret_time_tot_intensity_array = np.column_stack((unique_ret_times, total_intensities))
    make_vlines_plot(title, x_label, y_label, {"total_intensities": ret_time_tot_intensity_array},
                     plot_fname, num_decimals_ret_time_accuracy, default_x_max, y_max, loc="upper left")
    return ret_time_tot_intensity_array


def plot_select_mz_intensity_v_ret_time(fname, ms_level, mz_list_to_plot, data_array, num_decimals_ms_accuracy,
                                        num_decimals_ret_time_accuracy, out_dir):
    """
    Plot total intensity versus retention times (combines retention times in this method; calls plotting function)
    :param fname: str, name of the file where the data originated
    :param ms_level: str, used to distinguish between different MS output of the same input file (no overwriting)
    :param data_array: ndarray (n x 3) with M/Z, intensity, and retention times
    :param mz_list_to_plot: list, with up to 5 mz values to plot vs time on the same plot
    :param num_decimals_ms_accuracy: int, number of decimal points in MS accuracy, for rounding
    :param num_decimals_ret_time_accuracy: number of decimal points in retention time accuracy, for rounding
    :param out_dir: None or str, provides location where new file should be saved (None for current directory)
    :return: ndarray, (m x 2), were m is the number of unique retention times, in first column. Second column is
        total intensity for that retention time.
    """
    default_x_max = 16.
    data_x_max = 0.
    x_index = 2

    if len(mz_list_to_plot) > 5:
        warning("Error while attempting to plot select M/Z values versus retention times.\n    This "
                "method expects at most 5 M/Z values to display on one plot. This plot will not be produced.")
        return
    if len(mz_list_to_plot) == 0:
        warning("Error while attempting to plot select M/Z values versus retention times.\n    No "
                "M/Z values provided. This plot will not be produced.")
        return
    if len(mz_list_to_plot) == 1:
        title = f"Intensity versus Retention Time for M/Z={mz_list_to_plot[0]}"
    else:
        title = "Intensity versus Retention Time for Selected M/Z Values"
    # At least sometimes, mz_list_to_plot and data_array are not already rounded, so doing so here
    mz_list_to_plot = np.around(mz_list_to_plot, num_decimals_ms_accuracy)
    data_array[:, 0] = np.around(data_array[:, 0], num_decimals_ms_accuracy)
    # wait to check for max retention time (in case it does not apply to chosen mz values, but not intensity, to have
    #    more consistent y-axis ranges
    max_intensity = np.max(data_array[:, 1])
    y_max = find_pos_plot_limit(max_intensity)

    inten_time_dict = {}
    for mz_val in mz_list_to_plot:
        sub_data_array = data_array[data_array[:, 0] == mz_val]
        if len(sub_data_array) < 1:
            warning(f"No retention time data found for M/Z value {mz_val} from {os.path.relpath(fname)}.\n    This "
                    f"M/Z will be omitted from the plot.")
        else:
            curve_label = f"{mz_val:.{num_decimals_ms_accuracy}f}"
            # make this x, y, so ret_time, intensity
            inten_time_dict[curve_label] = np.column_stack((sub_data_array[:, x_index], sub_data_array[:, 1]))
            sub_array_max_x = np.max(sub_data_array[:, x_index])
            if sub_array_max_x > data_x_max:
                data_x_max = sub_array_max_x

    if data_x_max > default_x_max:
        warning(f"The default maximum x-axis value ({default_x_max}) is less than the maximum x-axis value in the "
                f"data ({data_x_max}). Not all data will be shown.")
    x_label = "Retention time (min)"
    y_label = "Intensity (unscaled)"
    suffix = "_int_v_time"
    if "_ms" not in fname.lower():
        suffix = f"_ms{ms_level}" + suffix
    plot_fname = create_out_fname(fname, suffix=suffix, ext='png', base_dir=out_dir)
    make_vlines_plot(title, x_label, y_label, inten_time_dict, plot_fname, num_decimals_ret_time_accuracy,
                     default_x_max, y_max, loc="upper left")

    # Maybe later... would need to re-slice data
    # inten_time_dict = defaultdict(lambda: None)
    # y_val_dict = defaultdict(lambda: None)
    # curve_label = defaultdict(lambda: "")
    # mz_counter = 0
    # make_fig(plot_fname + "_make_fig",
    #          x_array=inten_time_dict[0], y1_array=y_val_dict[0], y1_label=curve_label[0], color1=NREL_COLORS[1],
    #          x2_array=inten_time_dict[1], y2_array=inten_time_dict[1], y2_label=curve_label[1], color2=NREL_COLORS[2],
    #          x3_array=inten_time_dict[2], y3_array=inten_time_dict[2], y3_label=curve_label[2], color3=NREL_COLORS[3],
    #          x4_array=inten_time_dict[3], y4_array=inten_time_dict[3], y4_label=curve_label[3], color4=NREL_COLORS[4],
    #          x5_array=inten_time_dict[4], y5_array=inten_time_dict[4], y5_label=curve_label[4], color5=NREL_COLORS[5],
    #          x_label=x_label, y_label=y_label, loc=0, title=title)
    return inten_time_dict


def plot_mz_v_intensity(fname, data_array_dict, num_decimals_ms_accuracy, out_dir):
    """
    Plot m/z v intensities for all entries in the data_dict_array (key is ms_level)
    :param fname: str, name of the file where the data originated
    :param data_array_dict: dict, str (label): ndarray (n x 3) with M/Z, intensity, and retention times
    :param num_decimals_ms_accuracy: int, number of decimal points in MS accuracy, for rounding
    :param out_dir: None or str, provides location where new file should be saved (None for current directory)
    :return: ndarray, (m x 2), were m is the number of unique retention times, in first column. Second column is
        total intensity for that retention time.
    """
    labels = list(data_array_dict.keys())
    first_label = labels[0]
    lower_fname = fname.lower()
    if isinstance(first_label, str):
        if "ms" in first_label:
            level = first_label
        else:
            level = f"ms{first_label}"
    else:
        # assumes numeric if the MS2, and the level is the ionization energy; only include the level if single level
        level = "ms2"
        if len(labels) == 1:
            ion_energy = f"hcd{first_label}"
            if ion_energy not in lower_fname:
                if level in lower_fname:
                    level = ion_energy
                else:
                    level += "_" + ion_energy
    title = f"M/Z versus Intensity from {level.upper()} Data"
    suffix = "_mz_v_int"
    if level not in lower_fname:
        suffix = "_" + level + suffix
    plot_fname = create_out_fname(fname, suffix=suffix, ext='png', base_dir=out_dir)
    default_x_max = 1000.
    data_max_x = 0.
    data_max_y = 0.

    for data_array in data_array_dict.values():
        current_max_x = np.max(data_array[:, 0])
        if current_max_x > data_max_x:
            data_max_x = current_max_x

        current_max_y = np.max(data_array[:, 1])
        if current_max_y > data_max_y:
            data_max_y = current_max_y

    if data_max_x > default_x_max:
        x_value_warning(data_max_x, default_x_max)

    y_max = find_pos_plot_limit(data_max_y)

    make_vlines_plot(title, "M/Z Values", "Intensity (unscaled)", data_array_dict, plot_fname,
                     num_decimals_ms_accuracy, default_x_max, y_max)


def initial_output(fname, fname_lower, ms_array, ms_level, max_unique_mz_to_collect, max_num_mz_in_stdout, threshold,
                   num_decimals_ms_accuracy, ret_time_accuracy, num_decimals_ret_time_accuracy, out_dir,
                   quit_after_mzml_to_csv, direct_injection=False):
    final_str = ''
    if "sorted" in fname_lower or "clean" in fname_lower and not direct_injection:
        # This means the data is already processed, so can run as is
        return ms_array

    if ms_array.shape[0] < 2:
        trimmed_mz_array = ms_array
    else:
        final_str += f"\nProcessing MS Level {ms_level.replace('_', ', ').replace('p', '.')} data"
        print(f"\nProcessing MS Level {ms_level.replace('_', ', ').replace('p', '.')} data")
        trimmed_mz_array = trim_close_mz_vals(ms_array, num_decimals_ms_accuracy, threshold,
                                              ret_time_accuracy, max_num_output_mzs=max_unique_mz_to_collect)
        if quit_after_mzml_to_csv:
            print_high_inten_short(ms_level, trimmed_mz_array, max_output_mzs=max_num_mz_in_stdout)

    # check for a vector, to be treated differently
    if len(trimmed_mz_array.shape) == 1:
        final_str += f"Skipping plotting for MS{ms_level} since there is only one M/Z value after clean-up."
        print(f"Skipping plotting for MS{ms_level} since there is only one M/Z value after clean-up.")
    else:
        if direct_injection or np.isnan(trimmed_mz_array[-1][2]):
            plot_mz_v_intensity(fname, {ms_level: ms_array}, num_decimals_ms_accuracy, out_dir)
        else:
            # produce intensity vs retention plots only if there is retention data
            if not np.isnan(trimmed_mz_array[0][2]):
                # if desired, can return save the raw plotting data by grabbing the return, ndarray, (m x 2)
                plot_total_intensity_v_ret_time(fname, ms_level, ms_array, num_decimals_ret_time_accuracy, out_dir)
                plot_select_mz_intensity_v_ret_time(fname, ms_level, trimmed_mz_array[:, 0], ms_array,
                                                    num_decimals_ms_accuracy, num_decimals_ret_time_accuracy, out_dir)
    return trimmed_mz_array, final_str
