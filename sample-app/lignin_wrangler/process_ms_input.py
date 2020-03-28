# -*- coding: utf-8 -*-

"""
ms_input_process.py
Methods called by ms2molecules to read and clean up (e.g. remove low intensity peaks) ms input
"""
import os
import re
import warnings
from collections import OrderedDict
import pymzml
import xml
import numpy as np
from common_wrangler.common import (read_csv_header, warning, InvalidDataError, create_out_fname, get_fname_root,
                                    make_dir, str_to_file, write_csv, quote, dequote, round_to_fraction)
from lignin_wrangler.match_formulas import calc_accuracy_ppm
from lignin_wrangler.create_library import make_image_grid
from lignin_wrangler.lignin_common import (CSV_RET_HEADER, TYPICAL_CSV_HEADER, MZML_EXT, RET_TIME_CSV_HEADER,
                                           DEF_SUFFIX, DEF_LONG_SUFFIX, MATCH_STR_FMT, REL_INTENSITY,
                                           SHORT_OUTPUT_HEADERS, FORMULA_SMI_DICT, OUTPUT_HEADERS,
                                           MATCH_STR_HEADER, MZ_STR_HEADER, MZ_STR_FMT, MIN_ERR_MW, MIN_ERR,
                                           MIN_ERR_FORMULA, MIN_ERR_DBE, MIN_ERR_MATCH_TYPE, M_Z, INTENSITY, RET_TIME,
                                           CALC_MW, PPM_ERR, PARENT_FORMULA, DBE, MATCH_TYPE, STRUCT_DIR)


def check_input_csv_header(fname):
    """
    Checks first line of specified for expected header
    :param fname: str, the location of the file to check the header
    :return: num_header_lines, int: 1 by default; 0 if it appears that the header is missing
    """
    num_header_lines = 1
    potential_header = read_csv_header(fname)
    base_fname = os.path.relpath(fname)
    if potential_header is None:
        raise InvalidDataError(f"Input file may be blank: {base_fname}")
    while potential_header[0].startswith("#"):
        with open(fname) as f:
            for row in f:
                if row.startswith("#"):
                    num_header_lines += 1
                else:
                    potential_header = row.strip().split(",")
                    potential_header = [dequote(x) for x in potential_header]
                    break
    if potential_header != TYPICAL_CSV_HEADER and potential_header != CSV_RET_HEADER:
        try:
            # Still move on to reading values, but first check if there may not be a header
            if len(potential_header) > 1:
                # if right into values (that is, no trouble converting to float), continue to reading values
                float(potential_header[0])
                float(potential_header[1])
                num_header_lines = 0
                warning(f"No header found in file: {base_fname}\n    Will attempt to read data as M/Z and intensity.")
            else:
                raise ValueError
        except ValueError:
            # check that the difference is not a trivial difference in case
            if (len(potential_header) in [2, 3]) and (potential_header[0].lower() == TYPICAL_CSV_HEADER[0].lower()) \
                    and (potential_header[1].lower() == TYPICAL_CSV_HEADER[1].lower()):
                pass
            else:
                warning(f"While reading file: {base_fname}\n    Did not find the expected headers "
                        f"'{TYPICAL_CSV_HEADER}', but '{potential_header}'\n Will attempt to read data as M/Z, "
                        f"intensity, and, if there is a third column, retention time (in min).")
    return num_header_lines


def read_validate_csv_data(fname, fname_lower, ms_accuracy, direct_injection):
    """
    Read given filename for data, and check that data is in the expected format
    :param fname: str, location of file to read and validate its data
    :param fname_lower: str, just the base name of the file, in lowercase
    :param direct_injection: boolean, if True and there is retention data, peaks from different retention times will
        be combined
    :param ms_accuracy: float, the accuracy for the difference in M/Z values to consider them identical
    :return: data_array, a numpy array with required dimensions (3 columns), sorted by final then first column
    """
    ms_level_match = re.search(r'ms(\d+)', fname_lower)
    if ms_level_match:
        ms_level = str(ms_level_match.group(1))
        if ms_level == "2":
            mz_match = re.search(r'_mz(\d+)', fname_lower)
            if mz_match:
                # now check for a decimal following the last matched; for fnames, "p" is used instead period in floats
                ms_level += mz_match.group(0)
                decimal_match = re.search(rf'{mz_match.group(0)}p(\d+)', fname_lower)
                if decimal_match:
                    ms_level += f"p{decimal_match.group(1)}"
    else:
        # if not indicated by title, assume MS1 for now
        ms_level = "1"
        print(f"Assuming MS1 for file: {os.path.relpath(fname)}")

    num_header_lines = check_input_csv_header(fname)

    data_array = np.genfromtxt(fname, dtype=np.float, delimiter=',', skip_header=num_header_lines)

    # check_input_csv_header does not prevent files without correct number of columns from getting here,
    #     So start by checking that. Shape will throw an error if a vector, so check that first
    if len(data_array.shape) == 1:
        num_cols = 1
    else:
        num_cols = data_array.shape[1]
    if num_cols > 3 or num_cols < 2:
        raise InvalidDataError(f"Error while reading file: {fname}"
                               f"\n    This program expects each file to contain two or three comma-separated "
                               f"columns of data:\n        {TYPICAL_CSV_HEADER} (and optionally) "
                               f"{RET_TIME_CSV_HEADER}\n"
                               f"    However, the data in this file has {num_cols} column(s). Program exiting.")

    # Clean up for all files first
    # Get rid of any rows with nan,
    # in order to get rid of invalid rows (with nan, like ,,, rows), first need to git rid of any intentionally
    #     nan columns (specifically the retention time column); we'll replace the ret_time column if needed
    data_array = data_array[:, ~np.all(np.isnan(data_array), axis=0)]
    # now, remove rows that are nan (as happens with some csv output),
    data_array = data_array[~np.isnan(data_array).any(axis=1)]
    # and any zero intensity (will speed up other possible processes below
    data_array = data_array[data_array[:, 1] > 0]

    # can't use old num_col variable, as the shape may have changed above
    if data_array.shape[1] == 2:
        # two-column CSVs do not have retention time; we'll add a column with nan so the array matches those from
        #     mzML files, which do contain retention time data. If there are three columns, assume retention time
        #     is the third, and no analysis of that data (e.g. check for repeats with direct injection) is needed
        data_array = np.insert(data_array, 2, np.nan, axis=1)

    if direct_injection:
        if not np.isnan(data_array[0][-1]):
            data_array[:, 2] = np.nan
        data_array = avg_duplicate_mz_intensities(data_array, ms_accuracy)

    return OrderedDict({ms_level: data_array})


def process_mzml_input(fname, num_decimals_ms_accuracy=4, ms_accuracy=0.0001, ret_time_accuracy=0.25,
                       direct_injection=False):
    """
    Gather m/z, intensity, and retention time data from the multiple spectra in the mzML output
    :param fname: location of mzML file to read
    :param num_decimals_ms_accuracy: int, the number of decimal significant digits for M/Z values
    :param ms_accuracy: float, the accuracy for the difference in M/Z values to consider them identical
    :param ret_time_accuracy: float, accuracy used to determine if retention times are significantly different
    :param direct_injection: boolean which determines of spectra should be combined.
    :return: ndarray with three columns: m/z, intensity, retention time, and MS level reported by the mzML file
    """
    ms_run_data = pymzml.run.Reader(fname)

    data_dict = {}
    sorted_data_dict = OrderedDict()
    num_spectra = ms_run_data.info['spectrum_count']
    num_spectra_digits = len(str(num_spectra))
    found_ms_level_0 = False
    n = 0  # make IDE happy
    # noinspection PyUnresolvedReferences
    try:
        for n, spec in enumerate(ms_run_data):
            # FYI: spectra not ordered by ms_level
            if n % 1000 == 0 and n != 0:
                print(f"    read {n:{num_spectra_digits}}/{num_spectra} spectra")
            ms_level = str(spec.ms_level)
            if ms_level == "0":
                # MS level 0 is UV absorption, which we do not use
                found_ms_level_0 = True
                continue
            if ms_level == "2":
                precursor_list = spec.selected_precursors
                for precursor_dict in precursor_list:
                    ms_level += "_mz" + f"{precursor_dict['mz']:.{num_decimals_ms_accuracy}f}".replace(".", "p")
            ret_time = spec.scan_time_in_minutes()
            spec_array = spec.peaks("raw")
            # remove 0 intensity values here, to reduce array sizes before sorting
            spec_array = spec_array[spec_array[:, 1] > 0]
            if direct_injection:
                spec_array = np.insert(spec_array, 2, np.nan, axis=1)
            else:
                spec_array = np.insert(spec_array, 2, ret_time, axis=1)
            if ms_level in data_dict:
                data_dict[ms_level] = np.concatenate((data_dict[ms_level], spec_array), axis=0)
            else:
                data_dict[ms_level] = spec_array
    except xml.etree.ElementTree.ParseError as e:
        warning(f"Problem reading file: {fname}\n    {e}\n    Exiting reading file.")
    # the pymzml.run.Reader object must be explicitly closed
    ms_run_data.close()

    cur_n = n + 1
    print(f"Read {cur_n:{num_spectra_digits}}/{num_spectra} spectra")
    if found_ms_level_0:
        print("Skipped MS Level 0 (UV absorption) data")

    # If direct injection, the only last step is to sort by retention time and then M/Z
    # May already be sorted, but sorting to be certain
    if not direct_injection:
        for ms_level, data_array in data_dict.items():
            original_len = len(data_array)
            sorted_data_dict[ms_level] = avg_duplicate_mz_ret_times(data_array, ms_accuracy, ret_time_accuracy)
            final_len = len(sorted_data_dict[ms_level])
            removed = original_len - final_len
            print(f"Removed {removed} MS{ms_level} peaks with insignificant differences in retention time "
                  f"and M/Z values.")
        return sorted_data_dict

    # Round to ms_accuracy, then combine if the same (helps with series of similar values).
    unique_mz_dict = OrderedDict()
    for ms_level, data_array in data_dict.items():
        unique_mz_dict[ms_level] = avg_duplicate_mz_intensities(data_array, ms_accuracy)
    return unique_mz_dict


def avg_duplicate_mz_ret_times(data_array, ms_accuracy, ret_time_accuracy):
    """
    If the difference between retention times is less than the tolerance, combine that data, then average intensities
    if there are any duplicate (within M/Z accuracy) M/Z values
    :param data_array: numpy array with peak data: M/Z, intensity, retention time (nan is direct injection)
    :param ms_accuracy: float, the accuracy for the difference in M/Z values to consider them identical
    :param ret_time_accuracy: float, accuracy used to determine if retention times are significantly different
    :return: data_array with unique retention times (within tolerance of DECIMALS_RET_TIME) and unique MZ for each time
    """
    data_array = data_array[data_array[:, 1] > 0]
    # rounding is needed now because of np.unique below
    data_array[:, 2] = round_to_fraction(data_array[:, 2], ret_time_accuracy)
    unique_ret_times = np.unique(data_array[:, 2])
    cleaned_array_to_round_mz = None
    # not doing a check to see if they were already unique to exit early, because still want to check for MZ uniqueness
    for ret_time in unique_ret_times:
        unique_ret_time_data_array = data_array[data_array[:, 2] == ret_time]
        if len(unique_ret_time_data_array) > 1:
            unique_ret_time_data_array = avg_duplicate_mz_intensities(unique_ret_time_data_array,
                                                                      ms_accuracy)
        if cleaned_array_to_round_mz is None:
            cleaned_array_to_round_mz = unique_ret_time_data_array
        else:
            cleaned_array_to_round_mz = np.concatenate((cleaned_array_to_round_mz, unique_ret_time_data_array), axis=0)
    return cleaned_array_to_round_mz


def avg_duplicate_mz_intensities(data_array, ms_accuracy):
    """
    Multiple cases can lead to duplicate mz intensities, such as combining data from different retention times.
    This method removes the duplicates and averages their intensities.
    This method assumes that the data is already sorted by MZ.
    :param data_array: array which may have duplicate MZ values
    :param ms_accuracy: float, the accuracy for the difference in M/Z values to consider them identical
    :return: ndarray, array with unique MZ values and intensities that were averaged for the same mz value
    """
    # rounding needed now because of np.unique below
    data_array[:, 0] = round_to_fraction(data_array[:, 0], ms_accuracy)
    data_array = data_array[data_array[:, 0].argsort()]
    unique_mz_array, inverse_array = np.unique(data_array[:, 0], return_inverse=True)
    num_unique_mz = len(unique_mz_array)
    # if there are no duplicate values, can return the input, just now rounded (if not already)
    if num_unique_mz == data_array.shape[0]:
        return data_array
    intensity_array = data_array[:, 1]

    # Create an empty array with the correct final dimensions; we'll fill in the correct intensity values
    nan_array = np.full((num_unique_mz, 1), np.nan)
    rt_time_array = np.full((num_unique_mz, 1), data_array[0][2])
    unique_data_array_to_round_mz = np.column_stack((unique_mz_array, nan_array, rt_time_array))
    # FYI: Stacking removed rounding, but not yet needed so holding off
    last_inverse_index = inverse_array[0]
    last_intensity_list = [intensity_array[0]]

    for inverse_index, intensity in zip(inverse_array[1:], intensity_array[1:]):
        if last_inverse_index == inverse_index:
            last_intensity_list.append(intensity)
        else:
            unique_data_array_to_round_mz[last_inverse_index][1] = np.average(last_intensity_list)
            last_inverse_index = inverse_index
            last_intensity_list = [intensity]
    # still need to add intensity to last row--done below
    unique_data_array_to_round_mz[-1][1] = np.average(last_intensity_list)
    # averaging may have introduced unneeded decimals in intensity, so remove them
    unique_data_array_to_round_mz[:, 1] = np.around(unique_data_array_to_round_mz[:, 1], 0)
    return unique_data_array_to_round_mz


def process_blank_file(args):
    """
    Read blank file
    :param args: command-line input and default values for program options
    :return: dict of ndarray, with MS level as key, and sorted peak data as values
    """
    blank_base_name = os.path.basename(args.blank_file_name)
    print(f"\nReading blank file: {blank_base_name}")
    blank_fname_lower = blank_base_name.lower()
    if blank_fname_lower.endswith(MZML_EXT.lower()):
        blank_data_array_dict = process_mzml_input(args.blank_file_name, args.num_decimals_ms_accuracy,
                                                   args.ms_accuracy, args.ret_time_accuracy,
                                                   direct_injection=args.direct_injection)
        for ms_level in blank_data_array_dict.keys():
            if np.isnan(blank_data_array_dict[ms_level][0][2]):
                blank_data_array_dict[ms_level] = blank_data_array_dict[ms_level][blank_data_array_dict
                                                                                  [ms_level][:, 0].argsort()]
            else:
                # this sort is needed; array arrives here sorted first by retention time
                blank_data_array_dict[ms_level] = \
                    blank_data_array_dict[ms_level][np.lexsort((-blank_data_array_dict[ms_level][:, 1],
                                                                blank_data_array_dict[ms_level][:, 0]))]

    else:
        # already checked that only MZML_EXT or CSV_EXT)
        blank_data_array_dict = read_validate_csv_data(args.blank_file_name, blank_fname_lower, args.ms_accuracy,
                                                       args.direct_injection)
    for ms_level, blank_data_array in blank_data_array_dict.items():
        # f_out = create_out_fname(args.blank_file_name, suffix=f"_ms{ms_level}_sorted", ext='csv',
        #                      base_dir=args.out_dir)
        # # noinspection PyTypeChecker
        # if args.unlabeled_csvs:
        #     np.savetxt(f_out, blank_data_array[:, :2], fmt=args.numpy_save_fmt, delimiter=',')
        # else:
        #     np.savetxt(f_out, blank_data_array, fmt=args.numpy_save_fmt, delimiter=',',
        #                header=quote('","'.join(CSV_RET_HEADER)), comments='')
        # print(f'Wrote sorted, unfiltered, MS{ms_level} data from specified blank file to: {os.path.relpath(f_out)}')
        print_clean_csv(args.blank_file_name, blank_fname_lower, ms_level, blank_data_array, "", args.direct_injection,
                        args.unlabeled_csvs, args.numpy_save_fmt, args.out_dir)

    # def print_clean_csv(fname, fname_lower, ms_level, data_array, comment, direct_injection, omit_csv_headers,
    #                     numpy_save_fmt, out_dir):
    #     if ms_level in fname_lower:
    #         suffix = ""
    #     else:
    #         suffix = f"_ms{ms_level}"
    #     if "clean" not in fname:
    #         suffix = suffix + "_clean"
    #     if direct_injection and "direct" not in fname_lower:
    #         suffix = suffix + "_direct"
    #     # data_array will already be properly sorted; not rounded but okay because printing takes care of this
    #     f_out = create_out_fname(fname, suffix=suffix, ext='csv', base_dir=out_dir)
    #     # noinspection PyTypeChecker
    #     if omit_csv_headers:
    #         np.savetxt(f_out, data_array[:, :2], fmt=numpy_save_fmt, delimiter=',')
    #     else:
    #         np.savetxt(f_out, data_array, fmt=numpy_save_fmt, delimiter=',',
    #                    header=comment + quote('","'.join(CSV_RET_HEADER)), comments='')
    #     print(f"Wrote file: {os.path.relpath(f_out)}")

    if len(blank_data_array_dict) == 0:
        raise InvalidDataError(f"Found no spectra to analyze in file: {blank_base_name}\n    Exiting program.")
    return blank_data_array_dict


def compare_blank(ms_data_array, blank_data_array, ms_accuracy, ret_time_accuracy, num_decimals_ms_accuracy,
                  ppm_threshold):
    """
    this method subtracts the intensity of each ms in the blank_data_array from the intensity of the matching
    ms in the ms_data_array
    :param blank_data_array: ndarray, the data array from processing a blank run of ms
    :param ms_data_array: ndarray, the data array from processing an actual ms run
    :param ms_accuracy: the accuracy of the MS machine used (e.g. 0.0001)
    :param ret_time_accuracy: float, accuracy used to determine if retention times are significantly different
    :param ppm_threshold: str, the tolerance (in ppm) to consider two M/Z values identical
    :param num_decimals_ms_accuracy: int, number of decimal points in MS accuracy, for rounding

    :return: ms_data_array is altered in place by subtracting blank intensities,
        ret_type (int) 0 for matched MZ in retention time(s); 1 for no matched retention times; 2 for matched
            retention time but no matched MZ
    """
    # For subtracting blank data, match different retention times (if not nan); make sure at least some overlap
    if np.isnan(blank_data_array[0][-1]) and np.isnan(ms_data_array[0][-1]):
        return subtract_blank_data(blank_data_array, ms_data_array, ms_accuracy)
    else:
        common_ret_times, blank_ret_time_dict, ms_ret_time_dict = \
            find_common_ret_times(blank_data_array, ms_data_array, ret_time_accuracy, num_decimals_ms_accuracy,
                                  ppm_threshold)
        # FYI: mz needs to to be rounded for both blank_ret_time_dict and ms_ret_time_dict
        if len(common_ret_times) == 0:
            return ms_data_array, 1
        else:
            ret_type = 2
            for ret_time_str in common_ret_times:
                blank_data = blank_ret_time_dict[ret_time_str]
                ms_data = ms_ret_time_dict[ret_time_str]
                ms_ret_time_dict[ret_time_str], ret_type = subtract_blank_data(blank_data, ms_data, ms_accuracy,
                                                                               ret_type)
            ms_ret_times = list(ms_ret_time_dict.keys())
            clean_ms_array = ms_ret_time_dict[ms_ret_times[0]]
            for ret_time_counter in range(1, len(ms_ret_times)):
                clean_ms_array = np.concatenate((clean_ms_array, ms_ret_time_dict[ms_ret_times[ret_time_counter]]),
                                                axis=0)
            return clean_ms_array, ret_type


def find_common_ret_times(blank_data_array, ms_data_array, ret_time_accuracy, num_decimals_ms_accuracy, ppm_threshold):
    """
    Facilitates comparing two arrays including ret time by making them into dicts with the ret times (as strings)
    as keys. Keeps all the ms run data in the dict, but only the data for common ret times from the blank data
    :param blank_data_array: ndarray with m/z, intensity, and retention time
    :param ms_data_array: ndarray with m/z, intensity, and retention time
    :param ret_time_accuracy: float, accuracy used to determine if retention times are significantly different
    :param num_decimals_ms_accuracy: int, number of decimal points in MS accuracy, for rounding
    :param ppm_threshold: str, the tolerance (in ppm) to consider two M/Z values identical
    :return: lists of retention times and dicts with the retention times as keys and ndarrays of peak and intensity
        data only for that retention time
    """
    # Already rounded before getting here, so not rounding again here
    blank_ret_times = np.unique(blank_data_array[:, 2])
    ms_ret_times = np.unique(ms_data_array[:, 2])

    # because of accuracy tolerance, not just using np.unique, but iterating to find those within tolerance
    common_ret_times = []
    ms_ret_time_dict = {}
    blank_ret_time_dict = {}

    blank_ret_time_counter = 0
    # want to keep all the ms run data (in a dict to make it easier to match up to blank data), but we don't
    #     need to keep all the blank data
    float_error_factor = 1.000001
    for ret_time in ms_ret_times:
        ret_time_str = str(ret_time)
        # grabbing sub array
        ms_ret_time_dict[ret_time_str] = ms_data_array[ms_data_array[:, 2] == ret_time]
        if blank_ret_time_counter < len(blank_ret_times):
            try:
                while ret_time > blank_ret_times[blank_ret_time_counter] * float_error_factor:
                    blank_ret_time_counter += 1
                diff = abs(ret_time - blank_ret_times[blank_ret_time_counter])
                # the multiplication below is needed because of machine precision error in storing floats
                if diff <= ret_time_accuracy:
                    common_ret_times.append(ret_time_str)
                    blank_ret_time = blank_ret_times[blank_ret_time_counter]
                    # grabbing sub array
                    sub_array = blank_data_array[blank_data_array[:, 2] == blank_ret_time]
                    if ret_time_str in blank_ret_time_dict:
                        # this is very unlikely to happen, but I don't mind checking for edge cases
                        blank_ret_time_dict[ret_time_str] = np.concatenate((blank_ret_time_dict[ret_time_str],
                                                                           sub_array), axis=0)
                        blank_ret_time_dict[ret_time_str][:, 2] = np.nan
                        blank_ret_time_dict[ret_time_str] = \
                            trim_close_mz_vals(blank_ret_time_dict[ret_time_str], num_decimals_ms_accuracy,
                                               ppm_threshold, ret_time_accuracy, len(blank_ret_time_dict[ret_time_str]))
                    else:
                        blank_ret_time_dict[ret_time_str] = sub_array
            except IndexError as e:
                if "out of bounds" in e.args[0]:
                    continue
                else:
                    raise InvalidDataError(e.args[0])
    return common_ret_times, blank_ret_time_dict, ms_ret_time_dict


def subtract_blank_data(blank_data, ms_data, ms_tol, ret_type=2):
    """
    Since looking at differences (not making a unique list) this method does not need mz's to be rounded first
    :param blank_data:
    :param ms_data:
    :param ms_tol:
    :param ret_type: int, 0 for matched MZ in retention time; 2 for matched retention time but no matched MZ
    :return:
    """
    blank_counter = 0
    ms_counter = 0
    try:
        while ms_counter < len(ms_data) and blank_counter < len(blank_data):
            diff = abs(blank_data[blank_counter][0] - ms_data[ms_counter][0])
            # add a little buffer so machine precision doesn't make it not match when it should
            if diff <= ms_tol * (1 + ms_tol):
                ms_data[ms_counter][1] = ms_data[ms_counter][1] - blank_data[blank_counter][1]
                ret_type = 0
                blank_counter += 1
            elif ms_data[ms_counter][0] < blank_data[blank_counter][0]:
                ms_counter += 1
            else:
                blank_counter += 1
                min_blank_val = ms_data[ms_counter][0] - ms_tol * (1. - ms_tol)
                while blank_data[blank_counter][0] < min_blank_val:
                    blank_counter += 1
        return ms_data, ret_type
    except IndexError as e:
        if "out of bounds" in e.args[0]:
            return ms_data, ret_type
        else:
            raise InvalidDataError("Error in subtract_blank_data method")


def prune_intensity(data_array, min_percent_intensity, comment):
    """
    Trim noise from data array
    :param data_array: numpy array with peak data: M/Z, intensity, retention time (nan is direct injection)
    :param min_percent_intensity: float, percent
    :param comment: str to save comments to add to output file
    :return: updated data_array (smaller)
    """
    orig_num_rows = len(data_array)
    # Calculate the minimum intensity used for screening if a M/Z will be checked for MW matching
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        max_intensity = np.amax(data_array[:, 1])
    min_req_intensity = max_intensity * min_percent_intensity / 100.
    data_array = data_array[data_array[:, 1] >= min_req_intensity]

    num_rows_after_screening = len(data_array)
    screening_criterion = f"{int(min_req_intensity)}, which is {min_percent_intensity}% of the maximum intensity"
    comment += "# Removed peaks with intensities less than " + screening_criterion + "\n"
    print(f"    Read {orig_num_rows} peaks, with maximum intensity {int(max_intensity)}")
    print(f"    {num_rows_after_screening} of these have at least the minimum intensity specified for analysis")
    return data_array, comment


def process_ms_run_file(args, fname, blank_data_array_dict):
    """
    Read ms_data and remove low intensity peaks
    :param args: command-line input and default values for program options
    :param fname: the file name of the file with ms data
    :param blank_data_array_dict: dictionary of blank data (keys are ms level); empty dict if no blank data provided
    :return:
    """
    base_name = os.path.basename(fname)
    print(f"\nReading file: {os.path.basename(fname)}")
    if args.out_dir is None:
        # if out_dir was specified, the directory has already been created; otherwise, make a directory just for this
        #     file's output
        args.out_dir = get_fname_root(fname) + DEF_SUFFIX
        make_dir(args.out_dir)
    fname_lower = base_name.lower()

    # Reading file
    if fname_lower.endswith(MZML_EXT.lower()):
        data_array_dict = process_mzml_input(fname, args.num_decimals_ms_accuracy, args.ms_accuracy,
                                             args.ret_time_accuracy, args.direct_injection)
    else:
        # previously screened that only CSV_EXT or MZML_EXT get to this point
        data_array_dict = read_validate_csv_data(fname, fname_lower, args.ms_accuracy, args.direct_injection)
        # not all mz entries in data_array_dict will be rounded, but wait to round until needed
    if len(data_array_dict) == 0:
        warning(f"Found no spectra to analyze in file: {os.path.relpath(fname)}\n    Skipping file.")
        return None

    # Clean up noise and save, if not already clean
    for ms_level, data_array in data_array_dict.items():
        print(f"Read MS Level {ms_level}")
        # # todo: it does not look like rounding below is needed
        # if not np.isnan(data_array_dict[ms_level][0][2]):
        #     data_array_dict[ms_level][:, 2] = round_to_fraction(data_array_dict[ms_level][:, 2],
        #                                                         args.ret_time_accuracy)
        if args.direct_injection or not ("clean" in fname_lower):
            comment = ""
            if ms_level in blank_data_array_dict:
                data_array, ret_type = compare_blank(data_array, blank_data_array_dict[ms_level], args.ms_accuracy,
                                                     args.ret_time_accuracy, args.num_decimals_ms_accuracy,
                                                     args.threshold)
                if ret_type == 1:
                    warning(f"No common retention times found for blank file: {os.path.relpath(args.blank_file_name)}\n"
                            f"    and ms run file: {os.path.relpath(fname)}\n")
                elif ret_type == 2:
                    warning(f"No common M/Z values found for blank file: {os.path.relpath(args.blank_file_name)}\n"
                            f"    and ms run file: {os.path.relpath(fname)}\n")
                else:
                    comment = f" Subtracted blank run data provided in file: {os.path.relpath(args.blank_file_name)}"
                    print("   " + comment)
                    comment = "#" + comment + "\n"

            # whether or not remove blank, want to prune and print
            data_array, comment = prune_intensity(data_array, args.min_rel_intensity, comment)
            # removing blanks changes sorting *if* there is retention time; change back (to be consistent with
            #     other output) now that pruned (rather than before), because it is faster to sort a shorter array
            if not np.isnan(data_array[-1][2]):
                # sorting needed; arrives sorted first by retention time
                data_array = data_array[np.lexsort((-data_array[:, 1], data_array[:, 0]))]
            print_clean_csv(fname, fname_lower, ms_level, data_array, comment, args.direct_injection,
                            args.unlabeled_csvs, args.numpy_save_fmt, args.out_dir)
            data_array_dict[ms_level] = data_array
    return data_array_dict


def print_clean_csv(fname, fname_lower, ms_level, data_array, comment, direct_injection, omit_csv_headers,
                    numpy_save_fmt, out_dir):
    if "ms" + ms_level in fname_lower:
        suffix = ""
    else:
        suffix = f"_ms{ms_level}"
    if "clean" not in fname:
        suffix = suffix + "_clean"
    if direct_injection:
        if omit_csv_headers:
            suffix = suffix + "_unlabeled"
        elif "direct" not in fname_lower:
            suffix = suffix + "_direct"
    # data_array will already be properly sorted; not rounded but okay because printing takes care of this
    f_out = create_out_fname(fname, suffix=suffix, ext='csv', base_dir=out_dir)
    # noinspection PyTypeChecker
    if omit_csv_headers:
        np.savetxt(f_out, data_array[:, :2], fmt=numpy_save_fmt, delimiter=',')
    else:
        np.savetxt(f_out, data_array, fmt=numpy_save_fmt, delimiter=',',
                   header=comment + quote('","'.join(CSV_RET_HEADER)), comments='')
    print(f"Wrote file: {os.path.relpath(f_out)}")


def write_output(fname, ms_level, num_matches, short_output_list, long_output_list, matched_formulas,
                 combined_out_fname, omit_mol_ion_flag, deprot_flag, prot_flag, write_mode, out_dir):
    """
    Print output from matching M/Z to lignin molecule library
    :param fname: location of input file processed
    :param ms_level: int, type of MS output, for output name so there are separate files from multiple-channel input
    :param num_matches: the number of matches made between input M/Z and MW in lignin library
    :param short_output_list: list of dicts of summary matching data (one list per match)
    :param long_output_list: list of dicts of extended matching data (sorted by MZ values)
    :param matched_formulas: set of formula names that were matched to M/Z values
    :param combined_out_fname: None or string if output from multiple files is to be written to one file
    :param omit_mol_ion_flag: boolean to indicate if molecular ion matches were not attempted (True) or sought (False)
    :param deprot_flag: boolean to indicate if matches were found for molecular ions
    :param prot_flag: flag to indicate if matches were found for molecular ions
    :param write_mode: flag to indicate if matches were found for molecular ions
    :param out_dir: location of output directory, or None if the current directory is the output directory
    :return: n/a; several output files created
    """
    # prepare string for txt output file
    if write_mode == 'a':
        short_txt_output_str = ''
    else:
        short_txt_output_str = MATCH_STR_HEADER
    for mz_dict in short_output_list:
        peak_str = MZ_STR_FMT.format(mz_dict[M_Z], mz_dict[INTENSITY], mz_dict[RET_TIME])
        short_txt_output_str += MATCH_STR_FMT.format(peak_str, mz_dict[REL_INTENSITY], mz_dict[CALC_MW],
                                                     mz_dict[PPM_ERR], mz_dict[PARENT_FORMULA], mz_dict[DBE],
                                                     mz_dict[MATCH_TYPE])

    ms_str = f"_ms{ms_level}"
    if ms_str in fname:
        suffix = DEF_SUFFIX
        ext_suffix = DEF_LONG_SUFFIX
    else:
        suffix = ms_str + DEF_SUFFIX
        ext_suffix = ms_str + DEF_LONG_SUFFIX
    f_out_txt = create_out_fname(fname, suffix=suffix, base_dir=out_dir, ext="txt")
    f_out_csv = create_out_fname(fname, suffix=suffix, base_dir=out_dir, ext="csv")
    if combined_out_fname:
        f_out_long = create_out_fname(combined_out_fname, suffix="_ext", base_dir=out_dir, ext="csv")
    else:
        f_out_long = create_out_fname(fname, suffix=ext_suffix, base_dir=out_dir, ext="csv")
    # Print quick summary; first note which types of matches were investigated
    if omit_mol_ion_flag:
        match_str_list = []
    else:
        match_str_list = ["molecular ion"]
    if deprot_flag:
        match_str_list.append("deprotonated ion")
    if prot_flag:
        match_str_list.append("protonated ion")
    print(f"    {num_matches} of these matched a MW in our dictionaries for a {' or a '.join(match_str_list)}")
    # save output to files
    short_write_mode = 'w'
    if num_matches == 0:
        warning(f"No MW to MZ matches (within specified ppm error) found for file: {os.path.basename(fname)}\n    "
                f"Summary output will not be printed.")
    else:
        str_to_file(short_txt_output_str, os.path.relpath(f_out_txt), print_info=True, mode=short_write_mode)
        write_csv(short_output_list, os.path.relpath(f_out_csv), SHORT_OUTPUT_HEADERS, extrasaction="ignore",
                  mode=short_write_mode)
    if out_dir:
        struct_dir = os.path.join(out_dir, STRUCT_DIR)
    else:
        struct_dir = STRUCT_DIR
    make_dir(struct_dir)
    for formula in matched_formulas:
        my_formula = formula.replace("*", "")
        make_image_grid(formula, list(FORMULA_SMI_DICT[my_formula]), out_dir=struct_dir, write_output=False)

    # print long output even if no matches
    write_csv(long_output_list, os.path.relpath(f_out_long), OUTPUT_HEADERS, extrasaction="ignore", mode=write_mode)


def trim_close_mz_vals(mz_data_array, num_decimals_ms_accuracy, ppm_threshold, ret_time_accuracy, max_num_output_mzs=5):
    """
    Create new lists so that there are not multiple M/Z values within the allowed ppm tolerance unless they have
    significantly different retention times.
    :param mz_data_array: ndarray (n, 3, where n>1) of the filtered peak data (M/Z values, intensities, retention time)
    :param num_decimals_ms_accuracy: int, the number of decimals in accuracy
    :param ppm_threshold: float, the tolerance (in ppm) to consider two M/Z values identical
    :param ret_time_accuracy: float, accuracy used to determine if retention times are significantly different
    :param max_num_output_mzs: int, max number of mz values to process
    :return: high_inten_mz_dict, dict (mz_str as keys) with data for each unique mz value
    """
    # Sort to encounter highest intensity MZs first
    sorted_array = np.flip(mz_data_array[mz_data_array[:, 1].argsort()], 0)
    high_inten_mz_array = None
    processed_mz_strs = {}
    for mz_vals in sorted_array:
        # check for uniqueness with string, which discards insignificant digits and avoids floating-point matching
        mz_str = f"{mz_vals[0]:.{num_decimals_ms_accuracy}f}"
        unique_mz = True
        if mz_str in processed_mz_strs.keys() and (np.isclose(mz_vals[2], processed_mz_strs[mz_str], equal_nan=True)
                                                   or abs(mz_vals[2] - processed_mz_strs[mz_str])
                                                   <= ret_time_accuracy * (1. + ret_time_accuracy)):
            unique_mz = False
        else:
            for selected_mz, selected_ret_time in processed_mz_strs.items():
                # compare both retention time and M/Z values to determine uniqueness
                if np.isclose(mz_vals[2], selected_ret_time, equal_nan=True):
                    pmm_err = calc_accuracy_ppm(mz_vals[0], float(selected_mz))
                    if abs(pmm_err) < ppm_threshold:
                        unique_mz = False
                        break
            processed_mz_strs[mz_str] = mz_vals[2]
        if unique_mz:
            if high_inten_mz_array is None:
                high_inten_mz_array = mz_vals
            # elif len(mz_vals) == 3:
            else:
                high_inten_mz_array = np.row_stack((high_inten_mz_array, mz_vals))
            # else:
            #     high_inten_mz_array = np.concatenate((high_inten_mz_array, mz_vals), axis=0)
                if high_inten_mz_array.shape[0] == max_num_output_mzs:
                    break
    num_unique_peaks = len(high_inten_mz_array)
    if len(sorted_array) == num_unique_peaks:
        print(f"    All {num_unique_peaks} peaks are unique: their differences in M/Z values are greater than the\n"
              f"        specified ppm tolerance and/or, if retention times are reported, their differences in\n"
              f"        retention times are greater than the specified retention time accuracy.")
    else:
        print(f"    Found {len(high_inten_mz_array)} unique peaks after removing peaks with M/Z values within the "
              f"specified ppm tolerance and,\n"
              f"        if retention time data is used, also within the specified retention time accuracy reported.\n"
              f"        From a group of peaks within these tolerances, only the highest intensity peak is reported \n"
              f"        (without intensity averaging or centroid calculations).")
    return high_inten_mz_array


def print_high_inten_short(ms_level, trimmed_mz_array, max_output_mzs=5):
    """
    print out the closest matches to the top 5 intensities, showing their ppm error
    :param ms_level: str, MS level read
    :param trimmed_mz_array: ndarray of peak data, sorted by highest intensity to lowest
    :param max_output_mzs: int, max number of mz values to print
    :return: n/a, prints to stdout
    """
    if "_" in ms_level:
        ms_level = ms_level.replace("_", ", ").replace("p", ".")

    high_inten_str = f"Summary output for MS level {ms_level}: (up to) {max_output_mzs} unique M/Z values with " \
                     f"the highest intensities\n" + MZ_STR_HEADER

    # sometimes, just one row in trimmed_mz_array (a vector) and needs to be treated differently
    if len(trimmed_mz_array.shape) == 1:
        high_inten_str += MZ_STR_FMT.format(trimmed_mz_array[0], trimmed_mz_array[1], trimmed_mz_array[2])
    else:
        for mz_id, mz_vals in enumerate(trimmed_mz_array):
            if mz_id < max_output_mzs:
                high_inten_str += MZ_STR_FMT.format(mz_vals[0], mz_vals[1], mz_vals[2])
            else:
                break

    print(high_inten_str)


def print_high_inten_long(ms_level, high_int_peak_str_list, long_output_dict, max_output_mzs=5):
    """
    print out the closest matches to the top 5 intensities, showing their ppm error
    :param ms_level: str, MS level read
    :param high_int_peak_str_list: list, strings of peak data for the highest intensity unique peaks, in descending
        order of intensity
    :param long_output_dict: OrderedDict, keys are strings of peak data, and values are the extended output dicts
    :param max_output_mzs: int, max number of mz values to process
    :return: n/a, prints to stdout
    """
    if "_" in ms_level:
        ms_level = ms_level.replace("_", ", ").replace("p", ".")

    high_int_print_dict = {}

    high_inten_str = f"Summary output for MS level {ms_level}: closest matches for (up to) the top " \
                     f"{max_output_mzs} unique M/Z values with the highest intensities.\n" + MATCH_STR_HEADER

    # the high_int_peak_str_list is sorted by M/Z; print in order of highest intensity
    for peak_str in high_int_peak_str_list:
        output_dict = long_output_dict[peak_str]
        intensity_int = int(output_dict[INTENSITY])
        high_int_print_dict[intensity_int] = MATCH_STR_FMT.format(peak_str, output_dict[REL_INTENSITY],
                                                                  output_dict[MIN_ERR_MW], output_dict[MIN_ERR],
                                                                  output_dict[MIN_ERR_FORMULA],
                                                                  output_dict[MIN_ERR_DBE],
                                                                  output_dict[MIN_ERR_MATCH_TYPE])
    intensity_list = sorted(list(high_int_print_dict), reverse=True)
    for intensity_int in intensity_list:
        high_inten_str += high_int_print_dict[intensity_int]

    print(high_inten_str)
