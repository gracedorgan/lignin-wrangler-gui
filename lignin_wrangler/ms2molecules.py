#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ms2molecules.py
Read raw CSVs output from MS and output likely molecular formulas, specifically from lignin and model lignin samples
"""

import argparse
import fnmatch
import os
import sys
import numpy as np
from collections import defaultdict
from common_wrangler.common import (warning, InvalidDataError, GOOD_RET, INPUT_ERROR, check_for_files, make_dir,
                                    IO_ERROR, INVALID_DATA)
from lignin_wrangler import __version__
from lignin_wrangler.lignin_common import (DEF_SUFFIX, DEF_LONG_SUFFIX, CSV_EXT, MZML_EXT, DEF_MS_ACCURACY,
                                           DEF_RET_TIME_ACCURACY, ION_ENERGIES, PARENT_MZ, PARENT_FORMULA,
                                           PARENT_MATCH_ERR, ION_MZ_DICT)
from lignin_wrangler.match_formulas import compare_mz_mw
from lignin_wrangler.process_ms_input import (process_blank_file, process_ms_run_file, write_output,
                                              print_high_inten_long)
from lignin_wrangler.structure_search import (make_ms2_dict, make_dbe_mw_graphs, find_substructure_sets)
from lignin_wrangler.ms_plotting import initial_output, plot_mz_v_intensity

DEF_MIN_INTENSITY = 5.
DEF_THRESHOLD = 10.


def validate_decimal_input(input_accuracy, option_str):
    """
    Make sure that the provided input is within the allowable range, and determine the number of significant decimals
    :param input_accuracy: str or float
    :param option_str: the option corresponding to the input, to allow a more helpful error message
    :return: input_accuracy as float, num_decimals_accuracy as int
    """
    max_val = 1.
    min_val = 0.000000001
    tolerance = 1e-6
    try:
        input_accuracy = float(input_accuracy)
        if input_accuracy < min_val or input_accuracy > max_val:
            raise ValueError
        base_10_log = np.log10(input_accuracy)
        if not abs(base_10_log - round(base_10_log, 0) < tolerance):
            remainder = 1. % input_accuracy
            if not remainder < tolerance:
                raise ValueError
        num_decimals_accuracy = int(len(str(input_accuracy)) - str(input_accuracy).index('.') - 1.)

    except ValueError:
        raise InvalidDataError(f"Read '{input_accuracy}' for the {option_str} option.\n"
                               f"    This tolerance must be a non-negative fraction of 1 (1 must be a multiple of the "
                               f"tolerance), between {min_val} and {max_val}.")
    return input_accuracy, num_decimals_accuracy


def validate_input(args):
    """
    Checks for valid command-line input and performs any required casting
    :param args: command-line input and default values for program options
    """
    # '-d', '-e', '-f',  and 'l' skipped: they are already the required type (str) and validation performed as
    #     part of the function that looks for files to process
    # '-s' skipped: Boolean will be returned by argparse, and an error if the user tries to give it a value

    try:
        # if already a float, no problem
        args.threshold = float(args.threshold)
        if args.threshold < 0 or args.threshold > 1000:
            raise ValueError
    except ValueError:
        raise InvalidDataError(f"Read '{args.threshold}' for the threshold value (in ppm; '-t' option) for matching "
                               f"M/Z to MW. \n    This must be a non-negative number, no greater than 1000.")

    args.ms_accuracy, args.num_decimals_ms_accuracy = validate_decimal_input(args.ms_accuracy, "'-a'/'--ms_accuracy'")
    args.ret_time_accuracy, args.num_decimals_ret_time_accuracy = validate_decimal_input(args.ret_time_accuracy,
                                                                                         "'-r'/'--ret_time_accuracy'")
    if args.unlabeled_csvs:
        args.direct_injection = True
        # When unlabeled CSVS are chosen, peaks should be combined as with direct injection
        args.numpy_save_fmt = f'%.{args.num_decimals_ms_accuracy}f,%.0f'
    else:
        args.numpy_save_fmt = f'%.{args.num_decimals_ms_accuracy}f,%.0f,%.{args.num_decimals_ret_time_accuracy}f'

    try:
        args.min_rel_intensity = float(args.min_rel_intensity)
        if args.min_rel_intensity < 0 or args.min_rel_intensity > 100:
            raise ValueError
    except ValueError:
        raise InvalidDataError(f"Read {args.min_rel_intensity}% for the minimum relative intensity (percent of "
                               f"the maximum intensity required\n    for peak to be analyzed; "
                               f"'-m' option). This must be a non-negative number, no greater than 100.")


def print_header(args):
    print(f"Running ms2molecules from lignin_wrangler version {__version__}")
    if args.direct_injection:
        print(f"    Direct injection option chosen: retention times will not be reported. Non-unique peaks "
              f"(M/Z values within\n"
              f"        the specified machine accuracy of {args.ms_accuracy}) will be reported as one peak with the "
              f"average intensity.")
    if args.quit_after_mzml_to_csv:
        print(f"    Converting mzML files to CSV files with no further analysis.")
    else:
        print(f"    Entries with intensities less than {args.min_rel_intensity:.1f}% of the maximum intensity will be "
              f"skipped.\n"
              f"    M/Z values are considered unique if they differ by more than the specified mass measurement error\n"
              f"        of {args.threshold:.1f} ppm or less and, if retention data is to be reported, the retention "
              f"times differ by\n"
              f"        more than the specified {args.ret_time_accuracy:.2f} min accuracy. If multiple peaks are found "
              f"to be within both the\n"
              f"        mass and retention time accuracies, only the peak with the highest intensity will be reported\n"
              f"        (without intensity averaging or reporting centroid results).\n"
              f"    Retention times are reported in minutes.")


def parse_cmdline(argv):
    """
    Returns the parsed argument list and return code.
    :param argv: `argv` is a list of arguments, or `None` for ``sys.argv[1:]``.
    :return: args: the parsed argument list
    :return: int: the return code. returns GOOD_RET if no error, INPUT_ERROR if error
    """
    if argv is None:
        argv = sys.argv[1:]

    # initialize the parser object:
    parser = argparse.ArgumentParser(description=f"This script is designed to match M/Z values to molecular formulas "
                                                 f"postulated (or confirmed) to be potential lignin fragmentation "
                                                 f"products. For each matched molecular formula, the program will "
                                                 f"output which chemical species in our library match that formula.\n"
                                                 f"If MS2 data is provided, it will provide additional information to "
                                                 f"aid identifying the chemical species of the MS2 precursor ion and "
                                                 f"its daughters, searching within the lignin compound library.\n" 
                                                 f"The program can read mzML and CSV files. CSV files are expected to "
                                                 f"have either 2 columns (M/Z,INTENSITY) or 3 columns (M/Z,INTENSITY,"
                                                 f"RETENTION TIME), and may have any number of comment lines at the "
                                                 f"beginning of the file (comment lines must start with '#') and "
                                                 f"0 or 1 header lines.\n"
                                                 f"The generated output files depend on the options chosen below. "
                                                 f"Unless 'clean' or 'sorted' is found in the file name, the program "
                                                 f"will output processed peak data (zero intensity peaks removed, "
                                                 f"etc.). Unless the '-q' option is selected, the program will output "
                                                 f"three summary files. 1) A CSV file with abbreviated output "
                                                 f"reporting only M/Z values which match (within the specified ppm "
                                                 f"threshold) a molecular formula in our library (accounting for "
                                                 f"possible protonation or deprotonation). The default output file "
                                                 f"name will have the same base file name as the input file, but "
                                                 f"ending in '{DEF_SUFFIX}.csv'. 2) The same data will be output in "
                                                 f"a file ending '{DEF_SUFFIX}.txt', formatted for easier viewing. "
                                                 f"3) and a CSV file ending in '{DEF_LONG_SUFFIX}.csv', which also "
                                                 f"includes the closest matching molecular formula (even if the ppm "
                                                 f"error is not within the specified threshold), and SMILES strings "
                                                 f"and compound names (when available) from our lignin compound "
                                                 f"library.\n"
                                                 f"By default, matches are attempted for M/Z values that correspond to "
                                                 f"molecular ions (unless the '-x' option is used) and to "
                                                 f"singly-protonated ions (if there is a '+' in the file "
                                                 f"name) or singly-deprotonated ions (if there is a '-' in the file "
                                                 f"name).")
    parser.add_argument("-a", "--ms_accuracy", help=f"The precision of the MS used, with default value "
                                                    f"{DEF_MS_ACCURACY}. This value is used as the tolerance to "
                                                    f"determine if two M/Z values should be considered identical when "
                                                    f"comparisons are required such as when subtracting noise (if "
                                                    f"'blank' output is provided) and/or combining peaks with different"
                                                    f" retention times, as done when direct injection is specified "
                                                    f"with the '-i'/'--direct_injection' option or if there are "
                                                    f"retention times whose difference is less than the specified "
                                                    f"retention time accuracy (the '-r'/'ret_time_accuracy' option).",
                        default=DEF_MS_ACCURACY)
    parser.add_argument("-b", "--blank_file_name", help=f"The file name of a blank ms run.", default=None)
    parser.add_argument("-c", "--combined_output_fname", help=f"Specifies to combine output from all files processed "
                                                              f"in an invocation of this program into files with a "
                                                              f"base name provided with this option.  This overrides "
                                                              f"the default behavior to write output from "
                                                              f"different files to separate output files, with "
                                                              f"'{DEF_SUFFIX}' added to the base name.", default=None)
    parser.add_argument("-d", "--directory", help=f"By default, the program will search the current directory for "
                                                  f"files with the '{MZML_EXT}' extension to process, and if no "
                                                  f"'{MZML_EXT}' files are found, it will look for '{CSV_EXT}' files, "
                                                  f"ignoring those with names matching output of this program. If this "
                                                  f"option is not used and either the '-f' or '-l' options are used, "
                                                  f"the program will instead search for exact matches to provided "
                                                  f"names. Use this option to specify that a search should be "
                                                  f"performed even if the '-f' or '-l' options are used, or to change "
                                                  f"the search location to a different directory. If you wish to also "
                                                  f"search in subdirectories, include use the '-s' flag.", default=None)
    parser.add_argument("-f", "--file_name", help="File name to read.", default=None)
    parser.add_argument("-i", "--direct_injection", help="If this option is chosen and retention time is included in "
                                                         "input, the program will combine redundant M/Z values from "
                                                         "different spectra, reporting the average intensity and "
                                                         "omitting retention times.", action='store_true')
    parser.add_argument("-l", "--list_file", help="File name of file containing a list of files to be read, one file "
                                                  "name per line.", default=None)
    parser.add_argument("-m", "--min_rel_intensity", help=f"By default, this program will only analyze peaks that have "
                                                          f"intensity at least {DEF_MIN_INTENSITY}%% of the largest "
                                                          f"peak. Use this option to change the value from this "
                                                          f"default.", default=DEF_MIN_INTENSITY)
    parser.add_argument("-n", "--non_dom_iso_flag", help=f"By default, this program will add *'s next to the formulas "
                                                         f"of non-dominant isotopes. The number of *'s corresponds to "
                                                         f"the number of steps away from the most dominant isotopes "
                                                         f"that this isotope.", action='store_false')
    parser.add_argument("-o", "--out_dir", help=f"By default, output created for each file will be saved to a "
                                                f"directory named with the base name of the file plus the suffix read, "
                                                f"'{DEF_SUFFIX}'. If this option is chosen, all output (including "
                                                f"from multiple files, if multiple files are read) will be saved to "
                                                f"the specified directory name. Directories will be created if they "
                                                f"do not yet exist.", default=None)
    parser.add_argument("-p", "--plot_data", help=f"This option will create the intensity versus M/Z plots for all "
                                                  f"files provided with either the -f/--file_name option or the "
                                                  f"-l/--list_file options. If there is retention time in the "
                                                  f"data, peaks with different retention times will be combined by"
                                                  f"averaging intensities, as they are done in direct injection. "
                                                  f"If retention data is included, additional plots will be created: "
                                                  f"total intensity versus time, and unique M/Z values (differ more "
                                                  f"than specified ppm tolerance) intensity versus time.",
                        action='store_true')
    parser.add_argument("-q", "--quit_after_mzml_to_csv", help="Flag do indicate that files should be read, low "
                                                               "intensity peaks removed (using min_rel_intensity and, "
                                                               "if provided, blank run data), and saved to a csv.",
                        action='store_true')
    parser.add_argument("-r", "--ret_time_accuracy", help=f"The precision of the retention times (if applicable) "
                                                          f"included, with default value {DEF_RET_TIME_ACCURACY}. "
                                                          f"This value is used as the tolerance to determine whether "
                                                          f"two peaks with M/Z values within MS accuracy (the "
                                                          f"'-a'/'ms_accuracy' option) should be combined (averaging "
                                                          f"intensity).", default=DEF_RET_TIME_ACCURACY)
    parser.add_argument("-s", "--sub_dir_flag", help="Flag do indicate that subdirectories should be searched, in "
                                                     "addition to the directory specified by the '-d' option. This "
                                                     "option is ignored if the -f/--file_name or -l/--list_file "
                                                     "options are used.", action='store_true')
    parser.add_argument("-t", "--threshold", help=f"Accuracy threshold in ppm. The default value is {DEF_THRESHOLD} "
                                                  f"ppm. This threshold is used to determine which, if any, MWs in our "
                                                  f"dictionary match the provided M/Z, assuming that the charge is +1 "
                                                  f"or -1.", default=DEF_THRESHOLD)
    parser.add_argument("-u", "--unlabeled_csvs", help=f"This option omits comments, labels, and retention times in "
                                                       f"output CSVs, as this is the format required by some other "
                                                       f"MS output processing programs.", action='store_true')
    parser.add_argument("-x", "--omit_mol_ion", help="Flag do indicate that the program should not attempt to match "
                                                     "M/Z values to molecular ion MWs (which is the default action).",
                        action='store_true')
    args = None

    try:
        args = parser.parse_args(argv)
        validate_input(args)
    except (InvalidDataError, SystemExit) as e:
        if hasattr(e, 'code') and e.code == 0:
            return args, GOOD_RET
        warning(e)
        parser.print_help()
        return args, INPUT_ERROR

    return args, GOOD_RET


def collect_check_process_file_list(args):
    # find files to process, allowing any combination of a file name, a list of file names, searching dir...

    # first look for mzML files
    process_file_list = check_for_files(args.file_name, args.list_file, search_pattern=MZML_EXT,
                                        search_dir=args.directory, search_sub_dir=args.sub_dir_flag,
                                        warn_if_no_matches=False)
    # if no mzml files, look for csv files
    if len(process_file_list) == 0:
        process_file_list = check_for_files(args.file_name, args.list_file, search_pattern=CSV_EXT,
                                            search_dir=args.directory, search_sub_dir=args.sub_dir_flag,
                                            warn_if_no_matches=False)

    # Now check that didn't accidentally get program output or wrong file type
    filtered_file_list = []
    for fname in process_file_list:
        fname_lower = os.path.basename(fname).lower()
        if not (fname_lower.endswith(MZML_EXT.lower()) or fname_lower.endswith(CSV_EXT.lower())):
            warning(f"The expected file extensions are '{MZML_EXT}' and '{CSV_EXT}'.\n"
                    f"    Encountered file: {os.path.relpath(fname)}.\n    Skipping file.")
            continue
        if not (fnmatch.fnmatch(fname_lower, "*matched.csv") or fnmatch.fnmatch(fname_lower, "*matched_ext.csv")):
            filtered_file_list.append(fname)
    if not len(filtered_file_list) and args.blank_file_name is None:
        raise InvalidDataError("No files found to process. Exiting program.")
    process_file_list = filtered_file_list

    # Make sure that blank files are not processed twice by removing from this list if it is there
    if args.blank_file_name:
        if args.blank_file_name in process_file_list:
            process_file_list.remove(args.blank_file_name)
        fname_lower = os.path.basename(args.blank_file_name).lower()
        if not (fname_lower.lower().endswith(MZML_EXT.lower()) or fname_lower.endswith(CSV_EXT.lower())):
            raise InvalidDataError(f"The expected file extensions for MS output are '{MZML_EXT}' and '{CSV_EXT}'.\n"
                                   f"    Specified blank file: {os.path.relpath(args.blank_file_name)}\n"
                                   f"    Exiting program.")

    # Now check names for protonated/deprotonated flags
    for fname in process_file_list:
        fname_lower = os.path.basename(fname).lower()
        pos_match = fnmatch.fnmatch(fname_lower, "*+*")
        neg_match = fnmatch.fnmatch(fname_lower, "*-*")
        if pos_match and neg_match:
            raise InvalidDataError(f"Found both a '+' and a '-' in the file name: {os.path.relpath(fname)}\n"
                                   f"    Only one of these characters can appear in a file name, as this "
                                   f"program uses these characters to determine if matches should be attempted for "
                                   f"protonated ('+') or deprotonated ('-') ion MWs.")
        if not (pos_match or neg_match) and not args.quit_after_mzml_to_csv:
            if args.omit_mol_ion:
                raise InvalidDataError(f"The '-x'/'--omit_mol_ion' option was selection, although there is no '+' nor "
                                       f"'-' in the file name: {os.path.relpath(fname)}\n    Thus, no matching type "
                                       f"has been selected. Program exiting.")
            else:
                warning(f"Since neither '+' nor '-' appear in the file name: {os.path.relpath(fname)}\n    "
                        f"Only matches to the molecular ion will be reported.")

    # While setting up, also make output directory if it does, so there is a place to save a CSV file
    if args.out_dir:
        make_dir(args.out_dir)

    return process_file_list


def main(argv=None):
    """
    Calls to other functions to perform all requested tasks
    :param argv: arguments
    :return: int depending on exit type (0 = good return; a positive integer for all others)
    """
    args, ret = parse_cmdline(argv)
    if ret != GOOD_RET or args is None:
        return ret

    try:
        process_file_list = collect_check_process_file_list(args)
        print_header(args)
        write_mode = 'w'
        ms2_dict = defaultdict(lambda: defaultdict(lambda: None))
        blank_data_array_dict = {}
        max_num_mz_in_stdout = 5
        if args.quit_after_mzml_to_csv:
            max_unique_mz_to_collect = max_num_mz_in_stdout
        else:
            max_unique_mz_to_collect = 1e9

        # blank file processing
        if args.blank_file_name is not None:
            fname = args.blank_file_name
            fname_lower = os.path.basename(fname).lower()
            blank_data_array_dict = process_blank_file(args)
            for ms_level, ms_array in blank_data_array_dict.items():
                initial_output(fname, fname_lower, ms_array, ms_level, max_unique_mz_to_collect, max_num_mz_in_stdout,
                               args.threshold, args.num_decimals_ms_accuracy, args.ret_time_accuracy,
                               args.num_decimals_ret_time_accuracy, args.out_dir, args.quit_after_mzml_to_csv,
                               args.direct_injection)

        # all other file processing
        gathered_ms2_data = False
        for fname in process_file_list:
            base_fname = os.path.basename(fname)
            fname_lower = base_fname.lower()
            data_array_dict = process_ms_run_file(args, fname, blank_data_array_dict)
            if data_array_dict is None:
                continue
            prot_flag = fnmatch.fnmatch(fname_lower, "*+*")
            deprot_flag = fnmatch.fnmatch(fname_lower, "*-*")
            for ms_level, ms_array in data_array_dict.items():
                trimmed_mz_array = initial_output(fname, fname_lower, ms_array, ms_level, max_unique_mz_to_collect,
                                                  max_num_mz_in_stdout, args.threshold, args.num_decimals_ms_accuracy,
                                                  args.ret_time_accuracy, args.num_decimals_ret_time_accuracy,
                                                  args.out_dir, args.quit_after_mzml_to_csv, args.direct_injection)
                if args.quit_after_mzml_to_csv or "blank" in fname_lower:
                    # move to next file without further analysis
                    continue

                num_matches, matched_formulas, short_output_list, long_output_dict, high_int_peak_str_list = \
                    compare_mz_mw(base_fname, trimmed_mz_array, args.threshold, args.omit_mol_ion,
                                  deprot_flag, prot_flag, args.non_dom_iso_flag, max_num_mz_in_stdout)

                # do the following even if previously processed, because likely that the processed files came from a
                #     run with args.quit_after_mzml_to_csv, so no yet done
                write_output(fname, ms_level, num_matches, short_output_list, long_output_dict.values(),
                             matched_formulas, args.combined_output_fname, args.omit_mol_ion, deprot_flag,
                             prot_flag, write_mode, args.out_dir)
                print_high_inten_long(ms_level, high_int_peak_str_list, long_output_dict)

                # if there is an combined_output_fname and more than one file, append the data, don't write over it
                if args.combined_output_fname:
                    write_mode = 'a'
                if "2" in ms_level and not ("blank" in fname_lower) and ("hcd" in fname_lower):
                    if len(process_file_list) < 2:
                        warning(f"Only one set of MS2 data has been read (from file {base_fname}).\n"
                                f"    No chemical species matching will be attempted, since at least two MS2 data "
                                f"sets must be provided for\n    this procedure, with one of the data sets coming "
                                f"from a 0 ionization energy run.")
                    else:
                        make_ms2_dict(fname_lower, ms_level, ms2_dict, trimmed_mz_array, long_output_dict,
                                      args.threshold, args.ms_accuracy, args.num_decimals_ms_accuracy)
                        gathered_ms2_data = True

        # If '-q' option used to exit without analysis, skip the next section (which skips to the end of the
        #     program
        if not args.quit_after_mzml_to_csv:
            if gathered_ms2_data:
                for fkey, fkey_dict in ms2_dict.items():
                    ion_energies = fkey_dict[ION_ENERGIES].keys()
                    num_ion_energies = len(ion_energies)
                    if num_ion_energies < 2:
                        warning(f"No chemical species matching will be attempted for files designated {fkey}.\n    "
                                f"For this functionality, at least two MS2 data sets must be provided, one of them "
                                f"from a 0 ionization energy run.\n    Note that the program matches sets of MS2 "
                                f"output by searching for file names that differ only the number after\n    'HCD' "
                                f"(case insensitive), which is read as the ionization energy.")
                    elif 0 not in ion_energies:
                        warning(f"Did not find 0 ionization energy output for the set of files designated {fkey}.\n    "
                                f"This output is used to identify the parent. Contact the developers for "
                                f"more options.")
                    else:
                        print(f"\nNow analyzing {fkey} files.\nUsing M/Z "
                              f"{fkey_dict[PARENT_MZ]:.{args.num_decimals_ms_accuracy}f} as the parent peak, as "
                              f"it is the closest peak in the 0 ionization output to the\n    specified precursor ion. "
                              f"The closest matching molecular formula, {fkey_dict[PARENT_FORMULA]} (with "
                              f"{fkey_dict[PARENT_MATCH_ERR]:.1f} ppm error),\n    will be used as the parent "
                              f"formula.")
                        plot_mz_v_intensity(fkey, fkey_dict[ION_MZ_DICT], args.num_decimals_ms_accuracy,
                                            args.out_dir)
                        make_dbe_mw_graphs(fkey, fkey_dict[ION_ENERGIES], args.out_dir)
                        find_substructure_sets(fkey, fkey_dict, args.threshold, args.num_decimals_ms_accuracy,
                                               args.out_dir)

    except IOError as e:
        warning(e)
        return IO_ERROR
    except InvalidDataError as e:
        warning(e)
        return INVALID_DATA

    return GOOD_RET  # success


if __name__ == "__main__":
    status = main()
    sys.exit(status)
