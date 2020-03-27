import fnmatch
import os
import zipfile
from collections import defaultdict
from os.path import dirname, realpath, join, isfile
import numpy as np
from common_wrangler.common import InvalidDataError, warning, read_csv_header, silent_remove
from flask import Flask, render_template, request, send_from_directory, send_file
from lignin_wrangler.lignin_common import MZ_STR_HEADER, MZ_STR_FMT, MATCH_STR_FMT, REL_INTENSITY, MIN_ERR, \
    MATCH_STR_HEADER, MIN_ERR_MW, MIN_ERR_FORMULA, MIN_ERR_DBE, MIN_ERR_MATCH_TYPE, ION_ENERGIES, ION_MZ_DICT, INTENSITY
from lignin_wrangler.match_formulas import calc_accuracy_ppm, compare_mz_mw
from lignin_wrangler.ms_plotting import initial_output, plot_mz_v_intensity
from lignin_wrangler.process_ms_input import check_input_csv_header, process_blank_file, process_ms_run_file, \
    write_output
from lignin_wrangler.lignin_common import (DEF_SUFFIX, DEF_LONG_SUFFIX, CSV_EXT, MZML_EXT, DEF_MS_ACCURACY,
                                           DEF_RET_TIME_ACCURACY, ION_ENERGIES, PARENT_MZ, PARENT_FORMULA,
                                           PARENT_MATCH_ERR, ION_MZ_DICT)
from lignin_wrangler.structure_search import  make_dbe_mw_graphs, get_all_substructures, \
    output_substructures, make_ms2_dict, find_substructure_sets

from rdkit import Chem
from werkzeug.utils import secure_filename
import re

app = Flask(__name__)

UPLOADS_PATH = join(dirname(realpath(__file__)), 'static/uploads/')
OUTPUT_PATH = join(dirname(realpath(__file__)), 'static/output/')
ALLOWED_EXTENSIONS = {'mzML', 'csv'}
TYPICAL_CSV_HEADER = ["M/Z", "INTENSITY"]
RET_TIME_CSV_HEADER = "RETENTION TIME (MIN)"


class MyArgs:
    num_decimals_ms_accuracy = 0
    num_decimals_ret_time_accuracy = 0
    numpy_save_fmt = None

    def __init__(self, ms_accuracy, blank_file_name, combined_output_fname, direct_injection, min_rel_intensity,
                 non_dom_iso_flag, plot_data, quit_after_mzml_to_csv,
                 ret_time_accuracy, threshold, unlabeled_csvs, picture_output_flag, omit_mol_ion):
        self.omit_mol_ion = omit_mol_ion

        self.ms_accuracy = ms_accuracy
        self.blank_file_name = blank_file_name
        self.combined_output_fname = combined_output_fname
        self.directory = UPLOADS_PATH
        self.file_name = None
        self.direct_injection = direct_injection
        self.list_file = None
        self.min_rel_intensity = min_rel_intensity
        self.non_dom_iso_flag = non_dom_iso_flag
        self.out_dir = OUTPUT_PATH
        self.plot_data = plot_data
        self.quit_after_mzml_to_csv = quit_after_mzml_to_csv
        self.ret_time_accuracy = ret_time_accuracy
        self.sub_dir_flag = False
        self.threshold = threshold
        self.unlabeled_csvs = unlabeled_csvs
        self.picture_output_flag = picture_output_flag
        print("self.picoutflag = ", self.picture_output_flag)


def validate_decimal_input(input_accuracy, option_str):
    """
    Make sure that the provided input is within the allowable range, and determine the number of significant decimals
    :param input_accuracy: str or float
    :param option_str: the option corresponding to the input, to allow a more helpful error message
    :return: input_accuracy as float, num_decimals_accuracy as int, my_error as string
    """
    max_val = 1.
    min_val = 0.000000001
    tolerance = 1e-6

    my_error = None
    input_accuracy = float(input_accuracy)
    if input_accuracy < min_val or input_accuracy > max_val:
        my_error = f"Read '{input_accuracy}' for the {option_str} option.\n    This tolerance must be a non-negative " \
                   f"fraction of 1 (1 must be a multiple of the tolerance), between {min_val} and {max_val}. "
        print("IN IF DECIS")
    base_10_log = np.log10(input_accuracy)
    if not abs(base_10_log - round(base_10_log, 0) < tolerance):
        remainder = 1. % input_accuracy
        if not remainder < tolerance:
            my_error = f"Read '{input_accuracy}' for the {option_str} option.\n    This tolerance must be a " \
                       f"non-negative " \
                       f"fraction of 1 (1 must be a multiple of the tolerance), between {min_val} and {max_val}. "
    if my_error:
        num_decimals_accuracy = None
    else:
        num_decimals_accuracy = int(len(str(input_accuracy)) - str(input_accuracy).index('.') - 1.)
    print("MY ERROR IN VAL DECIS", my_error)
    return input_accuracy, num_decimals_accuracy, my_error


def validate_input(args):
    """
    Checks for valid command-line input and performs any required casting
    :param args: objects containing all user input for this program
    :return: error as a str
    """
    # '-d', '-e', '-f',  and 'l' skipped: they are already the required type (str) and validation performed as
    #     part of the function that looks for files to process
    # '-s' skipped: Boolean will be returned by argparse, and an error if the user tries to give it a value

    # if already a float, no problem
    error = None
    args.threshold = float(args.threshold)
    if args.threshold < 0 or args.threshold > 1000:
        error = f"Read '{args.threshold}' for the threshold value (in ppm; '-t' option) for matching M/Z to MW. \n    " \
                f"This must be a non-negative number, no greater than 1000. "

    if args.blank_file_name:

        args.blank_file_name = os.path.join(UPLOADS_PATH, args.blank_file_name)
        if not os.path.isfile(args.blank_file_name):
            error = f"Read '{args.blank_file_name}' for the blank file name. \n This must be a real file you have " \
                    f"uploaded. "
    else:
        args.blank_file_name = None

    args.ms_accuracy, args.num_decimals_ms_accuracy, deci_error_1 = validate_decimal_input(args.ms_accuracy,
                                                                                           "'-a'/'--ms_accuracy'")
    args.ret_time_accuracy, args.num_decimals_ret_time_accuracy, deci_error_2 = validate_decimal_input(
        args.ret_time_accuracy,
        "'-r'/'--ret_time_accuracy'")
    if deci_error_1:
        error = deci_error_1
    if deci_error_2:
        error = deci_error_2
    if args.unlabeled_csvs:
        args.numpy_save_fmt = f'%.{args.num_decimals_ms_accuracy}f,%.0f'
    else:
        args.numpy_save_fmt = f'%.{args.num_decimals_ms_accuracy}f,%.0f,%.{args.num_decimals_ret_time_accuracy}f'

    args.min_rel_intensity = float(args.min_rel_intensity)
    if args.min_rel_intensity < 0 or args.min_rel_intensity > 100:
        error = f"Read {args.min_rel_intensity}% for the minimum relative intensity (percent of the maximum intensity " \
                f"required\n    for peak to be analyzed; '-m' option). This must be a non-negative number, no greater " \
                f"than 100. "

    # When unlabeled CSVS are chosen, peaks should be combined as with direct injection
    if args.unlabeled_csvs:
        args.numpy_save_fmt = f'%.{args.num_decimals_ms_accuracy}f,%.0f'
        args.direct_injection = True
    return error


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# todo: need to delete this method, get it to properly import from lignin wrangler
def trim_close_mz_vals(mz_data_array, num_decimals_ms_accuracy, ppm_threshold, max_output_mzs=5):
    """
    Create new lists so that there are not multiple M/Z values within the allowed ppm tolerance unless they have
    significantly different retention times.
    :param mz_data_array: ndarray (n, 3, where n>1) of the filtered peak data (M/Z values, intensities, retention time)
    :param num_decimals_ms_accuracy: int, the number of decimals in accuracy
    :param ppm_threshold: str, the tolerance (in ppm) to consider two M/Z values identical
    :param max_output_mzs: int, max number of mz values to process
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
        if mz_str in processed_mz_strs.keys() and np.isclose(mz_vals[2], processed_mz_strs[mz_str], equal_nan=True):
            unique_mz = False
        else:
            for selected_mz, selected_ret_time in processed_mz_strs.items():
                # compare both retention time and M/Z values to determine uniqueness
                if np.isclose(mz_vals[2], selected_ret_time, equal_nan=True):
                    pmm_err = calc_accuracy_ppm(mz_vals[0], float(selected_mz))
                    if pmm_err < ppm_threshold:
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
                if high_inten_mz_array.shape[0] == max_output_mzs:
                    break
    num_unique_peaks = len(high_inten_mz_array)
    if len(sorted_array) == num_unique_peaks:
        print(f"    All {num_unique_peaks} peaks are unique: their differences in M/Z values are greater than the\n"
              f"        specified ppm tolerance and/or, if retention times are reported, their differences in\n"
              f"        retention times are greater than the specified retention time accuracy.")
    else:
        print(f"    Found {len(high_inten_mz_array)} unique peaks after removing peaks with M/Z values within the "
              f"specified ppm tolerance and\n"
              f"        also within the specified retention time accuracy, if retention time data is to be is to be\n"
              f"        reported. From a group of peaks within these tolerances, only the highest intensity peak is \n"
              f"        reported (without intensity averaging or centroid calculations).")
    return high_inten_mz_array


def get_high_inten_short(ms_level, trimmed_mz_array, max_output_mzs=5):
    """
    return a string w the closest matches to the top 5 intensities, showing their ppm error
    :param ms_level: str, MS level read
    :param trimmed_mz_array: ndarray of peak data, sorted by highest intensity to lowest
    :param max_output_mzs: int, max number of mz values to print
    :return: n/a, prints to stdout
    """
    if "_" in ms_level:
        ms_level = ms_level.replace("_", ", ").replace("p", ".")

    high_inten_str = f"Summary output for MS level {ms_level}: (up to) {max_output_mzs} unique M/Z values with " \
                     f"the highest intensities\n" + MZ_STR_HEADER
    for mz_id, mz_vals in enumerate(trimmed_mz_array):
        if mz_id < max_output_mzs:
            high_inten_str += MZ_STR_FMT.format(mz_vals[0], mz_vals[1], mz_vals[2])
        else:
            break

    return high_inten_str


def get_high_inten_long(ms_level, high_int_peak_str_list, long_output_dict, max_output_mzs=5):
    """
    print out the closest matches to the top 5 intensities, showing their ppm error
    :param ms_level: str, MS level read
    :param high_int_peak_str_list: list, strings of peak data for the highest intensity unique peaks, in descending
        order of intensity
    :param long_output_dict: OrderedDict, keys are strings of peak data, and values are the extended output dicts
    :param max_output_mzs: int, max number of mz values to process
    :return: str
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

    return high_inten_str


def wrangler_main(args, process_file_list):
    """
    Calls to other functions to perform all requested tasks
    :param process_file_list:
    :param args: arguments
    :return: final_str: a string with all of the info that would normally be log outputs
    """

    final_str = ""
    write_mode = 'w'
    ms2_dict = defaultdict(lambda: defaultdict(lambda: None))
    dbe_dict = defaultdict(dict)
    mw_dict = defaultdict(dict)
    blank_data_array_dict = {}
    max_mz_in_stdout = 5
    if args.quit_after_mzml_to_csv:
        max_unique_mz_to_collect = max_mz_in_stdout
    else:
        max_unique_mz_to_collect = 1e9
        # blank file processing
        if args.blank_file_name is not None:
            fname = args.blank_file_name
            fname_lower = os.path.basename(fname).lower()
            blank_data_array_dict = process_blank_file(args)
            for ms_level, ms_array in blank_data_array_dict.items():
                array, my_str = initial_output(fname, fname_lower, ms_array, ms_level, max_unique_mz_to_collect,
                                               max_mz_in_stdout,
                                               args.threshold, args.num_decimals_ms_accuracy, args.ret_time_accuracy,
                                               args.num_decimals_ret_time_accuracy, args.out_dir,
                                               args.quit_after_mzml_to_csv,
                                               args.direct_injection)
                final_str += my_str

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
            trimmed_mz_array, my_str = initial_output(fname, fname_lower, ms_array, ms_level, max_unique_mz_to_collect,
                                                      max_mz_in_stdout, args.threshold, args.num_decimals_ms_accuracy,
                                                      args.ret_time_accuracy, args.num_decimals_ret_time_accuracy,
                                                      args.out_dir, args.quit_after_mzml_to_csv, args.direct_injection)
            final_str += my_str
            if args.quit_after_mzml_to_csv or "blank" in fname_lower:
                # move to next file without further analysis
                continue

            num_matches, matched_formulas, short_output_list, long_output_dict, high_int_peak_str_list = \
                compare_mz_mw(base_fname, trimmed_mz_array, args.threshold, args.omit_mol_ion,
                              deprot_flag, prot_flag, args.non_dom_iso_flag, max_mz_in_stdout)

            write_output(fname, ms_level, num_matches, short_output_list, long_output_dict.values(),
                         matched_formulas, args.combined_output_fname, args.omit_mol_ion, deprot_flag,
                         prot_flag, write_mode, args.out_dir)
            final_str += get_high_inten_long(ms_level, high_int_peak_str_list, long_output_dict)

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

    # Now that finished reading each file, exit (if only saving csv) or graph
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
    return final_str


# a simple page that says hello
@app.route('/hello')
def myhello():
    return render_template('hello.html', name='grace')


@app.route('/')
def index():
    return render_template('index.html')


# This route is expecting a parameter containing the name
# of a file. Then it will locate that file on the upload
# directory and show it on the browser, so if the user uploads
# an image, that image is going to be show after the upload
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOADS_PATH,
                               filename)


# This route is expecting a parameter containing the name
# of a file. Then it will locate that file in the output
# directory and show it on the browser, so if the user uploads
# an image, that image is going to be show after the upload
@app.route('/output/<filename>')
def outputted_file(filename):
    return send_from_directory(OUTPUT_PATH,
                               filename)


@app.route('/download_all')
def download_all(outputs):
    zipf = zipfile.ZipFile('LigninOutput.zip', 'w', zipfile.ZIP_DEFLATED)
    for file in outputs:
        zipf.write(os.path.join(OUTPUT_PATH, file))

    zipf.close()
    return send_file('LigninOutput.zip',
                     mimetype='zip',
                     attachment_filename='LigninOutput.zip',
                     as_attachment=True)


# Route that will process the file upload
@app.route('/upload', methods=['GET', 'POST'])
def upload():
    # Get the name of the uploaded files
    error = None
    silent_remove(UPLOADS_PATH, dir_with_files=True)
    os.mkdir(UPLOADS_PATH)
    silent_remove(OUTPUT_PATH, dir_with_files=True)
    os.mkdir(OUTPUT_PATH)
    uploaded_files = request.files.getlist('file[]')

    direct_injection = False
    non_dom_iso_flag = False
    quit_after_mzml_to_csv = False
    omit_mol_ion = False
    plot_data = False
    unlabeled_csvs = False
    picture_output_flag = False

    ms_accuracy = request.form['ms_accuracy']
    if not ms_accuracy:
        ms_accuracy = 0.0001
    blank_file_name = request.form['blank_file_name']
    combined_output_fname = request.form['combined_output_fname']
    if request.form.get('direct_injection'):
        direct_injection = True
    min_rel_intensity = request.form['min_rel_intensity']
    if not min_rel_intensity:
        min_rel_intensity = 2.
    if request.form.get('non_dom_iso_flag'):
        non_dom_iso_flag = True
    if request.form.get('plot_data'):
        plot_data = True
    if request.form.get('quit_after_mzml_to_csv'):
        quit_after_mzml_to_csv = True
    ret_time_accuracy = request.form['ret_time_accuracy']
    if not ret_time_accuracy:
        ret_time_accuracy = 0.25
    threshold = request.form.get('threshold')
    if not threshold:
        threshold = 10.
    if request.form.get('unlabeled_csvs'):
        unlabeled_csvs = True
    if request.form.get('picture_output_flag'):
        picture_output_flag = True
        print("GOT PIC OUT FLAG")
    if request.form.get('omit_mol_ion'):
        omit_mol_ion = True

    args = MyArgs(ms_accuracy, blank_file_name, combined_output_fname, direct_injection, min_rel_intensity,
                  non_dom_iso_flag, plot_data, quit_after_mzml_to_csv,
                  ret_time_accuracy, threshold, unlabeled_csvs, picture_output_flag, omit_mol_ion)

    filenames = []
    process_file_list = []
    if len(uploaded_files) == 0:
        error = 'No files were picked to be uploaded'
    for file in uploaded_files:
        # Check if the file is one of the allowed types/extensions
        if file and allowed_file(file.filename):
            # Make the filename safe, remove unsupported chars
            filename = secure_filename(file.filename)
            # Move the file form the temporal folder to the upload
            # folder we setup
            full_filename = os.path.join(UPLOADS_PATH, filename)
            process_file_list.append(full_filename)
            file.save(os.path.join(UPLOADS_PATH, filename))
            # Save the filename into a list, we'll use it later
            filenames.append(filename)
            # Redirect the user to the uploaded_file route, which
            # will basically show on the browser the uploaded file
        else:
            error = "file '{file}' does not exist or is not an allowed type."
    error = validate_input(args)
    if error:
        return render_template('error.html', error=error)
    log_output = wrangler_main(args, process_file_list)
    only_files = [f for f in os.listdir(OUTPUT_PATH) if isfile(join(OUTPUT_PATH, f))]
    # this is where I organize the output files into different categories
    matched_csvs = []
    clean_csvs = []
    graphs = []
    drawings_path = join(OUTPUT_PATH, "matched_formula_species")
    mol_drawings = [f for f in os.listdir(drawings_path) if isfile(join(drawings_path, f))]
    for f in only_files:
        if "_matched" in f:
            matched_csvs.append(f)
        if "_clean" in f:
            clean_csvs.append(f)
        if ".png" in f:
            if "_graph" in f or "v_int" in f:
                graphs.append(f)
            else:
                mol_drawings.append(f)

    # Load an html page with a link to each uploaded file
    return render_template('upload.html', filenames=filenames, error=error, matched_csvs=matched_csvs,
                           clean_csvs=clean_csvs, graphs=graphs, mol_drawings=mol_drawings, log_output=log_output)


if __name__ == '__main__':
    app.run()
