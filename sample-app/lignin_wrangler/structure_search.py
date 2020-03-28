# -*- coding: utf-8 -*-

"""
structure_search.py
Methods called by ms2molecules to identify molecular structures
"""

import os
import re
import warnings
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import variation, skew, kurtosis
from lignin_wrangler.match_formulas import calc_accuracy_ppm
from rdkit import Chem
from common_wrangler.common import (create_out_fname, warning, parse_stoich)
from lignin_wrangler.create_library import make_image_grid
from lignin_wrangler.lignin_common import (M_Z, FORMULA_SMI_DICT, INTENSITY, GRAPH_SUFFIX, MIN_ERR_FORMULA,
                                           MIN_ERR_DBE, MIN_ERR, PARENT_FORMULA, PARENT_MZ, MZ_STR_FMT,
                                           PARENT_MATCH_ERR, ION_ENERGIES, AVG_DBE, AVG_MZ,
                                           DAUGHTER_OUTPUT, DAUGHTER_MZ_ARRAY, ION_MZ_DICT, OUTSIDE_THRESH_MATCHES)

plt.style.use('seaborn-whitegrid')


def make_dbe_mw_graphs(fkey, ion_energies_dict, out_dir=None):
    """
    makes and saves a graph of the bde value vs fragmentation energy for each set
    of qualifying files. the file the graph is saved to will be the fkey+_dbe_graph.png.
    :param fkey: str, used to designate sets of MS2 data
    :param ion_energies_dict: dict with data used for parent structure analysis, including average MW and DBEs
    :param out_dir: None if default output location is to be used
    :return: nothing
    """
    energy_levels = sorted(list(ion_energies_dict.keys()))
    dbe_list = []
    dbe_dev = []
    dbe_var = []
    dbe_skew = []
    dbe_kurt = []
    mz_list = []
    mz_dev = []
    mz_var = []
    mz_skew = []
    mz_kurt = []
    for energy_level in energy_levels:
        weighted_avg_dbe, std_dev_dbe, variation_dbe, skew_dbe, kurtosis_dbe = ion_energies_dict[energy_level][AVG_DBE]
        dbe_list.append(weighted_avg_dbe)
        dbe_dev.append(std_dev_dbe)
        dbe_var.append(variation_dbe)
        dbe_skew.append(skew_dbe)
        dbe_kurt.append(kurtosis_dbe)
        weighted_avg_mz, std_dev_mz, variation_mz, skew_mz, kurtosis_mz = ion_energies_dict[energy_level][AVG_MZ]
        mz_list.append(weighted_avg_mz)
        mz_dev.append(std_dev_mz)
        mz_var.append(variation_mz)
        mz_skew.append(skew_mz)
        mz_kurt.append(kurtosis_mz)

    out_filename = create_out_fname(fkey, suffix=GRAPH_SUFFIX, base_dir=out_dir, ext="png")
    fig = plt.figure(figsize=(9, 12))
    # The add_subplot sometimes throws a warning that we want to ignore
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        ax1 = fig.add_subplot(411)
        ax2 = fig.add_subplot(413)
        ax3 = fig.add_subplot(412)
        ax4 = fig.add_subplot(414)
        # ax1.plot(energy_levels, dbe_list, 'or-')
    ax1.errorbar(energy_levels, dbe_list, yerr=dbe_dev, fmt='or-')
    ax1.set_title('DBE vs. Fragmentation Energy')
    ax1.set_xlabel('Fragmentation Energy')
    ax1.set_ylabel('Double Bond Equivalent')

    ax3.plot(energy_levels, dbe_var, 'b', label="variance")
    ax3.plot(energy_levels, dbe_skew, 'g', label="skew")
    ax3.plot(energy_levels, dbe_kurt, 'r', label="kurtosis")
    ax3.legend(loc=0)
    ax3.set_title('DBE Statistics vs. Fragmentation Energy')
    ax3.set_xlabel('Fragmentation Energy')
    ax3.set_ylabel('Property value')

    # now mw
    ax2.errorbar(energy_levels, mz_list, yerr=mz_dev, fmt='ob-')
    ax2.set_title('Weighted Average M/Z vs. Fragmentation Energy')
    ax2.set_xlabel('Fragmentation Energy')
    ax2.set_ylabel('Weighted Average M/Z')
    ax4.plot(energy_levels, mz_var, 'b', label="variance")
    ax4.plot(energy_levels, mz_skew, 'g', label="skew")
    ax4.plot(energy_levels, mz_kurt, 'r', label="kurtosis")
    ax4.legend(loc=0)
    ax4.set_title('M/Z Statistics vs. Fragmentation Energy')
    ax4.set_xlabel('Fragmentation Energy')
    ax4.set_ylabel('Property value')

    fig.tight_layout()
    fig.savefig(out_filename)
    print(f"Wrote file: {os.path.relpath(out_filename)}")
    plt.close()


def find_substructure_sets(fkey, fkey_ms2_dict, mz_threshold, num_decimals_ms_accuracy, out_dir):
    """
    f"If this option is chosen and there is MS2 data available, the program will use the "
    f"MS2 precursor M/Z value to identify the closest molecular formula and molecules in the "
    f"lignin library with that molecular formula. Then, all subformulas from MS2 data will "
    f"be analyzed to see if they are potential , and then search "
    f"the lignin library to find potential  search "
    f"for substructures, drawn pictures of "
    f"the compounds and their substructures will be output into the output "
    f"directory. To help parent compound identification, a file will be generated for each "
    f"identified potential parent and its substructures with molecular weights corresponding "
    f"to peaks from MS2 output. The file names will be labeled with the potential parent "
    f"molecule SMILES string."
    :param fkey:
    :param fkey_ms2_dict:
    :param mz_threshold: float, the maximum error allowed between M/Z values to consider them identical
    :param num_decimals_ms_accuracy:
    :param out_dir:
    :return:
    """
    parent_formula = fkey_ms2_dict[PARENT_FORMULA]
    parent_mz = fkey_ms2_dict[PARENT_MZ]

    # daughter output is not unique. the trimmed mz array includes matched not within tolerance (longer).
    # iterate through the daughter output as it is shorter.
    daughter_long_out_dict = fkey_ms2_dict[DAUGHTER_OUTPUT]
    daughter_keys = sorted(list(daughter_long_out_dict.keys()))
    if len(daughter_keys) == 0:
        warning(f"No potential daughter molecular formulas found for file set: {fkey}")
        return
    # start new list of unique M/Zs
    unique_daughter_out_list_dicts = [daughter_long_out_dict[daughter_keys[0]]]
    old_mz_val = daughter_long_out_dict[daughter_keys[0]][M_Z]
    old_intensity = daughter_long_out_dict[daughter_keys[0]][INTENSITY]
    print("Unique daughter peaks matching to a formula with the specified threshold:\n"
          " M/Z      Intensity Formula   PPM error")
    for daughter_key in daughter_keys[1:]:
        daughter_dict = daughter_long_out_dict[daughter_key]
        # skip isotopes
        if "*" in daughter_dict[MIN_ERR_FORMULA]:
            continue
        # switched to threshold for comparision, instead of ms_accuracy
        error_between_mz_values = calc_accuracy_ppm(daughter_dict[M_Z], old_mz_val)
        # if abs(daughter_dict[M_Z] - old_mz_val) < ms_accuracy:
        if abs(error_between_mz_values) < mz_threshold:
            if daughter_dict[INTENSITY] > old_intensity:
                unique_daughter_out_list_dicts[-1] = daughter_dict
                old_mz_val = daughter_dict[M_Z]
                old_intensity = daughter_dict[INTENSITY]
        else:
            # print the law row--prevents writing intermediate intensity before final (max) intensity found
            print(f"{old_mz_val:9.{num_decimals_ms_accuracy}f} {old_intensity:9.0f} "
                  f"{unique_daughter_out_list_dicts[-1][MIN_ERR_FORMULA]:>9} "
                  f"{unique_daughter_out_list_dicts[-1][MIN_ERR]:9.1f}")
            unique_daughter_out_list_dicts.append(daughter_dict)
            old_mz_val = daughter_dict[M_Z]
            old_intensity = daughter_dict[INTENSITY]
    # don't miss printing the last entry
    print(f"{old_mz_val:9.{num_decimals_ms_accuracy}f} {old_intensity:9.0f} "
          f"{unique_daughter_out_list_dicts[-1][MIN_ERR_FORMULA]:>9} "
          f"{unique_daughter_out_list_dicts[-1][MIN_ERR]:9.1f}")

    matched_outside_thresh_dict = fkey_ms2_dict[OUTSIDE_THRESH_MATCHES]
    daughter_keys = sorted(list(matched_outside_thresh_dict.keys()))
    if len(daughter_keys) > 0:
        print("\nUnique daughter peaks *not* matching to a formula with the specified threshold:\n"
              " M/Z      Intensity Formula   PPM error")
        last_out_dict = matched_outside_thresh_dict[daughter_keys[0]]
        last_out_mz = last_out_dict[M_Z]
        last_out_inten = last_out_dict[INTENSITY]
        for out_thresh_key in daughter_keys[1:]:
            out_thresh_dict = matched_outside_thresh_dict[out_thresh_key]
            error_between_mz_values = calc_accuracy_ppm(out_thresh_dict[M_Z], last_out_mz)
            if abs(error_between_mz_values) < mz_threshold:
                if out_thresh_dict[INTENSITY] > last_out_inten:
                    last_out_dict = out_thresh_dict
                    last_out_mz = last_out_dict[M_Z]
                    last_out_inten = last_out_dict[INTENSITY]
            else:
                print(f"{last_out_mz:9.{num_decimals_ms_accuracy}f} {last_out_inten:9.0f} "
                      f"{last_out_dict[MIN_ERR_FORMULA]:>9} {last_out_dict[MIN_ERR]:9.1f}")
                last_out_dict = out_thresh_dict
                last_out_mz = last_out_dict[M_Z]
                last_out_inten = last_out_dict[INTENSITY]
        print(f"{last_out_mz:9.{num_decimals_ms_accuracy}f} {last_out_inten:9.0f} "
              f"{last_out_dict[MIN_ERR_FORMULA]:>9} {last_out_dict[MIN_ERR]:9.1f}")

    num_unique_mz_within_tol = len(unique_daughter_out_list_dicts)
    subformula_list = remove_impossible_formulas(parent_formula, unique_daughter_out_list_dicts)
    num_possible_formulas = len(subformula_list)
    num_removed_formulas = num_unique_mz_within_tol - num_possible_formulas
    print(f"\nFound {num_unique_mz_within_tol} unique peaks that match a MW in the lignin library (within tolerance) "
          f"from all provided MS2\n    data, of which {num_removed_formulas} were determined to not be possible "
          f"daughter molecular formulas due to having more\n    element atoms than in the parent.")
    substructure_dict = get_all_substructures(parent_formula, subformula_list, num_decimals_ms_accuracy)
    if substructure_dict is not None:
        output_substructures(fkey, num_possible_formulas, substructure_dict,
                             f"{parent_mz:.{num_decimals_ms_accuracy}f}", out_dir)


def remove_impossible_formulas(parent_molecular_formula, long_output_list):
    """
    makes a list of formulas we have matched to peaks, but without the formulas that are not sub-formulas
    of the parent ("root") formula, or the formula that is originally to be fragmented.
    :param parent_molecular_formula: str, the molecular formula of the parent M/Z value
    :param long_output_list: a list of dictionaries for each peak in ms data
    :return: a list of dictionaries corresponding to the peaks of ms data without the formulas that are impossible
    to get from this fragmentation
    """
    subformula_list = []
    # get the parent stoich here so it does not need to be found for each sub formula
    parent_stoich = parse_stoich(parent_molecular_formula)
    for peak_dict in long_output_list:
        if is_sub_formula(parent_stoich, peak_dict[MIN_ERR_FORMULA]):
            subformula_list.append(peak_dict)
    return subformula_list


def is_sub_formula(parent_stoich_dict, pot_sub_formula):
    """
    A small method that returns true if pot_sub_formula is a sub formula of formula. That means all elements
    in the subformula must also be in the formula, and there must be lesser or equal quantities of each
    element in the subformula.
    :param parent_stoich_dict: dict, has the stoichiometry of the parent molecule (keys are element types, values
        are the number of that element type
    :param pot_sub_formula: str, the possible sub (daughter) formula
    :return: True if second formula is a subformula of the first. False otherwise.
    """
    pot_sub_formula_dict = parse_stoich(pot_sub_formula)
    # could have sub formulas without atom types in the parent (e.g. CH4 can break off a molecule that has an oxygen)
    # cannot have a sub formula if the potential sub formula has element types that aren't in the parent, or more of
    #     an element type than in the parent
    for element in pot_sub_formula_dict:
        if element not in parent_stoich_dict:
            return False
        if parent_stoich_dict[element] < pot_sub_formula_dict[element]:
            return False
    return True


def get_all_substructures(parent_molecular_formula, subformula_list, num_decimals_ms_accuracy):
    """
    This method takes in a long_output_list of dicts that represent formulas that
    are matched with molecular weights in the dataset. It will return a dictionary of dictionaries
    with all of the SMILE strings that qualify as substructures for each potential root.
    :param parent_molecular_formula: str, the molecular formula of the parent M/Z value
    :param subformula_list: a list of subformulas of the parent that matched M/Z values
    :param num_decimals_ms_accuracy: int, number of decimal points in MS accuracy, for rounding
    :return: a dict of dictionaries containing all of the substructure matches for all of
    the potential roots
    """
    substructure_dict = {}
    found_substructure = False
    for pot_parent_smi in FORMULA_SMI_DICT[parent_molecular_formula]:
        substruct_smis, substruct_mzs, num_mz_matches = \
            get_substructs(pot_parent_smi, subformula_list, parent_molecular_formula, num_decimals_ms_accuracy)
        if len(substruct_smis) > 0:
            substructure_dict[pot_parent_smi] = (substruct_smis, substruct_mzs, num_mz_matches)
            found_substructure = True
        else:
            print(f"No daughter substructures found for potential parent: {pot_parent_smi}")
    if found_substructure:
        return substructure_dict
    else:
        print(f"No daughter substructures found for any potential parent in our library.")
        return None


def output_substructures(fkey, num_possible_formulas, all_substructures, parent_mz_str, out_dir):
    print(f"Substructure matching for {fkey} from the {num_possible_formulas} possible daughter molecule formulas:")
    print_len = len(str(num_possible_formulas))
    for parent_index, potential_parent in enumerate(all_substructures):
        parent_label = f"Parent, {parent_mz_str}"
        parent_num = parent_index + 1
        sub_smis = all_substructures[potential_parent][0]
        sub_mzs = all_substructures[potential_parent][1]
        num_mz_matches = all_substructures[potential_parent][2]
        print(f"     Found at least 1 substructure for {num_mz_matches:{print_len}}/{num_possible_formulas} molecular "
              f"formulas for potential parent {parent_num}: {potential_parent}")
        base_name = f"{fkey}_parent{parent_num}_substructs.png"
        make_image_grid(base_name, [potential_parent] + sub_smis, labels=[parent_label] + sub_mzs,
                        out_dir=out_dir)


# todo: rearrangements
def get_substructs(parent_smi, list_dicts, parent_formula, num_decimals_ms_accuracy):
    """
    This method takes a SMILE string that we are using as the root_smi, and it gets
    the number of SMILE strings that are a substruct match
    :param parent_smi: the SMILE string of the root we are checking
    :param list_dicts: a list of dictionaries representing compounds that match to MWs in the data
    :param parent_formula: original formula of the root we are checking
    :param num_decimals_ms_accuracy: int, number of decimal points in MS accuracy, for rounding
    :return: a list of SMILE strings which are substructs, and a list of the mws of each struct
    """
    substruct_smis = []
    substruct_mzs = []
    root_mol = Chem.MolFromSmiles(parent_smi)
    num_mz_matches = 0
    for match_dict in list_dicts:
        if match_dict[MIN_ERR_FORMULA] != parent_formula:
            add_to_match_num = 0
            sub_formula = match_dict[MIN_ERR_FORMULA].replace("*", "")
            sub_formula_mz = match_dict[M_Z]
            for smi in FORMULA_SMI_DICT[sub_formula]:
                sub_mol = Chem.MolFromSmiles(smi)
                if root_mol.HasSubstructMatch(sub_mol):
                    substruct_smis.append(smi)
                    substruct_mzs.append(f"{sub_formula_mz:.{num_decimals_ms_accuracy}f}")
                    add_to_match_num = 1
            num_mz_matches += add_to_match_num
    return substruct_smis, substruct_mzs, num_mz_matches


def make_fkey(fname_lower):
    """
    This function creates a key that can be used in the dbe_dict, my removing the path and extension, keeps what
    remains as the key
    :param fname_lower: a valid filename for ms data (without location, lower case)
    :return: a string without the original fnames' extension or HCD's trailing digits; blank string if no HCD+number hit
    """
    lower_base_name = os.path.splitext(fname_lower)[0]
    hcd_match = re.search(r'hcd(\d+)', lower_base_name)
    if hcd_match:
        # hcd_match.group(1) to get number only, but that can cause problems with the splitting
        # (e.g. 0 may show up elsewhere, too) so adding hcd back with the join
        matched_text = hcd_match.group(0)
        energy_level = int(hcd_match.group(1))
        str_list = lower_base_name.split(matched_text)
        return "hcd".join(str_list), energy_level
    else:
        return "", None


def get_dbe_weighted_average(long_output_list):
    """
    calculates the weighted average dbe for all peaks in the long_output_list
    :param long_output_list: a list of dictionaries for every peak in the data
    :return: the weighted average dbe calculation for the top 5 peaks in this file
    """
    intensity_list = []
    dbe_list = []
    for t_dict in long_output_list:
        intensity_list.append(t_dict[INTENSITY])
        dbe_list.append(t_dict[MIN_ERR_DBE])
    return get_weight_avg_stats(dbe_list, intensity_list)


def get_weight_avg_stats(value_list, weights_list):
    # making a weighted DBE list instead of using the option in np.average to allow calculations of other stats
    tot_weights = sum(weights_list)
    weighted_values = []
    if len(value_list) < 2:
        return np.nan, np.nan, np.nan, np.nan, np.nan
    for value, weight in zip(value_list, weights_list):
        weighted_values.append(value * weight / tot_weights)
    weighted_values = np.asarray(weighted_values)
    weighted_avg = sum(weighted_values)
    std_dev = np.std(weighted_values)
    value_variation = variation(weighted_values)
    value_skew = skew(weighted_values)
    value_kurtosis = kurtosis(weighted_values)
    return weighted_avg, std_dev, value_variation, value_skew, value_kurtosis


def make_ms2_dict(fname_lower, ms_level, ms2_dict, trimmed_mz_array, long_output_dict, ppm_threshold, ms_accuracy,
                  num_decimals_ms_accuracy):
    """
    goes through a list of files and creates dicts storing data from ms2 files to be used to help identify parent
    molecules
    # mw_dict: mol. weight weighted average of ALL peaks/energy to frag energy
    # dbe_dict: all matches dbe weighted average/energy
    :param fname_lower: str, file name to process without location and all lowercase
    :param ms_level: str, if ms2 will have the precursor M/Z
    :param trimmed_mz_array: ndarray, n x 3 (where n is number of unique mz values), sorted by MZ
    :param ms2_dict: a dictionary mapping "fkeys" to a dictionary or file names to dbe values, average mw values, etc.
    :param long_output_dict: dict of dicts of calculated values from fname (mz_values_str is main key)
    :param ppm_threshold: float, in ppm max error to consider a match
    :param ms_accuracy: float, the accuracy for the difference in M/Z values to consider them identical
    :param num_decimals_ms_accuracy: int, the number of decimals for difference in M/Z values to consider them identical
    :return: n/a, updates dbe_dict and mw_dict
    """
    fkey, energy_level = make_fkey(fname_lower)
    # making sure f key exists
    if not fkey:
        return
    if not ms2_dict[fkey][ION_ENERGIES]:
        ms2_dict[fkey][ION_ENERGIES] = defaultdict(dict)

    if not ms2_dict[fkey][ION_MZ_DICT]:
        ms2_dict[fkey][ION_MZ_DICT] = {}

    if not ms2_dict[fkey][OUTSIDE_THRESH_MATCHES]:
        ms2_dict[fkey][OUTSIDE_THRESH_MATCHES] = {}

    if energy_level == 0:
        find_parent_formula(fkey, long_output_dict, ms2_dict, ms_level, trimmed_mz_array, num_decimals_ms_accuracy)

    output_within_err_dict = {}
    for mz_str, mz_dict in long_output_dict.items():
        # account for error in holding floating point numbers by adding a little bit to the threshold before evaluation
        if abs(mz_dict[MIN_ERR]) <= ppm_threshold * (1. + ms_accuracy):
            output_within_err_dict[mz_str] = mz_dict
        else:
            if mz_str in ms2_dict[fkey][OUTSIDE_THRESH_MATCHES]:
                if mz_dict[INTENSITY] > ms2_dict[fkey][OUTSIDE_THRESH_MATCHES][mz_str][INTENSITY]:
                    ms2_dict[fkey][OUTSIDE_THRESH_MATCHES][mz_str] = mz_dict
            else:
                ms2_dict[fkey][OUTSIDE_THRESH_MATCHES][mz_str] = mz_dict

    # get the weighted average to eventually make the dbe graph
    # it is possible that there are no matches; then the AVG_DBE will remain nan
    try:
        # for DBE, only use formulas for MW within tolerance of M/Z
        ms2_dict[fkey][ION_ENERGIES][energy_level][AVG_DBE] = get_dbe_weighted_average(output_within_err_dict.values())
    except ZeroDivisionError:
        # warning(f"There were no formula matches within the specified tolerance of {ppm_threshold} ppm for file: "
        warning(f"There were no formula matches with the specified threshold for file: {fname_lower}\n"
                f"    The program will continue without calculating a weighted-average DBE value from this file.")

    # get the weighted average to eventually make the MZ graph
    ms2_dict[fkey][ION_ENERGIES][energy_level][AVG_MZ] = get_weight_avg_stats(trimmed_mz_array[:, 0],
                                                                              trimmed_mz_array[:, 1])
    ms2_dict[fkey][ION_MZ_DICT][energy_level] = trimmed_mz_array

    # gather daughter data
    if energy_level != 0:
        if ms2_dict[fkey][DAUGHTER_MZ_ARRAY] is None:
            ms2_dict[fkey][DAUGHTER_MZ_ARRAY] = trimmed_mz_array
            ms2_dict[fkey][DAUGHTER_OUTPUT] = output_within_err_dict
        else:
            # remove duplicates later
            ms2_dict[fkey][DAUGHTER_MZ_ARRAY] = np.concatenate((ms2_dict[fkey][DAUGHTER_MZ_ARRAY], trimmed_mz_array),
                                                               axis=0)
            ms2_dict[fkey][DAUGHTER_OUTPUT].update(output_within_err_dict)


def find_parent_formula(fkey, long_output_dict, ms2_dict, ms_level, trimmed_mz_array, num_decimals_ms_accuracy):
    """
    Given the precursor MZ read from the MS level, find the closest matching peak and its closest matching molecular
    formula. Accounts for the potential for there to be more than one set of peak data for the MZ match (if there is
    retention data). If that happens, the peak with the highest intensity is selected (note: this is only needed to
    output the intensity for that match).
    :param fkey:
    :param long_output_dict:
    :param ms2_dict:
    :param ms_level:
    :param trimmed_mz_array:
    :param num_decimals_ms_accuracy: int, number of decimal points in MS accuracy, for rounding
    :return:
    """
    if 'mz' in ms_level:
        precursor_mz = float(ms_level.split('mz')[1].replace('p', '.'))
        lowest_ppm_error = 1e6
        best_match_intensity = 0
        best_match_peak_str = ""
        # do a loop because parent mz lists should be short
        for mz_vals in trimmed_mz_array:
            # before testing if this is a good match for the parent, make sure it is not an isotope
            current_peak_str = MZ_STR_FMT.format(mz_vals[0], mz_vals[1], mz_vals[2])
            current_output_dict = long_output_dict[current_peak_str]
            # do not use an isotope as a parent. There should be no isotopes if there is not also a parent, so move on
            if "*" in current_output_dict[MIN_ERR_FORMULA]:
                continue
            ppm_error = round(calc_accuracy_ppm(mz_vals[0], precursor_mz), 1)
            if abs(ppm_error) <= abs(lowest_ppm_error):
                # this means we have this m/z, but maybe not the highest intensity
                if np.isclose(ppm_error, lowest_ppm_error):
                    if int(mz_vals[1]) > best_match_intensity:
                        best_match_intensity = int(mz_vals[1])
                        best_match_peak_str = current_peak_str
                else:
                    ms2_dict[fkey][PARENT_MZ] = mz_vals[0]
                    lowest_ppm_error = ppm_error
                    best_match_intensity = int(mz_vals[1])
                    best_match_peak_str = current_peak_str
        print(f"\nBased on the MS2 0 ionization energy file: the precursor M/Z value is "
              f"{precursor_mz:.{num_decimals_ms_accuracy}f}.\n"
              f"    Will use the closest spectra peak M/Z value as te parent peak.")
    else:
        # if not a specified precursor, grab the highest intensity peak (rather than the closest)
        # since this is sorted, that will be the first one
        print("\nDid not find a precursor M/Z value in the file name. The precursor will be assumed to be the\n"
              "    highest intensity peak from MS2 with 0 ionization energy.")
        best_match_peak_str = list(long_output_dict.keys())[0]
        parent_output_dict = long_output_dict[best_match_peak_str]
        precursor_mz = parent_output_dict[M_Z]
        ms2_dict[fkey][PARENT_MZ] = precursor_mz

    parent_output_dict = long_output_dict[best_match_peak_str]
    ms2_dict[fkey][PARENT_FORMULA] = parent_output_dict[MIN_ERR_FORMULA]
    ms2_dict[fkey][PARENT_MATCH_ERR] = parent_output_dict[MIN_ERR]
    print(f"Using M/Z value {ms2_dict[fkey][PARENT_MZ]:.{num_decimals_ms_accuracy}f} as the parent peak, which matches "
          f"to {ms2_dict[fkey][PARENT_FORMULA]} with {ms2_dict[fkey][PARENT_MATCH_ERR]:.1f} ppm error. This molecular\n"
          f"    formula will be used as the parent formula.")
