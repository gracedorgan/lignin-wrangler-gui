# -*- coding: utf-8 -*-

"""
match_formulas.py
Methods called by ms2molecules to identify molecular structures
"""
from bisect import bisect_left
from collections import OrderedDict
import numpy as np
from lignin_wrangler.lignin_common import (MW_KEYS, M_Z, MAX_SIG_FIGS, FORMULA_DBE_DICT, FORMULA_SMI_DICT,
                                           SMI_NAME_DICT, SOURCE_FILE_NAME, INTENSITY, ISOTOPE_DELTA, MW_DEPROT_KEYS,
                                           MW_PROT_KEYS, MOL_ION_MW_COL, MOL_ION_FORMULA_COL, MOL_ION_SMI_COL,
                                           MOL_ION_NAME_COL, MIN_ERR_FORMULA, MIN_ERR, MIN_ERR_MW, FORMULA_LOOKUP_DICT,
                                           MIN_ERR_MATCH_TYPE, MIN_ERR_DBE, CALC_MW, PPM_ERR, DBE, PARENT_FORMULA,
                                           MATCH_TYPE, RET_TIME, REL_INTENSITY, NEG_ION_MW_COL,
                                           NEG_ION_FORMULA_COL, NEG_ION_SMI_COL, NEG_ION_NAME_COL, POS_ION_MW_COL,
                                           POS_ION_FORMULA_COL, POS_ION_SMI_COL, POS_ION_NAME_COL, MZ_STR_FMT,
                                           LIST_OUTPUT_HEADERS)


def calc_accuracy_ppm(calculated, measured):
    """
    perform the calculation to determine the difference in ppm between a theoretical molecular weight (here,
    calculated from the NIST masses for the most abundant isotopes) compared to a measured value (treating M/Z as
    the measured value, as we are assuming only +1 or -1).
    :param calculated: float, a calculated MW
    :param measured: float, a measured value treated as a molecular weight
    :return: float, signed mass measurement error (accuracy) of the measurement, in ppm, to evaluate if the two
        values represent the molecular masses of the same molecular formula
    """
    return (measured - calculated) / calculated * 1e6


def set_isotope_dict(m_z_val, match_tuple, ext_output_dict, short_output_list, threshold, num_matches, min_error,
                     match_type):
    """
    A helper method for compare_mz_mw. It runs find_mw_matches on match_tuple[1] in order to generate a dictionary entry
    for the mw that this m_z_val is an isotope of. Then, the method goes back in to add stars next to the formula to
    indicate that the mw is for an isotope.
    :param m_z_val: The m_z_val that we have decided is an isotope
    :param match_tuple: The tuple representing the mw that is the base for this isotope
    :param ext_output_dict: The dictionary that will eventually become output
    :param short_output_list:
    :param threshold:
    :param num_matches:
    :param min_error:
    :param match_type:
    :return: short_txt_output_str, num_matches, min_error, this_mw_matches
    """
    num_matches, min_error, this_mw_num_matches = find_mw_matches(MW_KEYS, MOL_ION_MW_COL, MOL_ION_FORMULA_COL,
                                                                  MOL_ION_SMI_COL, MOL_ION_NAME_COL, match_type,
                                                                  ext_output_dict, short_output_list, match_tuple[1],
                                                                  threshold, num_matches, min_error)
    stars = "*" * match_tuple[0]  # number of deltas away from original mw
    match_tuple[0] += 1  # make sure we know how many steps away we are
    formula = ext_output_dict[MIN_ERR_FORMULA]
    ext_output_dict[MIN_ERR_FORMULA] = formula + stars
    ext_output_dict[M_Z] = m_z_val
    ext_output_dict[MIN_ERR] = calc_accuracy_ppm(m_z_val, ext_output_dict[MIN_ERR_MW])
    return num_matches, min_error, this_mw_num_matches


def update_dicts_mw_match(calculated_mw, formula_key, mw_key, name_key, output_dict, ppm_diff,
                          smi_key, match_type):
    """
    A short helper method for matching mz values to formulas. Adds values into the output_dict, looking up values from
    library dictionaries as needed.
    """
    # round the same way made a string before, so things match!
    calc_mw_str = str(round(calculated_mw, MAX_SIG_FIGS))
    matching_formula = FORMULA_LOOKUP_DICT[formula_key][calc_mw_str]
    matching_dbe = FORMULA_DBE_DICT[matching_formula]
    matching_smi_list = FORMULA_SMI_DICT[matching_formula]
    name_list = []
    output_dict[CALC_MW] = calculated_mw
    output_dict[PPM_ERR] = round(ppm_diff, 3)
    output_dict[DBE] = matching_dbe
    output_dict[PARENT_FORMULA] = matching_formula
    output_dict[MATCH_TYPE] = match_type
    output_dict[mw_key].append((calculated_mw, round(ppm_diff, 1)))
    output_dict[formula_key].append((matching_formula, matching_dbe))
    output_dict[smi_key].append(matching_smi_list)
    for smi in matching_smi_list:
        name_list.append(SMI_NAME_DICT[smi])
    output_dict[name_key].append(name_list)


def match_update_dicts(ext_output_dict, formula_col, i, m_z_val, match_type, mol_name_col, mw_col,
                       mw_key_list, short_output_list, smi_col, threshold, min_error):
    calculated_mw = mw_key_list[i]
    ppm_diff = calc_accuracy_ppm(calculated_mw, m_z_val)
    # If even lower error matches are found, the values will be overwritten, as they should be
    if abs(ppm_diff) < abs(min_error):
        min_error = ppm_diff
        calc_mw_str = str(round(calculated_mw, MAX_SIG_FIGS))
        matching_formula = FORMULA_LOOKUP_DICT[formula_col][calc_mw_str]
        ext_output_dict[MIN_ERR] = round(min_error, 3)
        ext_output_dict[MIN_ERR_MW] = calculated_mw
        ext_output_dict[MIN_ERR_MATCH_TYPE] = match_type
        ext_output_dict[MIN_ERR_FORMULA] = FORMULA_LOOKUP_DICT[formula_col][calc_mw_str]
        ext_output_dict[MIN_ERR_DBE] = FORMULA_DBE_DICT[matching_formula]

    # separately check if the match is within the error threshold
    if abs(ppm_diff) > threshold:
        # False indicates no match within threshold
        return False, min_error
    update_dicts_mw_match(calculated_mw, formula_col, mw_col, mol_name_col, ext_output_dict, ppm_diff, smi_col,
                          match_type)
    short_output_list.append(ext_output_dict)
    return True, min_error


def find_mw_matches(mw_key_list, mw_col, formula_col, smi_col, mol_name_col, match_type, ext_output_dict,
                    short_output_list, mz_val, threshold, num_matches, min_error):
    """
    This is where the program finds matches between input M/Z values and the dictionary, and updates str,
        list, and dict that collects output to be displayed
    :param mw_key_list: list of floats; the MWs in the lignin fragment library as floats to allow comparision of
        MWs to a M/Z value. The lists are different for molecular ions, deprotonated ions, and protonated ions.
    :param mw_col: str, used as the output column header and key for dict collecting output
    :param formula_col: str, used as the output column header and key for dict collecting output
    :param smi_col: str, used as the output column header and key for dict collecting output
    :param mol_name_col: str, used as the output column header and key for dict collecting output
    :param match_type: str to designate if the m/z was matched to a the full molecular MW ("same"), deprotonated ("-H"),
        or protonated ("+H")
    :param ext_output_dict: collects the output for the extended output
    :param short_output_list: list to be printed to the short csv output
    :param mz_val: float, input value to potentially match to library MW
    :param threshold: float, the maximum allowed error to consider a MW a match to a M/Z value
    :param num_matches: the number of MW matches within tolerance for input MS data
    :param min_error: float, the signed minimum error found in attempting to match MW to M/Z values
    :return: short_txt_output_str, num_matches updated if any matches were found, new_min_error
    """
    # bisect_left returns the index (stored in close_mw_idx) of the closest value that is greater than the search value
    close_mw_idx = bisect_left(mw_key_list, mz_val)
    this_mw_num_matches = 0
    # check the close_mw_idx to see if the corresponding MW is within tolerance (the threshold). If so, also check
    #     higher indexes until the inaccuracy is greater than the threshold (unlikely but possible, especially if a
    #     high threshold is used
    i = close_mw_idx
    while i < len(mw_key_list):
        match_flag, min_error = match_update_dicts(ext_output_dict, formula_col, i, mz_val, match_type,
                                                   mol_name_col, mw_col, mw_key_list, short_output_list,
                                                   smi_col, threshold, min_error)
        if match_flag:
            this_mw_num_matches += 1
            num_matches += 1
            i += 1
        else:
            break
    # now checking MWs than the match and higher indexes until the inaccuracy is greater than the threshold
    i = close_mw_idx - 1
    while i > 0:
        match_flag, min_error = match_update_dicts(ext_output_dict, formula_col, i, mz_val, match_type,
                                                   mol_name_col, mw_col, mw_key_list, short_output_list,
                                                   smi_col, threshold, min_error)
        if match_flag:
            this_mw_num_matches += 1
            num_matches += 1
            i -= 1
        else:
            break
    return num_matches, min_error, this_mw_num_matches


def check_for_isomers(ext_output_dict, intensity, iso_flag, isotope_parents, m_z_val, min_error, num_matches,
                      short_output_list, this_mw_num_matches, threshold):
    if this_mw_num_matches > 0:  # if this is a "normal" match, add to parent list.
        my_tuple = [1, m_z_val, intensity]
        if my_tuple not in isotope_parents:
            isotope_parents.append(my_tuple)
    else:  # no "normal" matches, but let's see if this is an isotope.
        if iso_flag:
            for match_tuple in reversed(isotope_parents):
                ppm_diff = calc_accuracy_ppm(m_z_val, match_tuple[1] + match_tuple[0] * ISOTOPE_DELTA)
                if abs(ppm_diff) < threshold and intensity < match_tuple[2]:  # this is an isotope
                    # I do this below because the ext_output_dict will be nearly the same for the isotope and
                    # it's parent, besides the *'s, intensity, mz val, and min err
                    num_matches, min_error, this_mw_num_matches = \
                        set_isotope_dict(m_z_val, match_tuple, ext_output_dict, short_output_list, threshold,
                                         num_matches, min_error, "same")
                    break
                elif m_z_val - match_tuple[1] > (match_tuple[0] + 1) * ISOTOPE_DELTA:
                    isotope_parents.remove(match_tuple)
    return min_error, num_matches


def compare_mz_mw(base_source_fname, mz_data_array, threshold, omit_mol_ion_flag, deprot_flag, prot_flag,
                  iso_flag, max_output_mzs):
    """
    Main work of comparing M/Z data to MW in library
    Processes given file name (file location) to look for MW that match M/Z in the CSV file, within the specified ppm
    :param base_source_fname: the file name containing the MS input that is being processed
    :param mz_data_array: ndarray of shape (m, 3) with peak data
    :param threshold: float, threshold in ppm used for determining which, if any, MW are returned for each MW
    :param omit_mol_ion_flag: boolean to indicate if molecular ion matches were not attempted (True) or sought (False)
    :param deprot_flag: boolean, if true, will try to match M/Z to parent molecule MW with 1 fewer H atom
    :param prot_flag: boolean, if true, will try to match M/Z to parent molecule MW with 1 additional H atom
    :param iso_flag: boolean to indicate of non-dominant isomers should be matched
    :param max_output_mzs: int, max number of peaks to be shown in summary output
    :return: multiple outputs, from an int number of MZ-to-formula matches to summary_output_dict
    """
    matched_formulas = set()
    num_matches = 0
    isotope_parents = []
    short_output_list = []
    long_output_dict = OrderedDict()

    # may come sorted by MW, so sort either way, to be sure
    mz_data_array = mz_data_array[np.lexsort((mz_data_array[:, 0], -mz_data_array[:, 1]))]
    high_int_peak_str_list = []
    for mz_id, mz_vals in enumerate(mz_data_array):
        if mz_id == max_output_mzs:
            break
        peak_str = MZ_STR_FMT.format(mz_vals[0], mz_vals[1], mz_vals[2])
        high_int_peak_str_list.append(peak_str)

    # resort by so overall descending by MZ
    if np.isnan(mz_data_array[0][2]):
        mz_data_array = mz_data_array[mz_data_array[:, 0].argsort()]
    else:
        mz_data_array = mz_data_array[np.lexsort((-mz_data_array[:, 1], mz_data_array[:, 0]))]

    for mz_vals in mz_data_array:
        mz_val = mz_vals[0]
        intensity = mz_vals[1]
        ret_time = mz_vals[2]
        # Todo, add rel_intensity calculation after a standard is chosen
        rel_intensity = np.nan
        # holders for minimum error matches (start with a very high number and it will be replaced below)
        min_error = 1e6
        ext_output_dict = {SOURCE_FILE_NAME: base_source_fname, M_Z: mz_val, INTENSITY: intensity, RET_TIME: ret_time,
                           REL_INTENSITY: rel_intensity, CALC_MW: 0, PPM_ERR: 0, PARENT_FORMULA: "", DBE: 0,
                           MATCH_TYPE: "",
                           MOL_ION_MW_COL: [], MOL_ION_FORMULA_COL: [], MOL_ION_SMI_COL: [], MOL_ION_NAME_COL: [],
                           NEG_ION_MW_COL: [], NEG_ION_FORMULA_COL: [], NEG_ION_SMI_COL: [], NEG_ION_NAME_COL: [],
                           POS_ION_MW_COL: [], POS_ION_FORMULA_COL: [], POS_ION_SMI_COL: [], POS_ION_NAME_COL: []}
        if not omit_mol_ion_flag:
            # Find matches for molecular ion
            num_matches, min_error, this_mw_num_matches = \
                find_mw_matches(MW_KEYS, MOL_ION_MW_COL, MOL_ION_FORMULA_COL, MOL_ION_SMI_COL, MOL_ION_NAME_COL, "same",
                                ext_output_dict, short_output_list, mz_val, threshold, num_matches, min_error)
            min_error, num_matches = check_for_isomers(ext_output_dict, intensity, iso_flag, isotope_parents,
                                                       mz_val, min_error, num_matches, short_output_list,
                                                       this_mw_num_matches, threshold)
        if deprot_flag:
            # same for deprotonated ion
            num_matches, min_error, this_mw_num_matches = \
                find_mw_matches(MW_DEPROT_KEYS, NEG_ION_MW_COL, NEG_ION_FORMULA_COL, NEG_ION_SMI_COL, NEG_ION_NAME_COL,
                                "-H", ext_output_dict, short_output_list, mz_val, threshold, num_matches, min_error)

            min_error, num_matches = check_for_isomers(ext_output_dict, intensity, iso_flag, isotope_parents,
                                                       mz_val, min_error, num_matches, short_output_list,
                                                       this_mw_num_matches, threshold)
        if prot_flag:
            # and then protonated ion
            num_matches, min_error, this_mw_num_matches = \
                find_mw_matches(MW_PROT_KEYS, POS_ION_MW_COL, POS_ION_FORMULA_COL, POS_ION_SMI_COL, POS_ION_NAME_COL,
                                "+H", ext_output_dict, short_output_list, mz_val, threshold, num_matches, min_error)
            min_error, num_matches = check_for_isomers(ext_output_dict, intensity, iso_flag, isotope_parents,
                                                       mz_val, min_error, num_matches, short_output_list,
                                                       this_mw_num_matches, threshold)

        # for prettier printing, remove brackets from strings of lists
        for dict_key in LIST_OUTPUT_HEADERS:
            ext_output_dict[dict_key] = str(ext_output_dict[dict_key])[1:-1]

        # save top intensity data for summary output
        peak_str = MZ_STR_FMT.format(mz_vals[0], mz_vals[1], mz_vals[2])
        # save all data for extended output
        long_output_dict[peak_str] = ext_output_dict
        # to prevent printing the same grid with and without a *
        if "*" not in ext_output_dict[MIN_ERR_FORMULA]:
            matched_formulas.add(ext_output_dict[MIN_ERR_FORMULA])
    return num_matches, matched_formulas, short_output_list, long_output_dict, high_int_peak_str_list
