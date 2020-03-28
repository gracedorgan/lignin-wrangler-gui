#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Launches steps required to build lignin
Multiple output options, from tcl files to plots
"""
import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from collections import (defaultdict)
from configparser import ConfigParser
from common_wrangler.common import (MAIN_SEC, GOOD_RET, INPUT_ERROR, KB, H, KCAL_MOL_TO_J_PART,
                                    INVALID_DATA, OUT_DIR, InvalidDataError, warning, process_cfg, make_dir,
                                    create_out_fname, str_to_file, round_sig_figs, write_csv)
from rdkit.Chem import (MolToSmiles, MolFromMolBlock, Kekulize)
# from rdkit.Chem import (MolToSmiles, MolFromMolBlock, Kekulize, Mol)
from rdkit.Chem.AllChem import (Compute2DCoords, EmbedMolecule, MMFFOptimizeMolecule)
from rdkit.Chem.Draw import MolToFile
from rdkit.Chem.rdMolInterchange import MolToJSON
from rdkit.Chem.rdmolfiles import MolToPDBBlock
from rdkit.Chem.rdmolops import AddHs
from lignin_wrangler import __version__
from lignin_wrangler.kmc_common import (Event, Monomer, E_BARRIER_KCAL_MOL, E_BARRIER_J_PART, TEMP, INI_MONOS,
                                        MAX_MONOS, SIM_TIME, GROW, DEF_E_BARRIER_KCAL_MOL, OX, MONOMER, OLIGOMER,
                                        ADJ_MATRIX, RANDOM_SEED, S, G, C, CHAIN_LEN, BONDS, RCF_YIELDS, RCF_BONDS,
                                        MAX_NUM_DECIMAL, MONO_LIST, CHAIN_MONOS, CHAIN_BRANCH_COEFF, RCF_BRANCH_COEFF,
                                        CHAIN_ID, DEF_CHAIN_ID, PSF_FNAME, DEF_PSF_FNAME, DEF_TOPPAR, TOPPAR_DIR,
                                        DEF_RXN_RATES, BOND_TYPE_LIST, INT_TO_TYPE_DICT, TIME, N_FINAL)
from lignin_wrangler.kmc_functions import (run_kmc, generate_mol, gen_tcl, count_bonds,
                                           count_oligomer_yields, analyze_adj_matrix)


# Config keys #
CONFIG_KEY = 'config_key'
OUT_FORMAT_LIST = 'output_format_list'
BASENAME = 'outfile_basename'
IMAGE_SIZE = 'image_size'
SAVE_CSV = 'csv'
SAVE_JSON = 'json'
SAVE_PDB = 'pdb'
SAVE_PNG = 'png'
SAVE_SMI = 'smi'
SAVE_SVG = 'svg'
SAVE_TCL = 'tcl'

OUT_TYPE_LIST = [SAVE_CSV, SAVE_JSON, SAVE_PDB, SAVE_PNG,  SAVE_SMI, SAVE_SVG, SAVE_TCL]
OUT_TYPE_STR = "', '".join(OUT_TYPE_LIST)
SAVE_FILES = 'save_files_boolean'
ADD_RATES = 'add_rates_list'
RXN_RATES = 'reaction_rates_at_298K'
SG_RATIOS = 'sg_ratio_list'
C_LIGNIN = 'c_lignin_flag'
NUM_REPEATS = 'num_repeats'
DYNAMICS = 'dynamics_flag'
OLIGOMERS = 'oligomers'
MONOMERS = 'monomers'
PLOT_BONDS = 'plot_bonds'
SUPPRESS_SMI = 'suppress_smi_output'
BREAK_CO = 'break_co_bonds'

PLOT_COLORS = [(0, 0, 0), (1, 0, 0), (0, 0, 1), (0, 0.6, 0), (0.6, 0, 0.6), (1, 0.549, 0),
               (0, 0.6, 0.6), (1, 0.8, 0), (0.6078, 0.2980, 0), (0.6, 0, 0), (0, 0, 0.6)]

ADD_RATE = "Add rate (monomers/s)"
SG_RATIO = "S:G Ratio"
N_INI = "Initial Num Monomers"
RCF_MONOS = "RCF Monomers"
IND_SUM_HEADERS = [ADD_RATE, SG_RATIO, N_INI, N_FINAL, RCF_MONOS] + BOND_TYPE_LIST
RCF_MONOS_AVG = "RCF Monomers Avg"
RCF_MONOS_STD = "RCF Monomers Std Dev"
AVG = " Avg"
STD = " Std Dev"
REPEATS_HEADERS = [ADD_RATE, SG_RATIO, N_INI, N_FINAL, RCF_MONOS_AVG, RCF_MONOS_STD]
for stat in [AVG, STD]:
    for b_type in BOND_TYPE_LIST:
        REPEATS_HEADERS.append(b_type + stat)

# Defaults #
DEF_TEMP = 298.15  # K
DEF_MAX_MONOS = 10  # number of monomers
DEF_SIM_TIME = 3600  # simulation time in seconds (1 hour)
DEF_SG = 1
DEF_INI_MONOS = 2
# Estimated addition rate below is based on: https://www.pnas.org/content/early/2019/10/25/1904643116.abstract
#     p. 23121, col2, "0.1 fmol s^-1" for 100micro-m2 surface area; estimated area for lignin modeling is 100nm^2
#     thus 0.1 fmols/ second * 1.00E-15 mol/fmol * 6.022E+23 particles/mol  * 100 nm^2/100micron^2 = 6 monomers/s
#     as an upper limit--rounded down to 1.0 monomers/s--this is just an estimate
DEF_ADD_RATE = 1.0
DEF_IMAGE_SIZE = (1200, 300)
DEF_BASENAME = 'lignin-kmc-out'
DEF_NUM_REPEATS = 1

DEF_VAL = 'default_value'
DEF_CFG_VALS = {OUT_DIR: None, OUT_FORMAT_LIST: None, ADD_RATES: [DEF_ADD_RATE], INI_MONOS: DEF_INI_MONOS,
                MAX_MONOS: DEF_MAX_MONOS, SIM_TIME: DEF_SIM_TIME, SG_RATIOS: [DEF_SG], TEMP: DEF_TEMP,
                RANDOM_SEED: None, BASENAME: DEF_BASENAME, IMAGE_SIZE: DEF_IMAGE_SIZE, DYNAMICS: False, C_LIGNIN: False,
                E_BARRIER_KCAL_MOL: DEF_E_BARRIER_KCAL_MOL, E_BARRIER_J_PART: None, SAVE_FILES: False,
                SAVE_CSV: False, SAVE_JSON: False, SAVE_PDB: False, SAVE_PNG: False, SAVE_SMI: False, SAVE_SVG: False,
                SAVE_TCL: False, CHAIN_ID: DEF_CHAIN_ID, PSF_FNAME: DEF_PSF_FNAME, TOPPAR_DIR: DEF_TOPPAR,
                NUM_REPEATS: DEF_NUM_REPEATS, PLOT_BONDS: False, SUPPRESS_SMI: False, BREAK_CO: False,
                }

REQ_KEYS = {}


def plot_bond_error_bars(x_axis, y_axis_val_dicts, y_axis_std_dev_dicts, y_val_key_list, x_axis_label, y_axis_label,
                         plot_title, plot_fname):
    plt.figure(figsize=(3, 5))
    for y_idx, y_key in enumerate(y_val_key_list):
        plt.errorbar(x_axis, y_axis_val_dicts[y_key], yerr=y_axis_std_dev_dicts[y_key], linestyle='none', marker='.',
                     markersize=10, markerfacecolor=PLOT_COLORS[y_idx], markeredgecolor=PLOT_COLORS[y_idx],
                     label=y_key, capsize=3, ecolor=PLOT_COLORS[y_idx])

    if len(x_axis) > 1:
        plt.xscale('log')

    [plt.gca().spines[i].set_linewidth(1.5) for i in ['top', 'right', 'bottom', 'left']]
    plt.gca().tick_params(axis='both', which='major', labelsize=14, direction='in', pad=8, top=True, right=True,
                          width=1.5, length=6)
    plt.gca().tick_params(axis='both', which='minor', labelsize=14, direction='in', pad=8, top=True, right=True,
                          width=1, length=4)
    plt.ylabel(y_axis_label, fontsize=14)
    plt.xlabel(x_axis_label, fontsize=14)

    if '%' in y_axis_label:
        plt.ylim([0.0, 1.0])
    # needed to adjust legends differently for two types of plot
    if y_val_key_list[0] in BOND_TYPE_LIST:
        plt.legend(fontsize=14, loc='upper right', bbox_to_anchor=(1.45, 1.05), frameon=False)
    else:
        plt.legend(fontsize=14, loc='upper right', bbox_to_anchor=(1.7, 1.05), frameon=False)

    plt.title(plot_title)
    plt.savefig(plot_fname, bbox_inches='tight', transparent=True)
    print(f"Wrote file: {plot_fname}")
    plt.close()


def get_avg_percent_bonds(bond_list, num_opts, summary_lists, num_trials, break_co_bonds=False):
    """
    Given adj_list for a set of options, with repeats for each option, find the avg and std dev of percent of each
    bond type
    :param bond_list: list of strings representing each bond type
    :param num_opts: number of options specified (should be length of adj_lists)
    :param summary_lists: list of lists of summaries: outer is for each option, inner is for each repeat
    :param num_trials: number of repeats (should be length of inner adj_lists list)
    :param break_co_bonds: Boolean, to determine whether determine oligomers and remaining bonds after removing C-O
        bonds to simulate RCF
    :return: avg_bonds, std_bonds: list of floats, list of floats: for each option tested, the average and std dev
                  of bond distributions (percentages)
    """
    bond_percents = {}
    avg_bonds = {}
    std_bonds = {}

    if break_co_bonds:
        rcf_monomers = [[summary_lists[j][i][RCF_MONOS][1]/summary_lists[j][i][N_FINAL]
                         for i in range(num_trials)] for j in range(num_opts)]
        avg_monomers = [np.mean(rcf_monomers)]
        std_monomers = [np.sqrt(np.var(rcf_monomers))]
    else:
        avg_monomers, std_monomers = None, None

    for bond_type in bond_list:
        bond_percents[bond_type] = [[summary_lists[j][i][BONDS][bond_type]/sum(summary_lists[j][i][BONDS].values())
                                     for i in range(num_trials)] for j in range(num_opts)]
        avg_bonds[bond_type] = [np.mean(bond_pcts) for bond_pcts in bond_percents[bond_type]]
        std_bonds[bond_type] = [np.sqrt(np.var(bond_pcts)) for bond_pcts in bond_percents[bond_type]]
    return avg_bonds, std_bonds, avg_monomers, std_monomers


def create_bond_v_sg_plots(add_rate_str, cfg, sg_adjs):
    all_avg_bonds, all_std_bonds, avg_monos, std_monos = get_avg_percent_bonds(BOND_TYPE_LIST, len(cfg[SG_RATIOS]),
                                                                               sg_adjs, cfg[NUM_REPEATS], cfg[BREAK_CO])
    title = f"Add rate {add_rate_str} monomer/s"
    x_axis_label = 'SG Ratio'
    y_axis_label = 'Bond Type Yield (%)'
    fname = create_out_fname(f'bond_dist_v_sg_{add_rate_str}', base_dir=cfg[OUT_DIR], ext='.png')
    plot_bond_error_bars(cfg[SG_RATIOS], all_avg_bonds, all_std_bonds, BOND_TYPE_LIST,
                         x_axis_label, y_axis_label, title, fname)


def create_dynamics_plots(add_rate_str, bond_types, cfg, num_monos, num_oligs, sg_ratio):
    # Starting with num mon & olig vs timestep:
    len_y_val_key_list = [MONOMERS, OLIGOMERS]
    min_len = len(num_monos[0])
    avg_bond_types = {}
    std_bond_types = {}
    if cfg[NUM_REPEATS] > 1:
        # If there are multiple runs, arrays may be different lengths, so find shortest array
        min_len = len(num_monos[0])
        for mono_list in num_monos[1:]:
            if len(mono_list) < min_len:
                min_len = len(mono_list)
        # make lists of lists into np array
        sg_num_monos = np.asarray([np.array(num_list[:min_len]) for num_list in num_monos])
        # could save; for now, use to make images
        av_num_monos = np.mean(sg_num_monos, axis=0)

        sg_num_oligs = np.asarray([np.array(num_list[:min_len]) for num_list in num_oligs])
        av_num_oligs = np.mean(sg_num_oligs, axis=0)

        std_num_monos = np.std(sg_num_monos, axis=0)
        std_num_oligs = np.std(sg_num_oligs, axis=0)

        len_y_axis_val_dicts = {MONOMERS: av_num_monos, OLIGOMERS: av_num_oligs}
        len_y_axis_std_dev_dicts = {MONOMERS: std_num_monos, OLIGOMERS: std_num_oligs}

        for bond_type in BOND_TYPE_LIST:
            sg_bond_dist = np.asarray([np.array(bond_list[:min_len]) for
                                       bond_list in bond_types[bond_type]])
            avg_bond_types[bond_type] = np.mean(sg_bond_dist, axis=0)
            std_bond_types[bond_type] = np.std(sg_bond_dist, axis=0)

    else:
        len_y_axis_val_dicts = {MONOMERS: num_monos[0], OLIGOMERS: num_oligs[0]}
        len_y_axis_std_dev_dicts = {MONOMERS: None, OLIGOMERS: None}

        for bond_type in BOND_TYPE_LIST:
            avg_bond_types[bond_type] = bond_types[bond_type]
            std_bond_types[bond_type] = None
    timesteps = list(range(min_len))
    title = f"S:G Ratio {sg_ratio}, Add rate {add_rate_str} monomer/s"
    sg_str = f'{sg_ratio:.{3}g}'.replace("+", "").replace(".", "-")
    fname = create_out_fname(f'mono_olig_v_step_{sg_str}_{add_rate_str}', base_dir=cfg[OUT_DIR],
                             ext='.png')
    x_axis_label = 'Time step'
    y_axis_label = 'Number'
    plot_bond_error_bars(timesteps, len_y_axis_val_dicts, len_y_axis_std_dev_dicts, len_y_val_key_list,
                         x_axis_label, y_axis_label, title, fname)
    fname = create_out_fname(f'bond_dist_v_step_{sg_str}_{add_rate_str}', base_dir=cfg[OUT_DIR],
                             ext='.png')
    x_axis_label = 'Time step'
    y_axis_label = 'Number of Bonds'
    plot_bond_error_bars(timesteps, avg_bond_types, std_bond_types, BOND_TYPE_LIST,
                         x_axis_label, y_axis_label, title, fname)


def adj_analysis_to_stdout(ini_monos, max_monos, max_time, add_rate, sg_ratio,
                           final_time, summary, break_co_bonds=False, c_lignin=False):
    """
    Print key output to stdout, and the specified conditions
    :param ini_monos: int, initial number of monomers
    :param max_monos: int, max number of monomers
    :param max_time: float, max allowable simulation time in seconds
    :param add_rate: float, monomer addition rate in monomers/s
    :param sg_ratio: float, S:G ratio
    :param final_time: float, final time from simulation in seconds
    :param summary: a dictionary from analyze_adj_matrix
    :param break_co_bonds: Boolean, to determine whether determine oligomers and remaining bonds after removing C-O
        bonds to simulate RCF
    :param c_lignin: Boolean, to indicate if C-lignin was modeled
    :return: n/a: prints to stdout
    """
    chain_len_results = summary[CHAIN_LEN]
    num_monos_created = sum(summary[CHAIN_MONOS].values())
    option_summary = f"With options: initial monomers: {ini_monos}, max monomers: {max_monos}, " \
                     f"max time: {max_time:.2e} s, add rate: {add_rate} monomers/s, and "
    if c_lignin:
        option_summary += "100% C-lignin,"
    else:
        option_summary += f"S to G ratio: {sg_ratio},"

    print(option_summary)
    print(f"simulation created {num_monos_created} monomers in {final_time:.2e} s, which formed:")
    print_olig_distribution(chain_len_results, summary[CHAIN_BRANCH_COEFF])

    lignin_bonds = summary[BONDS]
    print(f"composed of the following bond types and number:")
    print_bond_type_num(lignin_bonds)

    if break_co_bonds:
        print("Breaking C-O intermonomer bonds to simulate RCF results in:")
        print_olig_distribution(summary[RCF_YIELDS], summary[RCF_BRANCH_COEFF])

        print(f"with the following remaining bond types and number:")
        print_bond_type_num(summary[RCF_BONDS])


def print_bond_type_num(lignin_bonds):
    bond_summary = ""
    for bond_type, bond_num in lignin_bonds.items():
        bond_summary += f"   {bond_type.upper():>4}: {bond_num:4}"
    bond_summary += "\n"
    print(bond_summary)


def print_olig_distribution(chain_len_results, coeff):
    for olig_len, olig_num in chain_len_results.items():
        if olig_len == 1:
            print(f"{olig_num:>8} monomer(s) (chain length 1)")
        elif olig_len == 2:
            print(f"{olig_num:>8} dimer(s) (chain length 2)")
        elif olig_len == 3:
            print(f"{olig_num:>8} trimer(s) (chain length 3)")
        else:
            print(f"{olig_num:>8} oligomer(s) of chain length {olig_len}, with branching coefficient "
                  f"{round(coeff[olig_len], 3)}")


def degree(adj):
    """
    Determines the degree for each monomer within the polymer chain. The "degree" concept in graph theory
    is the number of edges connected to a node. In the context of lignin, that is simply the number of
    connected residues to a specific residue, and can be used to determine derived properties like the
    branching coefficient.
    :param adj: scipy dok_matrix   -- the adjacency matrix for the lignin polymer that has been simulated
    :return: The degree for each monomer as a numpy array.
    """
    return np.bincount(adj.nonzero()[0])


def overall_branching_coefficient(adj):
    """
    Based on the definition in Dellon et al. (10.1021/acs.energyfuels.7b01150), this is the number of
       branched oligomers divided by the total number of monomers.
    This value is indifferent to the number of fragments in the output.

    :param adj: dok_matrix, the adjacency matrix for the lignin polymer that has been simulated
    :return: The branching coefficient that corresponds to the adjacency matrix
    """
    degrees = degree(adj)
    if len(degrees) == 0:
        return 0
    else:
        return np.sum(degrees >= 3) / len(degrees)


def get_bond_type_v_time_dict(adj_list, sum_len_larger_than=None):
    """
    given a list of adjs (one per timestep), flip nesting so have dictionaries of lists of type val vs. time
    for graphing, also created a 10+ list
    :param adj_list: list of adj dok_matrices
    :param sum_len_larger_than: None or an integer; if an integer, make a val_list that sums all lens >= that value
    :return: two dict of dicts
    """
    bond_type_dict = defaultdict(list)
    # a little more work for olig_len_monos_dict, since each timestep does not contain all possible keys
    olig_len_monos_dict = defaultdict(list)
    olig_len_count_dict = defaultdict(list)
    olig_count_dict_list = []
    frag_count_dict_list = []  # first make list of dicts to get max bond_length
    for adj in adj_list:  # loop over each timestep
        # this is keys = timestep  values
        count_bonds_list = count_bonds(adj)
        for bond_type in count_bonds_list:
            bond_type_dict[bond_type].append(count_bonds_list[bond_type])
        olig_yield_dict, olig_monos_dict, olig_branch_dict, olig_branch_coeff_dict = count_oligomer_yields(adj)
        olig_count_dict_list.append(olig_yield_dict)
        frag_count_dict_list.append(olig_monos_dict)
    # since breaking bonds is not allowed, the longest oligomer will be from the last step; ordered, so last len
    max_olig_len = list(frag_count_dict_list[-1].keys())[-1]
    # can now get the dict of lists from list of dicts
    for frag_count_list, olig_count_list in zip(frag_count_dict_list, olig_count_dict_list):
        for olig_len in range(1, max_olig_len + 1):
            olig_len_monos_dict[olig_len].append(frag_count_list.get(olig_len, 0))
            olig_len_count_dict[olig_len].append(olig_count_list.get(olig_len, 0))
    # now make a list of all values greater than a number, if given
    # first initialize as None so something can be returned, even if we are not summing over a particular number
    len_monos_plus_list = None
    len_count_plus_list = None
    if sum_len_larger_than:
        num_time_steps = len(adj_list)
        len_monos_plus_list = np.zeros(num_time_steps)
        len_count_plus_list = np.zeros(num_time_steps)
        # both dicts have same keys, so no worries
        for olig_len, val_list in olig_len_monos_dict.items():
            if olig_len >= sum_len_larger_than:
                len_monos_plus_list = np.add(len_monos_plus_list, val_list)
                len_count_plus_list = np.add(len_count_plus_list, olig_len_count_dict[olig_len])
    return bond_type_dict, olig_len_monos_dict, len_monos_plus_list, olig_len_count_dict, len_count_plus_list


def read_cfg(f_loc, cfg_proc=process_cfg):
    """
    Reads the given configuration file, returning a dict with the converted values supplemented by default values.

    :param f_loc: The location of the file to read.
    :param cfg_proc: The processor to use for the raw configuration values.  Uses default values when the raw
        value is missing.
    :return: A dict of the processed configuration file's data.
    """
    config = ConfigParser()
    good_files = config.read(f_loc)

    if not good_files:
        raise IOError(f"Could not find specified configuration file: {f_loc}")

    main_proc = cfg_proc(dict(config.items(MAIN_SEC)), DEF_CFG_VALS, REQ_KEYS)

    return main_proc


def parse_cmdline(argv=None):
    """
    Returns the parsed argument list and return code.
    :param argv: A list of arguments, or `None` for ``sys.argv[1:]``.
    """

    # initialize the parser object:
    parser = argparse.ArgumentParser(description=f"Create lignin chain(s) composed of 'S' ({S}) and/or 'G' ({G}) "
                                                 f"monolignols or 100% 'C' ({C})\nmonolignols, as described in  "
                                                 f"Orella, M., Gani, T. Z. H., Vermaas, J. V., Stone, M. L., Anderson, "
                                                 f"E. M.,\nBeckham, G. T., Brushett, Fikile R., Roman-Leshkov, Y. "
                                                 f"(2019). Lignin-KMC: A Toolkit for Simulating Lignin\nBiosynthesis. "
                                                 f"ACS Sustainable Chemistry & Engineering. "
                                                 f"https://doi.org/10.1021/acssuschemeng.9b03534.\n\nBy default, the "
                                                 f"Gibbs free energy barriers from this reference will be used, as "
                                                 f"specified in Tables S1 and S2.\nAlternately, the user may specify "
                                                 f"values as a dict of dict of dicts in a specified configuration file "
                                                 f"(specified\nwith '-c') using the '{E_BARRIER_KCAL_MOL}' or "
                                                 f"'{E_BARRIER_J_PART}'  parameters with corresponding units\n"
                                                 f"(kcal/mol or joules/particle, respectively), in a configuration file"
                                                 f" (see '-c' option). The format is (bond_type:\nmonomer(s) involved: "
                                                 f"units involved: ea_vals), for example:\n      "
                                                 f"ea_dict = {{{OX}: {{'G': {{{MONOMER}: 0.9, {OLIGOMER}: 6.3}}, "
                                                 f"'S': ""{{{MONOMER}: 0.6, {OLIGOMER}: " f"2.2}}}}, ...}}.\n\n"
                                                 f"All command-line options may alternatively be specified in a "
                                                 f"configuration file. Command-line (non-default)\nselections will "
                                                 f"override configuration file specifications.\n\n"
                                                 f"The default includes the number of each intermonomer bond type and "
                                                 f"a SMILES string printed to standard out.",
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("-a", "--add_rates", help=f"A comma-separated list of the rates of monomer addition to the "
                                                  f"system (in monomers/second), \nto be used when the '{MAX_MONOS}' "
                                                  f"('-m' option) is larger than '{INI_MONOS}' \n('-i' option), thus "
                                                  f"specifying monomer addition. The simulation will end when either "
                                                  f"there \nare no more possible reactions (including monomer "
                                                  f"addition) or when the '{SIM_TIME}' \n('-l' option) is reached, "
                                                  f"whichever comes first. Note: if there are spaces in the list of "
                                                  f"\naddition rates, the list must be enclosed in quotes to be read "
                                                  f"as a single string. The \ndefault list contains the single "
                                                  f"addition rate of {DEF_ADD_RATE} monomers/s.",
                        default=[DEF_ADD_RATE])
    parser.add_argument("-b", "--break_co_bonds", help=f"Flag to output results from C-O bonds to simulate RCF "
                                                       f"results. The default is False.", action="store_true")
    parser.add_argument("-c", "--config", help="The location of the configuration file in the 'ini' format. This file "
                                               "can be used to \noverwrite default values such as for energies.",
                        default=None, type=read_cfg)
    parser.add_argument("-d", "--out_dir", help="The directory where output files will be saved. The default is "
                                                "the current directory.", default=DEF_CFG_VALS[OUT_DIR])
    parser.add_argument("-dy", "--dynamics_flag", help=f"Select this option if dynamics (results per timestep) are "
                                                       f"requested. If chosen, plots of \nmonomers and oligomers vs "
                                                       f"timestep, and bond type percent vs timestep, will be saved. "
                                                       f"\nThey will be named 'bond_dist_v_step_*_#.png' and "
                                                       f"'mono_olig_v_step_*_#.png', where * \nrepresents the S:G "
                                                       f"ratio and # represents the addition rate. Note that this "
                                                       f"option \nsignificantly increases simulation time.",
                        action="store_true")
    parser.add_argument("-f", "--output_format_list", help="The type(s) of output format to be saved. Provide as a "
                                                           "space- or comma-separated list. \nNote: if the list has "
                                                           "spaces, it must be enclosed in quotes, to be treated as "
                                                           "a single \nstring. The currently supported types are: "
                                                           f"'{OUT_TYPE_STR}'.\nThe '{SAVE_CSV}' option produce "
                                                           f"summary csv files. The '{SAVE_JSON}' option will save a "
                                                           f"json format of\nRDKit's 'mol' (molecule) object.\nThe "
                                                           f"'{SAVE_TCL}' option will create a file for use with VMD to"
                                                           f" generate a psf file and 3D\nmolecules, as described in "
                                                           f"LigninBuilder, https://github.com/jvermaas/LigninBuilder,"
                                                           f"\n"
                                                           f"https://pubs.acs.org/doi/abs/10.1021/acssuschemeng.8b05665"
                                                           f".\nA base name for the saved files can be provided with "
                                                           f"the '-o' option. Otherwise, the \nbase name will be "
                                                           f"'{DEF_BASENAME}'.",
                        default=DEF_CFG_VALS[OUT_FORMAT_LIST])
    parser.add_argument("-i", "--initial_num_monomers", help=f"The initial number of monomers to be included in the "
                                                             f"simulation. The default is {DEF_INI_MONOS}.",
                        default=DEF_CFG_VALS[INI_MONOS])
    parser.add_argument("-l", "--length_simulation", help=f"The length of simulation (simulation time) in seconds. The "
                                                          f"default is {DEF_SIM_TIME} s.", default=DEF_SIM_TIME)
    parser.add_argument("-m", "--max_num_monomers", help=f"The maximum number of monomers to be studied. The default "
                                                         f"value is {DEF_MAX_MONOS}.", default=DEF_MAX_MONOS)
    parser.add_argument("-n", "--num_repeats", help=f"The number of times each combination of sg_ratio and add_rate "
                                                    f"will be tested. The default is {DEF_NUM_REPEATS}.",
                        default=DEF_NUM_REPEATS)
    parser.add_argument("-o", "--output_basename", help=f"The base name for output file(s). If an extension is "
                                                        f"provided, it will determine \nthe type of output. Currently "
                                                        f"supported output types are: \n'{OUT_TYPE_STR}'. Multiple "
                                                        f"output formats can be selected with the \n'-f' option. If "
                                                        f"the '-f' option is selected and no output base name "
                                                        f"provided, a default \nbase name of '{DEF_BASENAME}' will be "
                                                        f"used.", default=DEF_BASENAME)
    parser.add_argument("-p", "--plot_bonds", help=f"Flag to produce plots of the percent of each bond type versus S:G "
                                                   f"ratio(s). One plot will \nbe created per addition rate, named "
                                                   f"'bond_dist_v_sg_#.png', where # represents \nthe addition rate.",
                        action="store_true")
    parser.add_argument("-r", "--random_seed", help="A positive integer to be used as a seed value for testing. The "
                                                    "default is not to use a \nseed, to allow pseudorandom lignin "
                                                    "creation.", default=DEF_CFG_VALS[RANDOM_SEED])
    parser.add_argument("-s", "--image_size", help=f"The output size of svg or png files in pixels. The default size "
                                                   f"is {DEF_IMAGE_SIZE} pixels. \nTo use a different size, provide "
                                                   f"two integers, separated by a space or a comma. \nNote: if the "
                                                   f"list of two numbers has any spaces in it, it must be enclosed "
                                                   f"in quotes.",
                        default=DEF_IMAGE_SIZE)
    parser.add_argument("-cl", "--c_lignin", help="A flag to specify modeling C-lignin (with 100%% caffeoyl monomers).",
                        action="store_true")
    parser.add_argument("-sg", "--sg_ratios", help=f"A comma-separated list of the S:G (guaiacol:syringyl) ratios to "
                                                   f"be tested. \nIf there are spaces, the list must be enclosed in "
                                                   f"quotes to be read as a single string. \nThe default list "
                                                   f"contains the single value {DEF_SG}.", default=[DEF_SG])
    parser.add_argument("-t", "--temperature_in_k", help=f"The temperature (in K) at which to model lignin "
                                                         f"biosynthesis. The default is {DEF_TEMP} K.\nNote: this "
                                                         f"temperature must match the temperature at which the "
                                                         f"energy barriers were calculated. ",
                        default=DEF_TEMP)
    parser.add_argument("-x", "--no_smi", help=f"Flag to suppress determining the SMILES string for the output, which "
                                               f"is created by default.", action="store_true")
    parser.add_argument("--chain_id", help=f"Option for use when generating a tcl script: the chainID to be used in "
                                           f"generating a psf \nand/or pdb file from a tcl script (see LigninBuilder). "
                                           f"This should be one character. If a \nlonger ID is provided, it will be "
                                           f"truncated to the first character. The default value is {DEF_CHAIN_ID}.",
                        default=DEF_CHAIN_ID)
    parser.add_argument("--psf_fname", help=f"Option for use when generating a tcl script: the file name for psf and "
                                            f"pdb files that will \nbe produced from running a tcl produced by this "
                                            f"package (see LigninBuilder). The default \nvalue is {DEF_PSF_FNAME}.",
                        default=DEF_PSF_FNAME)
    parser.add_argument("--toppar_dir", help=f"Option for use when generating a tcl script: the directory name where "
                                             f"VMD should look for \nthe toppar file(s) when running the tcl file in "
                                             f"VMD (see LigninBuilder). The default value \nis '{DEF_TOPPAR}'.",
                        default=DEF_TOPPAR)

    args = None
    try:
        args = parser.parse_args(argv)
        # dict below to map config input and defaults to command-line input
        conf_arg_dict = {OUT_DIR: args.out_dir,
                         OUT_FORMAT_LIST: args.output_format_list,
                         ADD_RATES: args.add_rates,
                         DYNAMICS: args.dynamics_flag,
                         INI_MONOS: args.initial_num_monomers,
                         SIM_TIME: args.length_simulation,
                         MAX_MONOS: args.max_num_monomers,
                         BASENAME: args.output_basename,
                         IMAGE_SIZE: args.image_size,
                         SG_RATIOS: args.sg_ratios,
                         C_LIGNIN: args.c_lignin,
                         TEMP: args.temperature_in_k,
                         RANDOM_SEED: args.random_seed,
                         CHAIN_ID: args.chain_id,
                         PSF_FNAME: args.psf_fname,
                         TOPPAR_DIR: args.toppar_dir,
                         NUM_REPEATS: args.num_repeats,
                         PLOT_BONDS: args.plot_bonds,
                         SUPPRESS_SMI: args.no_smi,
                         BREAK_CO: args.break_co_bonds,
                         }
        if args.config is None:
            args.config = DEF_CFG_VALS.copy()
        # Now overwrite any config values with command-line arguments, only if those values are not the default
        for config_key, arg_val in conf_arg_dict.items():
            if not (arg_val == DEF_CFG_VALS[config_key]):
                args.config[config_key] = arg_val

    except (KeyError, IOError, SystemExit) as e:
        if hasattr(e, 'code') and e.code == 0:
            return args, GOOD_RET

        # only print the e if it has meaningful info; 2 simply is system exit from parser;
        #    tests that have triggered System Exit are caught and explained below
        if not e.args[0] == 2:
            warning(e)

        # Easy possible error is to have a space in a list; check for it
        check_arg_list = []

        for arg_str in ['-f', '--output_format_list', '-s', '--image_size', "-a", "--add_rates", "-sg", "--sg_ratios"]:
            if arg_str in argv:
                check_arg_list.append(arg_str)
        if len(check_arg_list) > 0:
            check_list = "', '".join(check_arg_list)
            warning(f"Check your entry/entries for '{check_list}'. If spaces separate list entries, "
                    f"enclose the whole list in quotes, or separate with commas only.")

        parser.print_help()
        return args, INPUT_ERROR
    return args, GOOD_RET


def calc_rates(temp, ea_j_part_dict=None, ea_kcal_mol_dict=None):
    """
    Uses temperature and provided Gibbs free energy barriers (at 298.15 K and 1 atm) to calculate rates using the
        Eyring equation
    Dict formats: dict = {rxn_type: {substrate(s): {sub_lengths (e.g. (monomer, monomer)): value, ...}, ...}, ...}

    Only ea_j_part_dict or ea_kcal_mol_dict are needed; if both are provided, only ea_j_part_dict will be used

    :param temp: float, temperature in K
    :param ea_j_part_dict: dictionary of Gibbs free energy barriers in Joule/particle, in format noted above
    :param ea_kcal_mol_dict: dictionary of Gibbs free energy barriers in kcal/mol, in format noted above
    :return: rxn_rates: dict of reaction rates units of 1/s
    """
    # want Gibbs free energy barriers in J/particle for later calculation;
    #     user can provide them in those units or in kcal/mol
    if ea_j_part_dict is None:
        ea_j_part_dict = {
            rxn_type: {substrate: {sub_len: ea_kcal_mol_dict[rxn_type][substrate][sub_len] * KCAL_MOL_TO_J_PART
                                   for sub_len in ea_kcal_mol_dict[rxn_type][substrate]}
                       for substrate in ea_kcal_mol_dict[rxn_type]} for rxn_type in ea_kcal_mol_dict}
    rxn_rates = {}
    for rxn_type in ea_j_part_dict:
        rxn_rates[rxn_type] = {}
        for substrate in ea_j_part_dict[rxn_type]:
            rxn_rates[rxn_type][substrate] = {}
            for substrate_type in ea_j_part_dict[rxn_type][substrate]:
                # rounding to reduce difference due to solely to platform running package
                rate = KB * temp / H * np.exp(-ea_j_part_dict[rxn_type][substrate][substrate_type] / KB / temp)
                rxn_rates[rxn_type][substrate][substrate_type] = round_sig_figs(rate, sig_figs=15)
    return rxn_rates


def create_initial_monomers(pct_s, monomer_draw):
    """
    Make a monomer list (length of monomer_draw) based on the types determined by the monomer_draw list and pct_s
    :param pct_s: float ([0:1]), fraction of  monomers that should be type "S"
    :param monomer_draw: a list of floats ([0:1)) to determine if the monomer should be type "G" (val < pct_s) or
                         "S", otherwise
    :return: list of Monomer objects of specified type
    """
    # if mon_choice < pct_s, make it an S; that is, the evaluation comes back True (=1='S');
    #     otherwise, get False = 0 = 'G'. Since only two options (True/False) only works for 2 monomers
    return [Monomer(INT_TO_TYPE_DICT[int(mono_type_draw < pct_s)], i) for i, mono_type_draw in enumerate(monomer_draw)]


def create_initial_events(initial_monomers, rxn_rates):
    """
    # Create event_dict that will oxidize every monomer
    :param initial_monomers: a list of Monomer objects
    :param rxn_rates: dict of dict of dicts of reaction rates in 1/s
    :return: a list of oxidation Event objects to initialize the state by allowing oxidation of every monomer
    """
    return [Event(OX, [mon.identity], rxn_rates[OX][mon.type][MONOMER]) for mon in initial_monomers]


def produce_output(adj_matrix, mono_list, cfg):
    mol3d = None  # make IDE happy
    if cfg[SUPPRESS_SMI] and not (cfg[SAVE_JSON] or cfg[SAVE_PNG] or cfg[SAVE_SVG]):
        format_list = [SAVE_TCL]
        mol = None  # Make IDE happy
    else:
        # Default out is SMILES, which requires getting an rdKit molecule object; also required for everything
        #    except the TCL format
        format_list = [SAVE_TCL, SAVE_JSON, SAVE_PDB, SAVE_PNG, SAVE_SVG]
        block = generate_mol(adj_matrix, mono_list)
        mol = MolFromMolBlock(block)
        if mol is None:
            # MolFromMolBlock returns None when there is an error
            raise InvalidDataError("Error in producing a RDKit mol file.")
        try:
            smi_str = MolToSmiles(mol) + '\n'
        # the error we want to capture is Boost.Python.ArgumentError, which is non-trivial to import, and don't want
        #     users to have to non-trivially install anything; thus, ignoring the PEP8 E722 ("do not use bare except,
        #     specify exception instead" warning since didn't see a way to suppress the error just in this case
        except:
            raise InvalidDataError("Error in producing SMILES string.")
        # if SMI is to be saved, don't output to stdout
        if cfg[SAVE_SMI]:
            fname = create_out_fname(cfg[BASENAME], base_dir=cfg[OUT_DIR], ext=SAVE_SMI)
            str_to_file(smi_str, fname, print_info=True)
        else:
            print("\nSMILES representation: \n", MolToSmiles(mol), "\n")
        if cfg[SAVE_PNG] or cfg[SAVE_SVG] or cfg[SAVE_JSON]:
            # PNG and SVG make 2D images and thus need coordinates
            # JSON will save coordinates--zero's if not computed; might as well compute and save non-zero values
            Compute2DCoords(mol)
        if cfg[SAVE_PNG]:
            # the only format that needs 3D coordinates
            # adding H's does not change the molfile in place, but returns a new one
            # we want a new mol file because the changes made here would make for weird 2D images
            mol3d = AddHs(mol)
            Kekulize(mol3d)
            # # below is how to copy the mol, but not needed because above line makes a new mol
            # mol3d = Mol(mol)
            EmbedMolecule(mol3d, randomSeed=2)
            MMFFOptimizeMolecule(mol3d)

    for save_format in format_list:
        if cfg[save_format]:
            fname = create_out_fname(cfg[BASENAME], base_dir=cfg[OUT_DIR], ext=save_format)
            if save_format == SAVE_TCL:
                gen_tcl(adj_matrix, mono_list, tcl_fname=fname, chain_id=cfg[CHAIN_ID],
                        psf_fname=cfg[PSF_FNAME], toppar_dir=cfg[TOPPAR_DIR], out_dir=cfg[OUT_DIR])
            if save_format == SAVE_JSON:
                json_str = MolToJSON(mol)
                str_to_file(json_str + '\n', fname)
            if save_format == SAVE_PNG or save_format == SAVE_SVG:
                MolToFile(mol, fname, size=cfg[IMAGE_SIZE])
            if save_format == SAVE_PDB:
                # Not directly using MolToPDBFile because it does not have an option to add a remark
                pdb_block = MolToPDBBlock(mol3d)
                remark = f"REMARK    Lignin molecule created by syn_lignin in lignin-wrangler version {__version__}\n"
                str_to_file(remark + pdb_block, fname)

            print(f"Wrote file: {fname}")


def initiate_state(add_rate, cfg, rep, sg_ratio, c_lignin=False):
    ini_num_monos = cfg[INI_MONOS]
    if c_lignin:
        initial_monomers = [Monomer(C, i) for i in range(ini_num_monos)]
    else:
        pct_s = sg_ratio / (1 + sg_ratio)
        if cfg[RANDOM_SEED]:
            # we don't want the same random seed for every iteration
            np.random.seed(cfg[RANDOM_SEED] + int(add_rate / 100 + sg_ratio * 10) + rep)
            monomer_draw = np.around(np.random.rand(ini_num_monos), MAX_NUM_DECIMAL)
        else:
            monomer_draw = np.random.rand(ini_num_monos)
        initial_monomers = create_initial_monomers(pct_s, monomer_draw)
    # initial event must be oxidation to create reactive species; all monomers may be oxidized
    initial_events = create_initial_events(initial_monomers, cfg[RXN_RATES])
    if cfg[MAX_MONOS] > cfg[INI_MONOS]:
        initial_events.append(Event(GROW, [], rate=add_rate))
    elif cfg[MAX_MONOS] < cfg[INI_MONOS]:
        warning(f"The specified maximum number of monomers ({cfg[MAX_MONOS]}) is less than the specified initial "
                f"number of monomers ({cfg[INI_MONOS]}).\nThe program will proceed with the with the initial number "
                f"of monomers with no addition of monomers.")
    return initial_events, initial_monomers


def validate_input(cfg):
    """
    Checking for errors at the beginning, so don't waste time starting calculations that will not be able to complete

    :param cfg: dict of configuration values
    :return: will raise an error if invalid data is encountered
    """
    # Don't use "if cfg[RANDOM_SEED]:", because that won't catch the user giving the value 0, which they might think
    #    would be a valid random seed, but won't work for this package because of later "if cfg[RANDOM_SEED]:" checks
    if cfg[RANDOM_SEED] is not None:
        try:
            # numpy seeds must be 0 and 2**32 - 1. Raise an error if the input cannot be converted to an int. Also raise
            #   an error for 0, since that will return False that a seed was provided in the logic in this package
            cfg[RANDOM_SEED] = int(cfg[RANDOM_SEED])
            if cfg[RANDOM_SEED] <= 0 or cfg[RANDOM_SEED] > (2**32 - 1):
                raise ValueError
        except ValueError:
            raise InvalidDataError(f"Invalid input provided for '{RANDOM_SEED}': '{cfg[RANDOM_SEED]}'. If you "
                                   f"would like to obtain consistent output by using a random seed, provide a "
                                   f"positive integer value no greater than 2**32 - 1.")

    if cfg[C_LIGNIN] and cfg[SG_RATIOS] != [DEF_SG]:
        raise InvalidDataError("The flag '-c'/'--c_lignin' was specified in addition to providing '-sg'/'--sg_ratios'."
                               "Only 100% C-lignin can be modeled with this program.")

    # Convert list entries. Will already be lists if defaults are used. Otherwise, they will be strings.
    list_args = [ADD_RATES, SG_RATIOS]
    arg, arg_val = "", ""  # to make IDE happy
    try:
        for arg in list_args:
            arg_val = cfg[arg]
            # Will be a string to process unless it is the default
            if isinstance(arg_val, str):
                raw_vals = arg_val.replace(",", " ").replace("(", "").replace(")", "").split()
                cfg[arg] = [float(val) for val in raw_vals]
            else:
                cfg[arg] = arg_val
            for val in cfg[arg]:
                if val < 0:
                    raise ValueError
                # okay for sg_ratio to be zero, but not add_rate
                elif val == 0 and arg == ADD_RATES:
                    raise ValueError
    except ValueError:
        raise InvalidDataError(f"Found {arg_val} for '{arg}'. This entry must be able to be "
                               f"converted to a list of positive floats.")

    # now testing for positive floats
    for req_pos_num in [SIM_TIME]:
        try:
            cfg[req_pos_num] = float(cfg[req_pos_num])
            if cfg[req_pos_num] <= 0:
                raise ValueError
        except ValueError:
            raise InvalidDataError(f"Found '{cfg[req_pos_num]}' input for '{req_pos_num}'. The {req_pos_num} must be "
                                   f"a positive number.")

    # Required ints
    for req_pos_int_arg in [INI_MONOS, MAX_MONOS, NUM_REPEATS]:
        try:
            cfg[req_pos_int_arg] = int(cfg[req_pos_int_arg])
            if cfg[req_pos_int_arg] < 0:
                raise ValueError
        except ValueError:
            raise InvalidDataError(f"Found '{cfg[req_pos_int_arg]}' input for '{req_pos_int_arg}'. The "
                                   f"{req_pos_int_arg} must be a positive integer.")

    try:
        # Will be a string to process unless it is the default
        if isinstance(cfg[IMAGE_SIZE], str):
            raw_vals = cfg[IMAGE_SIZE].replace(",", " ").replace("(", "").replace(")", "").split()
            if len(raw_vals) != 2:
                raise ValueError
            cfg[IMAGE_SIZE] = (int(raw_vals[0]), int(raw_vals[1]))
    except ValueError:
        raise InvalidDataError(f"Found '{cfg[IMAGE_SIZE]}' input for '{IMAGE_SIZE}'. The {IMAGE_SIZE} must be "
                               f"two positive numbers, separated either by a comma or a space.")

    # Check for valid output requests
    check_if_files_to_be_saved(cfg)

    # determine rates to use
    if cfg[E_BARRIER_KCAL_MOL] == DEF_CFG_VALS[E_BARRIER_KCAL_MOL] and \
            (cfg[E_BARRIER_J_PART] == DEF_CFG_VALS[E_BARRIER_J_PART]) and (cfg[TEMP] == DEF_TEMP):
        cfg[RXN_RATES] = DEF_RXN_RATES
    else:
        if int(cfg[TEMP]) != int(DEF_TEMP):
            warning(f"The program will continue, using a temperature other than {DEF_TEMP}. Ensure that the energy "
                    f"barriers being used where calculated at the provided temperature ({cfg[TEMP]}), otherwise "
                    f"cancel this run.")
        cfg[RXN_RATES] = calc_rates(cfg[TEMP], ea_j_part_dict=cfg[E_BARRIER_J_PART],
                                    ea_kcal_mol_dict=cfg[E_BARRIER_KCAL_MOL])


def check_if_files_to_be_saved(cfg):
    """
    Evaluate input for requests to save output and check for valid specified locations
    :param cfg: dict of configuration values
    :return: if the cfg designs that files should be created, returns an updated cfg dict, and raises errors if
              invalid data in encountered
    """
    if cfg[OUT_FORMAT_LIST]:
        # remove any periods to aid comparison; might as well also change comma to space and then split on just space
        out_format_list = cfg[OUT_FORMAT_LIST].replace(".", " ").replace(",", " ")
        format_set = set(out_format_list.split())
    else:
        format_set = set()

    if cfg[BASENAME] and (cfg[BASENAME] != DEF_BASENAME):
        # If cfg[BASENAME] is not just the base name, make it so, saving a dir or ext in their spots
        out_path, base_name = os.path.split(cfg[BASENAME])
        if out_path and cfg[OUT_DIR]:
            cfg[OUT_DIR] = os.path.join(cfg[OUT_DIR], out_path)
        elif out_path:
            cfg[OUT_DIR] = out_path
        base, ext = os.path.splitext(base_name)
        cfg[BASENAME] = base
        format_set.add(ext.replace(".", ""))

    if len(format_set) > 0:
        for format_type in format_set:
            if format_type in OUT_TYPE_LIST:
                cfg[SAVE_FILES] = True
                cfg[format_type] = True
            else:
                raise InvalidDataError(f"Invalid extension provided: '{format_type}'. The currently supported types "
                                       f"are: '{OUT_TYPE_STR}'")
    if cfg[PLOT_BONDS]:
        cfg[SAVE_FILES] = True

    # if out_dir does not already exist, recreate it, only if we will actually need it
    if cfg[SAVE_FILES] and cfg[OUT_DIR]:
        make_dir(cfg[OUT_DIR])


def main(argv=None):
    """
    Runs the main program.

    :param argv: The command line arguments.
    :return: The return code for the program's termination.
    """
    if argv is None:
        argv = sys.argv[1:]

    print(f"Running Lignin-KMC version {__version__} with command-line options: {' '.join(argv)}\n"
          f"Please cite: https://pubs.acs.org/doi/abs/10.1021/acssuschemeng.9b03534\n")

    args, ret = parse_cmdline(argv)
    if ret != GOOD_RET or args is None:
        return ret

    cfg = args.config

    try:
        # tests at the beginning to catch errors early
        validate_input(cfg)

        # to store csv output
        ind_output_summary_list = []
        avg_output_summary_list = []

        for add_rate in cfg[ADD_RATES]:
            sg_summaries = []
            add_rate_str = f'{add_rate:.{3}g}'.replace("+", "").replace(".", "-")
            # There will be a list of length 1 even if it is C-lignin, because of the SG_RATIOS default value
            for sg_ratio in cfg[SG_RATIOS]:
                # the initialized lists below are for storing repeats
                bond_types = defaultdict(list)
                num_monos = []
                num_oligs = []
                rep_summaries = []

                for rep in range(cfg[NUM_REPEATS]):
                    # decide on initial monomers, based on given sg_ratio, and create initial oxidation events
                    initial_events, initial_monomers = initiate_state(add_rate, cfg, rep, sg_ratio, cfg[C_LIGNIN])

                    # begin simulation
                    if cfg[RANDOM_SEED]:
                        random_seed = cfg[RANDOM_SEED] + rep
                    else:
                        random_seed = cfg[RANDOM_SEED]
                    result = run_kmc(cfg[RXN_RATES], initial_monomers, initial_events, n_max=cfg[MAX_MONOS],
                                     t_max=cfg[SIM_TIME], sg_ratio=sg_ratio, dynamics=cfg[DYNAMICS],
                                     random_seed=random_seed)

                    if cfg[DYNAMICS]:
                        last_adj = result[ADJ_MATRIX][-1]
                        last_mono_list = result[MONO_LIST][-1]
                        (bond_type_dict, olig_monos_dict, sum_monos_list, olig_count_dict,
                         sum_count_list) = get_bond_type_v_time_dict(result[ADJ_MATRIX], sum_len_larger_than=2)

                        for bond_type in BOND_TYPE_LIST:
                            bond_types[bond_type].append(bond_type_dict[bond_type])
                        num_monos.append(olig_count_dict[1])
                        num_oligs.append(sum_count_list)

                    else:
                        last_adj = result[ADJ_MATRIX]
                        last_mono_list = result[MONO_LIST]

                    # show results
                    summary = analyze_adj_matrix(last_adj, break_co_bonds=cfg[BREAK_CO])
                    rep_summaries.append(summary)
                    adj_analysis_to_stdout(cfg[INI_MONOS], cfg[MAX_MONOS], cfg[SIM_TIME], add_rate, sg_ratio,
                                           result[TIME][-1], summary, break_co_bonds=cfg[BREAK_CO],
                                           c_lignin=cfg[C_LIGNIN])

                    # Outputs
                    produce_output(last_adj, last_mono_list, cfg)
                    # It is possible to have no monomers in RCF yields...
                    if cfg[SAVE_CSV]:
                        if summary[RCF_MONOS] is None:
                            summary[RCF_MONOS] = {1: ""}
                        elif 1 not in summary[RCF_MONOS]:
                            summary[RCF_MONOS][1] = 0
                        ind_output_summary = {ADD_RATE: add_rate, SG_RATIO: sg_ratio, N_INI: cfg[INI_MONOS],
                                              N_FINAL: cfg[MAX_MONOS], RCF_MONOS: summary[RCF_MONOS][1]}
                        ind_output_summary.update(summary[BONDS])
                        ind_output_summary_list.append(ind_output_summary)
                        fname = create_out_fname(cfg[BASENAME], suffix="_ind", base_dir=cfg[OUT_DIR], ext=SAVE_CSV)
                        write_csv(ind_output_summary_list, os.path.relpath(fname), IND_SUM_HEADERS,
                                  extrasaction="ignore", print_message=False)
                # save for potential plotting
                sg_summaries.append(rep_summaries)
                if cfg[SAVE_CSV]:
                    all_avg_bonds, all_std_bonds, avg_monos, std_monos = get_avg_percent_bonds(BOND_TYPE_LIST, 1,
                                                                                               [rep_summaries],
                                                                                               cfg[NUM_REPEATS],
                                                                                               cfg[BREAK_CO])
                    avg_output_summary = {ADD_RATE: add_rate, SG_RATIO: sg_ratio, N_INI: cfg[INI_MONOS],
                                          N_FINAL: cfg[MAX_MONOS]}
                    if cfg[BREAK_CO]:
                        avg_output_summary[RCF_MONOS_AVG] = avg_monos[0]
                        avg_output_summary[RCF_MONOS_STD] = std_monos[0]
                    for bond_type in BOND_TYPE_LIST:
                        avg_output_summary[bond_type + AVG] = all_avg_bonds[bond_type][0]
                        avg_output_summary[bond_type + STD] = all_std_bonds[bond_type][0]
                    avg_output_summary_list.append(avg_output_summary)
                    fname = create_out_fname(cfg[BASENAME], suffix="_avg", base_dir=cfg[OUT_DIR], ext=SAVE_CSV)
                    write_csv(avg_output_summary_list, os.path.relpath(fname), REPEATS_HEADERS,
                              extrasaction="ignore", print_message=False)

                # Now that all repeats done, create plots for dynamics, if applicable
                if cfg[DYNAMICS]:
                    # create plots of num mon & olig vs timestep, and % bond time v timestep
                    create_dynamics_plots(add_rate_str, bond_types, cfg, num_monos, num_oligs, sg_ratio)
            if cfg[PLOT_BONDS]:
                create_bond_v_sg_plots(add_rate_str, cfg, sg_summaries)

    except (InvalidDataError, KeyError) as e:
        warning(e)
        return INVALID_DATA

    return GOOD_RET  # success


if __name__ == '__main__':
    status = main()
    sys.exit(status)
