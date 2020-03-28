#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
create_library.py
From molecular formula
"""

import os
import sys
import argparse
from copy import deepcopy

# from rdkit.Chem.rdMolDescriptors import CalcMolFormula
# from rdkit.Chem.AllChem import (Compute2DCoords)
# from rdkit.Chem.rdmolops import AddHs
from rdkit.Chem.rdMolDescriptors import CalcMolFormula
from rdkit.Chem.Draw import MolsToGridImage, MolToFile
from rdkit.Chem.rdDepictor import Compute2DCoords

from rdkit.Chem.rdmolfiles import MolToMolFile
# from rdkit.Chem.AllChem import (Compute2DCoords, EmbedMolecule, MMFFOptimizeMolecule)
from common_wrangler.common import (GOOD_RET, IO_ERROR, INVALID_DATA, INPUT_ERROR, InvalidDataError, warning,
                                    parse_stoich, create_out_fname, natural_keys, make_dir)
from lignin_wrangler.lignin_common import (LIGNIN_ISOTOPE_DICT, MASS, MAX_SIG_FIGS, MS_SIG_FIGS,
                                           FORMULA_SMI_DICT, MW_FORM_DICT, MW_DEPROT_FORM_DICT, MW_PROT_FORM_DICT,
                                           SMI_NAME_DICT, SMI_SOURCE_DICT, CARBON, HYDROG, FLORINE, CHLORINE,
                                           BROMINE, IODINE, NITROGEN, FORMULA_DBE_DICT)
from rdkit import Chem

PNG_DIR = os.path.join(os.path.dirname(__file__), 'data', 'smi_pngs')
SEP_KEY = '|'


def calc_dbe(stoich_dict):
    """
    Given the numbers of each atom time, calculate the double-bond equivalents
       DBE = C + 1 - H/2 - X/2 + N/2
    While this program is unlikely to have X or N, making this a general-purpose method
    :param stoich_dict: dictionary of atom types and quantity
    :return: dbe
    """
    halogens = 0
    for atom_type in [FLORINE, CHLORINE, BROMINE, IODINE]:
        if atom_type in stoich_dict:
            halogens += stoich_dict[atom_type]
    for atom_type in [CARBON, HYDROG, NITROGEN]:
        if atom_type not in stoich_dict:
            stoich_dict[atom_type] = 0
    dbe = stoich_dict[CARBON] + 1 - stoich_dict[HYDROG]/2 - halogens/2 + stoich_dict[NITROGEN]/2
    return dbe


def save_mol_files(smi_list, out_dir):
    """
    Given a list of smiles strings, save each in a separate file
    :param smi_list: str, standard SMILES format
    :param out_dir: None or str, if None saves file to current directory, if str to location in str
    :return: n/a, saves a mol file for each smi
    """
    for smi_str in smi_list:
        fname = create_out_fname(smi_str, ext='mol', base_dir=out_dir)
        mol = Chem.MolFromSmiles(smi_str)

        # simplest (no H, no coordinates)
        # MolToMolFile(mol, fname, includeStereo=False, kekulize=True)

        # 2D coords without H
        Chem.Kekulize(mol)
        Compute2DCoords(mol)
        MolToMolFile(mol, fname, includeStereo=False)

        # # 2D coords with H
        # Chem.Kekulize(mol)
        # m2 = AddHs(mol)
        # Compute2DCoords(m2)
        # MolToMolFile(m2, fname, includeStereo=False)

        # # 3D coords
        # Chem.Kekulize(mol)
        # # adding H's does not change the molfile
        # m2 = AddHs(mol)
        # EmbedMolecule(m2)
        # MMFFOptimizeMolecule(m2)
        # MolToMolFile(m2, fname, includeStereo=False)
        # fname = create_out_fname(smi_str, ext='pdb', base_dir=out_dir)
        # MolToPDBFile(m2, fname)

        # print(f"Wrote file: {fname}")


def smi_to_formula(smi_str):
    """
    Given a smiles string in arbitrary format, return the smiles string as produced by RDKit,
        the molecular formula, and the molecular weight using only the most abundant isotopes
    :param smi_str: str, standard SMILES format
    :return: str, the molecular formula in standard chemistry notation
    """
    # Use RDKit to make a SMILES from a SMILES so that we get a unique string for any given SMILES entry
    mol = Chem.MolFromSmiles(smi_str)
    if mol is None:
        raise InvalidDataError(f"The input SMILES string '{smi_str}' could not be recognized by RDKit")
    Chem.Kekulize(mol)
    rd_smi = Chem.MolToSmiles(mol, kekuleSmiles=True)
    mol_formula = CalcMolFormula(mol)
    stoich_dict = parse_stoich(mol_formula)
    dbe = calc_dbe(stoich_dict)
    mol_mass = 0
    for atom_type, num_atoms in stoich_dict.items():
        mass_most_abundant_isotope = LIGNIN_ISOTOPE_DICT[atom_type][MASS][0]
        mol_mass += mass_most_abundant_isotope * num_atoms

    mw_deprot = round(mol_mass - LIGNIN_ISOTOPE_DICT[HYDROG][MASS][0], MAX_SIG_FIGS)
    mw_prot = round(mol_mass + LIGNIN_ISOTOPE_DICT[HYDROG][MASS][0], MAX_SIG_FIGS)

    return rd_smi, mol_formula, round(mol_mass, MAX_SIG_FIGS), mw_deprot, mw_prot, dbe


def add_smi_to_dicts(mw_formula_dict, mw_deprot_formula_dict, mw_prot_formula_dict, form_smi_dict, form_dbe_dict,
                     smi_name_dict, smi_source_dict, entered_smi, mol_name=None, mol_source=None):
    """
    Given a SMILES string, and optionally a name and/or source, update the designated formulas

    :param mw_formula_dict: dict of strs, eg. {'77.03913': 'C6H6', ...},
           (keys=MW as str with 5 decimal places, values=molecular formulas (str))
    :param mw_deprot_formula_dict: dict of strs, eg. {'78.04695': 'C6H6', ...}, (MW's have H mass subtracted)
           (keys=MW as str with 5 decimal places, values=molecular formulas (str))
    :param mw_prot_formula_dict: dict of strs, eg. {'78.04695': 'C6H6', ...}, (MW's have H mass added)
           (keys=MW as str with 5 decimal places, values=molecular formulas (str))
    :param form_smi_dict: dict of sets, e.g. {'C6H6': {'C1=CC=CC=C1'}, ...}
           (keys=molecular formulas (str), values=set of corresponding SMILES strings)
    :param form_dbe_dict: dict of floats, e.g. {'C6H6': 4, ...}
           (keys=molecular formulas (str), values=double bond equivalents (float))
    :param smi_name_dict: dict of sets, e.g. {'C1=COCCC1': {'dihydropyran', '3,4-dihydro-2h-pyran'}, ...}
           (keys=molecular formulas (str), values=set names including IUPAC and common names)
    :param smi_source_dict: dict of sets, e.g. {'C6H6': {'common_molecule'}, ...}
           (keys=molecular formulas (str), values=set of corresponding SMILES strings)
    :param entered_smi: str, the user-inputted SMILES string
    :param mol_name: str, optional, a molecule name for the string
    :param mol_source: str, optional, a note on the source of the molecule (e.g. if from model compound study)
    :return: boolean if updated dictionary, and updates dictionaries
    """
    # this will track addition to any dictionary--not going granular; can do so if later wished
    addition_to_dict = False
    new_smi, formula, mw, mw_deprot, mw_prot, dbe = smi_to_formula(entered_smi)
    if formula in form_smi_dict and new_smi != '':
        # since a set, if already there, adding would not change, but nice to track if anything changes
        if new_smi not in form_smi_dict[formula]:
            addition_to_dict = True
            form_smi_dict[formula].add(new_smi)
    else:
        form_smi_dict[formula] = {new_smi}
        form_dbe_dict[formula] = dbe
        addition_to_dict = True
    mw_dict_names = ["molecular ion MW dictionary", "deprotonated MW dictionary", "protonated MW dictionary"]
    mw_dicts = [mw_formula_dict, mw_deprot_formula_dict, mw_prot_formula_dict]
    mw_inputs = [str(mw), str(mw_deprot), str(mw_prot)]
    for dict_name, mw_dict, new_mw in zip(mw_dict_names, mw_dicts, mw_inputs):
        if new_mw in mw_dict.keys():
            if mw_dict[new_mw] != formula:
                # hopefully never encountered
                warning(f"Unexpectedly, the MW {new_mw} was already in the {dict_name} pared with molecular "
                        f"formula {mw_dict[new_mw]}, while the input has formula {formula}")
        else:
            mw_dict[new_mw] = formula
            addition_to_dict = True
    val_dict_list = [(mol_name, smi_name_dict), (mol_source, smi_source_dict)]
    for opt_val, opt_dict in val_dict_list:
        if opt_val:
            # user may add a list of names--unlikely multiple sources, but won't hurt it
            # cannot split on comma, because that can be a part of a name. PubChem splits on ; so seems safe
            opt_val_list = opt_val.split(";")
            for val in opt_val_list:
                stripped_val = val.strip().lower()
                if stripped_val:
                    if new_smi in opt_dict.keys():
                        # as above, check is to check if we are changing any dictionaries
                        if stripped_val not in opt_dict[new_smi]:
                            addition_to_dict = True
                            opt_dict[new_smi].add(stripped_val)
                    else:
                        opt_dict[new_smi] = {stripped_val}
                        addition_to_dict = True
    return addition_to_dict


def make_image_grid(file_label, smi_list, labels=None, out_dir=PNG_DIR, mol_img_size=(400, 300), write_output=True):
    """
    Given a molecular formula (or other label) and the set of SMI, make an image grid of all smiles within
    https://www.rdkit.org/docs/GettingStartedInPython.html
    :param file_label: str, such as chemical formula that corresponds to all smiles in SMILES set
    :param smi_list: list or set of SMILES strings; used to generate images
    :param labels: if None, will use the smi_list as labels; otherwise a list to use
    :param out_dir: directory where the file should be saved
    :param mol_img_size: tuple of ints to determine size of individual molecules
    :param write_output: boolean to determine whether to write to screen that a file was created
    :return: N/A, save a file
    """
    mols = []
    for smi in smi_list:
        mol = Chem.MolFromSmiles(smi)
        Compute2DCoords(mol)
        mols.append(mol)

    if labels:
        img_labels = labels
    else:
        img_labels = smi_list

    if len(mols) == 1:
        # didn't see a way for RDKit to add a label to an image with a single molecule (grid image does not work
        # for one image), so add to file name
        file_label += '_' + img_labels[0]
    fname = create_out_fname(file_label, ext='png', base_dir=out_dir)
    if len(mols) == 1:
        MolToFile(mols[0], fname, size=mol_img_size)
    else:
        img_grid = MolsToGridImage(mols, molsPerRow=3, subImgSize=mol_img_size, legends=img_labels)
        img_grid.save(fname)
    if write_output:
        print(f"Wrote file: {os.path.relpath(fname)}")


def image_grid_mult_mw(mw_list, mw_formula_dict, form_smi_dict, out_dir=PNG_DIR, formula_label=False):
    """
    Save a png of molecular structures labeled with their MW, formula, and SMILES string
    :param mw_list: list of MWs to include in image (added so could chose a subset from mw_formula_dict)
    :param mw_formula_dict: dict of MW to formulas: {MW1: formula1, MW2, formula2, ...}
    :param form_smi_dict: dict of formulas to SMILES strings: {formula1: [smi_str1, smi_str2, ...], ...}
    :param out_dir: str, dir where to save file
    :param formula_label: boolean, if output file name should start with formula, not MW
    :return: N/A, saves file(s)
    """
    # make a separate image for each MW
    for mw in mw_list:
        formula = mw_formula_dict[mw]
        if formula_label:
            # convert MW back to float for rounding, then replace so only one period in name
            file_label = formula
        else:
            # convert MW back to float for rounding, then replace so only one period in name
            file_label = f"{float(mw):.{MS_SIG_FIGS}f}_{formula}".replace('.', '-')
        # sort the dict value (which is a list of smiles) for consistency
        sorted_smi_list = sorted(form_smi_dict[formula])
        make_image_grid(file_label, sorted_smi_list, out_dir=out_dir)


def parse_cmdline(argv):
    """
    Returns the parsed argument list and return code.
    `argv` is a list of arguments, or `None` for ``sys.argv[1:]``.
    """
    if argv is None:
        argv = sys.argv[1:]

    # initialize the parser object:
    parser = argparse.ArgumentParser(description="This script has two modes, chosen by selected '-f' or '-i': "
                                                 "1) The '-f' option: reads a file to add entries to "
                                                 "dictionaries of lignin decomposition molecules that may be "
                                                 "observed in mass spectrometry of lignin-derived compounds. Given "
                                                 "SMILES strings, and optionally/ideally molecular names and/or source "
                                                 "of the SMILES (e.g. observed in analysis of model compounds), the "
                                                 "dictionaries are expanded to include additional potentially "
                                                 "observed molecular weights and isomers. Note: it does not change "
                                                 "the original libraries within this package, but instead outputs "
                                                 "new libraries, which could be used to update the library in this "
                                                 "package. 2) The '-i' option: creates an image library of all "
                                                 "SMILES structures currently in the compound library (further details "
                                                 "provided under the '-i' option description).")
    parser.add_argument("-d", "--out_dir", help="A directory where output files should be saved. The default location "
                                                "is the current working directory.", default=None)
    parser.add_argument("-f", "--file_name", help=f"File name of values separated by '{SEP_KEY}' (to avoid conflicts "
                                                  f"with IUPAC molecule names) with up to 3 values per line: SMILES "
                                                  f"string (required), molecule name(s) (optional; split multiple "
                                                  f"names with a semicolon), source (e.g. model compound analysis)",
                        default=None)
    parser.add_argument("-i", "--image_library", help=f"Flag to request that the program create a 2D image library of "
                                                      f"the SMILES strings in the library. One file will be created "
                                                      f"per exact molecular weight (calculated only from the most "
                                                      f"abundant isotope). If there are multiple SMILES matches for a "
                                                      f"molecular formula, the name of the file is '{{molecular "
                                                      f"weight (with a '-' instead of a '.')}}_{{molecular formula}}"
                                                      f".png', and the images of each structure within the file will "
                                                      f"be labeled with its SMILES string. If there is only one "
                                                      f"structure in the library for a molecular formula, the SMILES "
                                                      f"string will be appended to the name. These files will be "
                                                      f"saved in the current directory, unless a different directory "
                                                      f"is specified with the '-o' option.", action='store_true')
    parser.add_argument("-m", "--mw_list", help="A list of molecular weight keys for making an image library.",
                        default=None)

    args = None
    try:
        args = parser.parse_args(argv)
        if not args.image_library and not args.file_name:
            raise InvalidDataError("Please choose to either provide a file_name ('-f') to read new dictionary "
                                   "entries, or the image_library flag ('-i') to request 2D image library.")
    except (KeyError, InvalidDataError, IOError, SystemExit) as e:
        if hasattr(e, 'code') and e.code == 0:
            return args, GOOD_RET
        warning(e)
        parser.print_help()
        return args, INPUT_ERROR
    return args, GOOD_RET


def pretty_print_dicts(mw_formula_dict, mw_deprot_formula_dict, mw_prot_formula_dict, form_smi_dict, form_dbe_dict,
                       smi_name_dict, smi_source_dict):
    """
    Sort various output differently depending of it has strings as number, SMILES strings.... then print so they
        can be recognized as dictionaries by python
    :param mw_formula_dict: dict of strs, eg. {'77.03913': 'C6H6', ...},
           (keys=MW as str with 5 decimal places, values=molecular formulas (str))
    :param mw_deprot_formula_dict: dict of strs, eg. {'78.04695': 'C6H6', ...}, (MW's have H mass subtracted)
           (keys=MW as str with 5 decimal places, values=molecular formulas (str))
    :param mw_prot_formula_dict: dict of strs, eg. {'78.04695': 'C6H6', ...}, (MW's have H mass added)
           (keys=MW as str with 5 decimal places, values=molecular formulas (str))
    :param form_smi_dict: dict of sets, e.g. {'C6H6': {'C1=CC=CC=C1'}, ...}
           (keys=molecular formulas (str), values=set of corresponding SMILES strings)
    :param form_dbe_dict: dict of floats, e.g. {'C6H6': 4, ...}
           (keys=molecular formulas (str), values=double bond equivalents (float))
    :param smi_name_dict: dict of sets, e.g. {'C1=COCCC1': {'dihydropyran', '3,4-dihydro-2h-pyran'}, ...}
           (keys=molecular formulas (str), values=set names including IUPAC and common names)
    :param smi_source_dict: dict of sets, e.g. {'C6H6': {'common_molecule'}, ...}
           (keys=molecular formulas (str), values=set of corresponding SMILES strings)
    :return: nothing--just prints
    """
    # key lists (to be used for matching MW) as lists of floats
    mw_dict_key_list = {'MW_KEYS': mw_formula_dict, 'MW_DEPROT_KEYS': mw_deprot_formula_dict,
                        'MW_PROT_KEYS': mw_prot_formula_dict}
    for key_list, mw_dict in mw_dict_key_list.items():
        temp_list = []
        for key in mw_dict:
            temp_list.append(float(key))
        print(f"{key_list} = {sorted(temp_list)}")
    print("")

    # output to be "naturally sorted"
    orig_name_modified_dict1 = {'MW_FORM_DICT': mw_formula_dict, 'MW_DEPROT_FORM_DICT': mw_deprot_formula_dict,
                                'MW_PROT_FORM_DICT': mw_prot_formula_dict, 'FORMULA_SMI_DICT': form_smi_dict,
                                'FORMULA_DBE_DICT': form_dbe_dict}
    for orig_name, mod_dict in orig_name_modified_dict1.items():
        dict_keys = list(mod_dict.keys())
        dict_keys.sort(key=natural_keys)
        if orig_name in ['FORMULA_SMI_DICT', 'FORMULA_DBE_DICT']:
            dict_keys.sort(key=len)
            dict_str = ", ".join([f"'{key}': {mod_dict[key]}" for key in dict_keys])
        else:
            dict_str = ", ".join([f"'{key}': '{mod_dict[key]}'" for key in dict_keys])
        print(f"{orig_name} = {{{dict_str}}}")
    print("")

    # sort SMILES by length after standard sort
    orig_name_modified_dict2 = {'SMI_NAME_DICT': smi_name_dict, 'SMI_SOURCE_DICT': smi_source_dict}
    for orig_name, mod_dict in orig_name_modified_dict2.items():
        dict_keys = sorted(mod_dict.keys())
        dict_keys.sort(key=len)
        dict_str = ", ".join([f"'{key}': {mod_dict[key]}" for key in dict_keys])
        print(f"{orig_name} = {{{dict_str}}}")


def process_input_file(input_fname, mw_formula_dict, mw_deprot_formula_dict, mw_prot_formula_dict, form_smi_dict,
                       form_dbe_dict, smi_name_dict, smi_source_dict):
    """
    Read the file and uses the data to update dictionaries
    :return: the number of entries that were added to the dictionaries
    """
    rel_path_name = os.path.relpath(input_fname)
    new_entries = 0
    with open(input_fname) as f:
        for line in f:
            stripped_line = line.strip()
            if len(stripped_line) == 0:
                continue
            line_list = [entry.strip() for entry in stripped_line.split(SEP_KEY)]
            # if there is no SMILES str, there is no way to properly add any data to the library
            if not line_list[0]:
                warning(f"In reading file: {rel_path_name}\n    Line: '{stripped_line}'\n        does not "
                        f"provide a SMILES string as the first '|'-separated entry. This line will be skipped.")
                continue
            # if there aren't 3 entries, pad with blank strings, as 2nd two are optional
            while len(line_list) < 3:
                line_list.append("")
            if len(line_list) > 3:
                rel_path = os.path.relpath(input_fname)
                raise InvalidDataError(f"Error while reading: {rel_path}\n    line: '{stripped_line}'\n"
                                       f"    Expected no more than 3 comma-separated values: \n        SMILES "
                                       f"string (only one per line),\n        molecule name(s) (separate "
                                       f"multiple names with semicolons),\n        string description of the "
                                       f"data source (with no commas or semicolons)")

            # being explicit in separating out line_list entries; do not change global variables
            new_entry_flag = add_smi_to_dicts(mw_formula_dict, mw_deprot_formula_dict, mw_prot_formula_dict,
                                              form_smi_dict, form_dbe_dict, smi_name_dict, smi_source_dict,
                                              line_list[0],  mol_name=line_list[1], mol_source=line_list[2])
            if new_entry_flag:
                new_entries += 1
    print(f"Completed reading file: {rel_path_name}\n    Added {new_entries} entries to the dictionaries\n")
    return new_entries


def main(argv=None):
    args, ret = parse_cmdline(argv)
    if ret != GOOD_RET or args is None:
        return ret

    try:
        # start with copies of global variable dicts; then only the copies will be altered
        if args.file_name:
            mw_formula_dict = MW_FORM_DICT.copy()
            mw_deprot_formula_dict = MW_DEPROT_FORM_DICT.copy()
            mw_prot_formula_dict = MW_PROT_FORM_DICT.copy()
            form_smi_dict = deepcopy(FORMULA_SMI_DICT)
            form_dbe_dict = FORMULA_DBE_DICT.copy()
            smi_name_dict = deepcopy(SMI_NAME_DICT)
            smi_source_dict = deepcopy(SMI_SOURCE_DICT)

            number_additions = process_input_file(args.file_name, mw_formula_dict, mw_deprot_formula_dict,
                                                  mw_prot_formula_dict, form_smi_dict, form_dbe_dict,
                                                  smi_name_dict, smi_source_dict)

            # Reading complete, now output
            if number_additions:
                pretty_print_dicts(mw_formula_dict, mw_deprot_formula_dict, mw_prot_formula_dict,
                                   form_smi_dict, form_dbe_dict, smi_name_dict, smi_source_dict)

        if args.image_library:
            if args.mw_list:
                mw_keys = [x.strip() for x in args.mw_list.split(",")]
            else:
                mw_keys = MW_FORM_DICT.keys()
            if args.out_dir:
                make_dir(args.out_dir)
            image_grid_mult_mw(mw_keys, MW_FORM_DICT, FORMULA_SMI_DICT, out_dir=args.out_dir)

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
