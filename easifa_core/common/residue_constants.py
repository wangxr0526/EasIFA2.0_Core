"""
Centralized non-standard residue mappings and validation logic.

Provides biochemically-informed mappings from non-standard amino acids
to their closest standard counterparts, rather than defaulting everything to GLY.
"""

import warnings

# Three-letter non-standard amino acid -> standard amino acid mapping
NONSTANDARD_RESIDUE_MAP = {
    # Histidine variants
    'HIE': 'HIS', 'HID': 'HIS', 'HIP': 'HIS',
    'HSE': 'HIS', 'HSD': 'HIS', 'HSP': 'HIS',
    # Selenomethionine (very common in crystal structures)
    'MSE': 'MET',
    # Phosphorylated residues
    'SEP': 'SER', 'TPO': 'THR', 'PTR': 'TYR',
    # Oxidized cysteine
    'CSO': 'CYS', 'CSD': 'CYS', 'CSS': 'CYS', 'OCS': 'CYS', 'CME': 'CYS',
    # Hydroxyproline
    'HYP': 'PRO',
    # Modified lysine
    'MLY': 'LYS', 'M3L': 'LYS', 'KCX': 'LYS',
    # Modified glutamate
    'PCA': 'GLU', 'CGU': 'GLU',
    # D-amino acids
    'DAL': 'ALA', 'DLE': 'LEU', 'DVA': 'VAL',
    # Others
    'AIB': 'ALA', 'ABA': 'ALA', 'SAR': 'GLY',
    'NLE': 'LEU', 'SEC': 'CYS', 'PYL': 'LYS',
}

# One-letter non-standard character -> standard character mapping (for sequence input)
NONSTANDARD_ONELETTER_MAP = {
    'X': 'G',   # Unknown -> GLY (consistent with existing training behavior)
    'B': 'D',   # ASX -> ASP
    'Z': 'E',   # GLX -> GLU
    'U': 'C',   # SEC -> CYS
    'O': 'K',   # PYL -> LYS
    'J': 'L',   # XLE -> LEU
}

STANDARD_AMINO_ACIDS = set('ACDEFGHIKLMNPQRSTVWY')


def validate_and_replace_residues_3letter(residue_names, threshold=0.2):
    """
    Validate and replace non-standard three-letter residue names.

    Uses NONSTANDARD_RESIDUE_MAP for biochemically-informed replacements.
    Unknown residues not in the map are replaced with GLY.

    Args:
        residue_names: list of three-letter residue name strings
        threshold: maximum fraction of non-standard residues allowed (default 0.2)

    Returns:
        tuple: (cleaned_names, replacement_records, nonstandard_count, total_count)

    Raises:
        ValueError: if non-standard residue fraction exceeds threshold
    """
    from torchdrug.data import Protein
    standard_residues = set(Protein.residue2id.keys())

    cleaned_names = []
    replacement_records = []
    nonstandard_count = 0
    total_count = len(residue_names)

    for i, name in enumerate(residue_names):
        if name in standard_residues:
            cleaned_names.append(name)
        elif name in NONSTANDARD_RESIDUE_MAP:
            replacement = NONSTANDARD_RESIDUE_MAP[name]
            cleaned_names.append(replacement)
            replacement_records.append(
                f"Residue {i+1}: {name} -> {replacement} (biochemical mapping)"
            )
            nonstandard_count += 1
        else:
            cleaned_names.append('GLY')
            replacement_records.append(
                f"Residue {i+1}: {name} -> GLY (unknown residue)"
            )
            nonstandard_count += 1

    if replacement_records:
        warnings.warn(
            f"Replaced {nonstandard_count}/{total_count} non-standard residues: "
            + "; ".join(replacement_records[:5])
            + (f" ... and {len(replacement_records)-5} more" if len(replacement_records) > 5 else "")
        )

    if total_count > 0 and nonstandard_count / total_count > threshold:
        raise ValueError(
            f"Too many non-standard residues: {nonstandard_count}/{total_count} "
            f"({nonstandard_count/total_count:.1%}) exceeds threshold {threshold:.0%}. "
            f"Replacements: {'; '.join(replacement_records[:10])}"
        )

    return cleaned_names, replacement_records, nonstandard_count, total_count


def validate_and_replace_sequence(sequence, threshold=0.2):
    """
    Validate and replace non-standard characters in a one-letter amino acid sequence.

    Args:
        sequence: amino acid sequence string (one-letter codes)
        threshold: maximum fraction of non-standard characters allowed (default 0.2)

    Returns:
        tuple: (cleaned_sequence, warnings_list)

    Raises:
        ValueError: if non-standard character fraction exceeds threshold
    """
    cleaned = []
    warnings_list = []
    nonstandard_count = 0

    for i, char in enumerate(sequence):
        upper_char = char.upper()
        if upper_char in STANDARD_AMINO_ACIDS:
            cleaned.append(upper_char)
        elif upper_char in NONSTANDARD_ONELETTER_MAP:
            replacement = NONSTANDARD_ONELETTER_MAP[upper_char]
            cleaned.append(replacement)
            warnings_list.append(f"Position {i+1}: '{upper_char}' -> '{replacement}'")
            nonstandard_count += 1
        else:
            cleaned.append('G')
            warnings_list.append(f"Position {i+1}: '{char}' -> 'G' (unknown)")
            nonstandard_count += 1

    if warnings_list:
        warnings.warn(
            f"Replaced {nonstandard_count}/{len(sequence)} non-standard characters in sequence: "
            + "; ".join(warnings_list[:5])
            + (f" ... and {len(warnings_list)-5} more" if len(warnings_list) > 5 else "")
        )

    if len(sequence) > 0 and nonstandard_count / len(sequence) > threshold:
        raise ValueError(
            f"Too many non-standard characters in sequence: {nonstandard_count}/{len(sequence)} "
            f"({nonstandard_count/len(sequence):.1%}) exceeds threshold {threshold:.0%}. "
            f"Details: {'; '.join(warnings_list[:10])}"
        )

    return ''.join(cleaned), warnings_list
