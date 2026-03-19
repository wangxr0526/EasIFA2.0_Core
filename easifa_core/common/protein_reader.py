import os
import warnings
import numpy as np
import torch
from rdkit import Chem
from tqdm.auto import tqdm
from torchdrug import data
from torchdrug import utils
from torchdrug.core import Registry as R
from copy import copy
import MDAnalysis as mda
import numpy as np
from periodictable import elements

from pathlib import Path

BONDS_1 = {
    'H': {
        'H': 74, 'C': 109, 'N': 101, 'O': 96, 'F': 92,
        'B': 119, 'Si': 148, 'P': 144, 'As': 152, 'S': 134,
        'Cl': 127, 'Br': 141, 'I': 161
    },
    'C': {
        'H': 109, 'C': 154, 'N': 147, 'O': 143, 'F': 135,
        'Si': 185, 'P': 184, 'S': 182, 'Cl': 177, 'Br': 194,
        'I': 214
    },
    'N': {
        'H': 101, 'C': 147, 'N': 145, 'O': 140, 'F': 136,
        'Cl': 175, 'Br': 214, 'S': 168, 'I': 222, 'P': 177
    },
    'O': {
        'H': 96, 'C': 143, 'N': 140, 'O': 148, 'F': 142,
        'Br': 172, 'S': 151, 'P': 163, 'Si': 163, 'Cl': 164,
        'I': 194
    },
    'F': {
        'H': 92, 'C': 135, 'N': 136, 'O': 142, 'F': 142,
        'S': 158, 'Si': 160, 'Cl': 166, 'Br': 178, 'P': 156,
        'I': 187
    },
    'B': {
        'H':  119, 'Cl': 175
    },
    'Si': {
        'Si': 233, 'H': 148, 'C': 185, 'O': 163, 'S': 200,
        'F': 160, 'Cl': 202, 'Br': 215, 'I': 243,
    },
    'Cl': {
        'Cl': 199, 'H': 127, 'C': 177, 'N': 175, 'O': 164,
        'P': 203, 'S': 207, 'B': 175, 'Si': 202, 'F': 166,
        'Br': 214
    },
    'S': {
        'H': 134, 'C': 182, 'N': 168, 'O': 151, 'S': 204,
        'F': 158, 'Cl': 207, 'Br': 225, 'Si': 200, 'P': 210,
        'I': 234
    },
    'Br': {
        'Br': 228, 'H': 141, 'C': 194, 'O': 172, 'N': 214,
        'Si': 215, 'S': 225, 'F': 178, 'Cl': 214, 'P': 222
    },
    'P': {
        'P': 221, 'H': 144, 'C': 184, 'O': 163, 'Cl': 203,
        'S': 210, 'F': 156, 'N': 177, 'Br': 222
    },
    'I': {
        'H': 161, 'C': 214, 'Si': 243, 'N': 222, 'O': 194,
        'S': 234, 'F': 187, 'I': 266
    },
    'As': {
        'H': 152
    }
}

BONDS_2 = {
    'C': {'C': 134, 'N': 129, 'O': 120, 'S': 160},
    'N': {'C': 129, 'N': 125, 'O': 121},
    'O': {'C': 120, 'N': 121, 'O': 121, 'P': 150},
    'P': {'O': 150, 'S': 186},
    'S': {'P': 186}
}

BONDS_3 = {
    'C': {'C': 120, 'N': 116, 'O': 113},
    'N': {'C': 116, 'N': 110},
    'O': {'C': 113}
}

MARGINS_EDM = [10, 5, 2]

# Atom idx for one-hot encoding
ATOM2IDX = {'C': 0, 'O': 1, 'N': 2, 'F': 3, 'S': 4, 'Cl': 5, 'Br': 6, 'I': 7}
IDX2ATOM = {0: 'C', 1: 'O', 2: 'N', 3: 'F', 4: 'S', 5: 'Cl', 6: 'Br', 7: 'I'}

# Atom idx for one-hot encoding
GEOM_ATOM2IDX = {'C': 0, 'O': 1, 'N': 2, 'F': 3, 'S': 4, 'Cl': 5, 'Br': 6, 'I': 7, 'P': 8}
GEOM_IDX2ATOM = {0: 'C', 1: 'O', 2: 'N', 3: 'F', 4: 'S', 5: 'Cl', 6: 'Br', 7: 'I', 8: 'P'}
# Atomic numbers (Z)
CHARGES = {'C': 6, 'O': 8, 'N': 7, 'F': 9, 'S': 16, 'Cl': 17, 'Br': 35, 'I': 53}

from .residue_constants import NONSTANDARD_RESIDUE_MAP

three_to_one = {
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E',
    'PHE': 'F', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N',
    'PRO': 'P', 'GLN': 'Q', 'ARG': 'R', 'SER': 'S',
    'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
}

# Extend three_to_one with non-standard residue mappings
# e.g., MSE -> MET -> 'M', SEP -> SER -> 'S', etc.
for _ns_res, _std_res in NONSTANDARD_RESIDUE_MAP.items():
    if _ns_res not in three_to_one and _std_res in three_to_one:
        three_to_one[_ns_res] = three_to_one[_std_res]

symbol2number = {v.symbol:k for k,v in elements._element.items()}
number2symbol = {v:k for k,v in symbol2number.items()}


def split_protein_chains(pdb_file, output_dir='.', return_file_paths=False):
    """
    拆分PDB中的蛋白质部分，根据不同chain生成多个PDB文件。
    
    参数：
    - pdb_file: 输入的PDB文件路径
    - output_dir: 输出目录，默认为当前目录
    """
    pdb_path = Path(pdb_file).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # 提取 PDB ID（不含扩展名）
    pdb_id = pdb_path.stem

    # 加载结构
    u = mda.Universe(str(pdb_path))

    # 选择蛋白质原子
    protein = u.select_atoms("protein")

    # 获取所有独特的 chainID
    chain_ids = sorted(set(res.segid for res in protein.residues))
    
    if len(chain_ids) == 1:
        return pdb_file.as_posix()
    
    output_files = []
    for chain_id in chain_ids:
        # 提取对应链的原子
        chain_atoms = protein.select_atoms(f"protein and chainID {chain_id}")
        if len(chain_atoms) == 0:
            continue

        # 生成输出文件路径：如 1abc+A.pdb
        output_file = output_dir / f"{pdb_id}_{chain_id}.pdb"
        chain_atoms.write(str(output_file))
        print(f"Saved chain {chain_id} to {output_file}")
        output_files.append(output_file.as_posix())

    if return_file_paths:
        return output_files


def get_bond_order(atom1, atom2, distance, check_exists=True, margins=MARGINS_EDM):
    distance = 100 * distance  # We change the metric

    # Check exists for large molecules where some atom pairs do not have a
    # typical bond length.
    if check_exists:
        if atom1 not in BONDS_1:
            return 0
        if atom2 not in BONDS_1[atom1]:
            return 0

    # margin1, margin2 and margin3 have been tuned to maximize the stability of the QM9 true samples
    if distance < BONDS_1[atom1][atom2] + margins[0]:

        # Check if atoms in bonds2 dictionary.
        if atom1 in BONDS_2 and atom2 in BONDS_2[atom1]:
            thr_bond2 = BONDS_2[atom1][atom2] + margins[1]
            if distance < thr_bond2:
                if atom1 in BONDS_3 and atom2 in BONDS_3[atom1]:
                    thr_bond3 = BONDS_3[atom1][atom2] + margins[2]
                    if distance < thr_bond3:
                        return 3  # Triple
                return 2  # Double
        return 1  # Single
    return 0  # No bond

def build_xae_molecule(positions, atom_types, margins=MARGINS_EDM):
    """ Returns a triplet (X, A, E): atom_types, adjacency matrix, edge_types
        args:
        positions: N x 3  (already masked to keep final number nodes)
        atom_types: N
        returns:
        X: N         (int)
        A: N x N     (bool) (binary adjacency matrix)
        E: N x N     (int)  (bond type, 0 if no bond) such that A = E.bool()
    """
    n = positions.shape[0]
    X = atom_types
    A = torch.zeros((n, n), dtype=torch.bool)
    E = torch.zeros((n, n), dtype=torch.int)



    pos = positions.unsqueeze(0)
    dists = torch.cdist(pos, pos, p=2).squeeze(0)
    for i in range(n):
        for j in range(i):

            pair = sorted([atom_types[i], atom_types[j]])
            order = get_bond_order(
                number2symbol[pair[0]], 
                number2symbol[pair[1]], 
                dists[i, j], 
                margins=margins
                )

            # TODO: a batched version of get_bond_order to avoid the for loop
            if order > 0:
                # Warning: the graph should be DIRECTED
                A[i, j] = 1
                E[i, j] = order

    return X, A, E

def create_cc_universe():
    u = mda.Universe.empty(
        n_atoms=2,
        n_residues=1,
        atom_resindex=[0, 0],
        residue_segindex=[0],
        trajectory=True 
    )
    u.add_TopologyAttr("name", ["C1", "C2"])
    u.add_TopologyAttr("type", ["C", "C"])
    u.add_TopologyAttr("element", ["C", "C"])
    u.add_TopologyAttr("resname", ["ALA"])
    u.add_TopologyAttr("resid", [1])
    u.add_TopologyAttr("segid", ["A"])
    u.atoms.positions = np.array([[0.0, 0.0, 0.0], [1.54, 0.0, 0.0]])
    return u

class RobustProteinMol:
    # Element types that cause MDAnalysis guess_bonds() to crash (no VDW radii)
    # Map to chemically similar standard elements
    _ELEMENT_REMAP = {
        'SE': 'S',   # Selenium → Sulfur (selenomethionine)
        'E': 'S',    # Misread SE → Sulfur
    }
    # Metal element types that should NOT participate in covalent bond guessing
    _METAL_TYPES = {
        'FE', 'ZN', 'MG', 'CA', 'CU', 'MN', 'CO', 'NI', 'NA', 'K',
        'MO', 'CD', 'W', 'V', 'CR', 'HG', 'PT', 'AU', 'AG',
    }

    def __init__(self, pdb_file=None, mol=None):
        if mol is not None:
            self.u = mol
        elif pdb_file is not None:
            self.u = mda.Universe(pdb_file)
        else:
            self.u = mda.Universe.empty(n_atoms=0)

        # Use "protein" selector to properly exclude water, ligands, and metals.
        # Also include known non-standard amino acid residues (MSE, SEP, etc.)
        nonstd_resnames = list(NONSTANDARD_RESIDUE_MAP.keys())
        sel_str = "protein"
        if nonstd_resnames:
            sel_str += " or resname " + " ".join(nonstd_resnames)
        try:
            selected = self.u.select_atoms(sel_str)
        except Exception:
            # Fallback: select all non-solvent atoms
            try:
                selected = self.u.select_atoms("not resname HOH WAT DOD TIP TIP3 SOL")
            except Exception:
                selected = self.u.atoms
        self.atoms = selected.atoms
        self.residues = selected.residues

        # 缓存属性
        self._atom_names = None
        self._atom_type = None
        self._atom2residue = None
        self._num_residue = None
        self._num_node = None
        self._chain_ids = None
        self._atom_positions = None
        self._residue_positions = None
        self._residue_types = None
        self._sequence = None
        
        self._bond_orders = None

    @property
    def atom_type(self):
        if self._atom_type is None:
            self._atom_type = []
            for atom in self.atoms:
                elem = atom.element.capitalize() if atom.element else None
                self._atom_type.append(symbol2number.get(elem, 0))
        return self._atom_type
    
    @property
    def atom_names(self):
        if self._atom_names is None:
            self._atom_names = self.atoms.names
        return self._atom_names

    @property
    def atom2residue(self):
        if self._atom2residue is None:
            self._atom2residue = [atom.resid for atom in self.atoms]
        return self._atom2residue

    @property
    def num_residue(self):
        if self._num_residue is None:
            self._num_residue = len(self.residues)
        return self._num_residue

    @property
    def atom_positions(self):
        if self._atom_positions is None:
            self._atom_positions = self.atoms.positions
        return self._atom_positions

    @property
    def residue_positions(self):
        if self._residue_positions is None:
            pos = []
            for res in self.residues:
                ca_atoms = res.atoms.select_atoms("name CA")
                if len(ca_atoms) == 1:
                    pos.append(ca_atoms.positions[0])
                else:
                    pos.append(np.array([np.nan, np.nan, np.nan]))
            self._residue_positions = np.array(pos)
        return self._residue_positions

    @property
    def num_node(self):
        # 默认节点为原子
        if self._num_node is None:
            self._num_node = len(self.atoms)
        return self._num_node

    @property
    def chain_ids(self):
        if self._chain_ids is None:
            self._chain_ids = [res.segid for res in self.residues]
        return self._chain_ids

    @property
    def residue_types(self):
        if self._residue_types is None:
            self._residue_types = [res.resname for res in self.residues]
        return self._residue_types

    @property
    def sequence(self):
        if self._sequence is None:
            self._sequence = self._get_protein_sequence()
        return self._sequence
    
    def _get_protein_sequence(self):
        residues = self.residues
        chain_ids = residues.segids
        resnames = residues.resnames

        sequence = ""
        prev_chain = None

        for chain, resname in zip(chain_ids, resnames):
            if chain != prev_chain and prev_chain is not None:
                sequence += "\n"  # 分隔不同链
            sequence += three_to_one.get(resname, 'X')  # 未知残基标记为X
            prev_chain = chain

        return sequence
    def _sanitize_atom_types_for_bonds(self):
        """Remap problematic element/type attributes so guess_bonds() can find VDW radii."""
        remapped = []
        for atom in self.u.atoms:
            atype = (atom.type or '').strip().upper()
            if atype in self._ELEMENT_REMAP:
                new_type = self._ELEMENT_REMAP[atype]
                remapped.append((atom.index, atom.type, atom.element))
                atom.type = new_type
                atom.element = new_type
        return remapped

    def _restore_atom_types(self, saved):
        """Restore original element/type attributes after bond guessing."""
        for idx, orig_type, orig_elem in saved:
            self.u.atoms[idx].type = orig_type
            self.u.atoms[idx].element = orig_elem

    def _bondable_element(self, atom):
        """Return a valid element symbol for bond order lookup, or None for metals/unknowns."""
        elem = (getattr(atom, 'element', '') or '').strip()
        if len(elem) > 1:
            elem = elem[0].upper() + elem[1:].lower()
        else:
            elem = elem.upper()
        if elem in BONDS_1:
            return elem
        upper = elem.upper()
        if upper in self._ELEMENT_REMAP:
            return self._ELEMENT_REMAP[upper]
        if upper in self._METAL_TYPES:
            return None
        # Try to infer from atom name
        name = (getattr(atom, 'name', '') or '').strip().upper()
        for prefix in ('C', 'N', 'O', 'S', 'H', 'P'):
            if name.startswith(prefix):
                return prefix
        return None

    def _fallback_bond_orders(self, dists):
        """Distance-based bond inference when MDAnalysis guess_bonds() fails."""
        bond_orders = []
        n = len(self.atoms)
        # Only check pairs within max single bond distance (~2.5 Å)
        close_mask = (dists < 2.5) & (dists > 0.1)
        pairs = close_mask.nonzero(as_tuple=False)
        for pair in pairs:
            i, j = pair[0].item(), pair[1].item()
            if i >= j:
                continue
            elem_i = self._bondable_element(self.atoms[i])
            elem_j = self._bondable_element(self.atoms[j])
            if elem_i is None or elem_j is None:
                continue
            order = get_bond_order(elem_i, elem_j, distance=dists[i, j].item())
            if order > 0:
                bond_orders.append((i, j, order))
        return bond_orders

    @property
    def bond_orders(self):
        if self._bond_orders is None:
            pos = torch.tensor(self.atom_positions, dtype=torch.float32).unsqueeze(0)
            dists = torch.cdist(pos, pos, p=2).squeeze(0)

            saved_types = self._sanitize_atom_types_for_bonds()
            try:
                self.u.atoms.guess_bonds()
                bond_orders = []
                for bond in self.u.bonds:
                    i, j = bond.indices
                    elem_i = bond.atoms[0].element
                    elem_j = bond.atoms[1].element
                    dist = dists[i, j].item()
                    order = get_bond_order(elem_i, elem_j, distance=dist)
                    bond_orders.append((i, j, order))
                self._bond_orders = bond_orders
            except Exception as e:
                warnings.warn(
                    f"MDAnalysis guess_bonds() failed: {e}. "
                    f"Using distance-based bond inference as fallback."
                )
                self._bond_orders = self._fallback_bond_orders(dists)
            finally:
                self._restore_atom_types(saved_types)
        return self._bond_orders

    # --- RDKit-compatible adapter methods for RobustMolecule.from_molecule_robust ---

    def GetNumBonds(self):
        """Return the number of bonds (RDKit-compatible API)."""
        return len(self.bond_orders)

    def GetBondWithIdx(self, idx):
        """Return a RobustBond adapter for the given bond index (RDKit-compatible API)."""
        i, j, order = self.bond_orders[idx]
        return RobustBond(i, j, order)

    def GetNumAtoms(self):
        """Return the number of atoms (RDKit-compatible API)."""
        return len(self.atoms)

    def GetAtomWithIdx(self, idx):
        """Return atom at given index, wrapped with RDKit-compatible API."""
        return RobustAtomWrapper(self.atoms[idx])

    def node_positions(self, use_residue_as_node=False):
        """Return atom or residue positions."""
        if use_residue_as_node:
            return self.residue_positions
        return self.atom_positions


_BOND_ORDER_TO_RDKIT_TYPE = {
    1: Chem.rdchem.BondType.SINGLE,
    2: Chem.rdchem.BondType.DOUBLE,
    3: Chem.rdchem.BondType.TRIPLE,
}


class RobustBond:
    """Adapter that mimics RDKit Bond API for use with torchdrug's Molecule graph construction."""

    def __init__(self, begin_idx, end_idx, order):
        self._begin = begin_idx
        self._end = end_idx
        self._order = order

    def GetBeginAtomIdx(self):
        return self._begin

    def GetEndAtomIdx(self):
        return self._end

    def GetBondType(self):
        return _BOND_ORDER_TO_RDKIT_TYPE.get(self._order, Chem.rdchem.BondType.SINGLE)

    def GetBondTypeAsDouble(self):
        return float(self._order)

    def GetBondDir(self):
        return 0  # NONE

    def GetStereo(self):
        return 0  # STEREONONE

    def GetStereoAtoms(self):
        return [0, 0]

    def GetIsAromatic(self):
        return False

    def GetIsConjugated(self):
        return False


class RobustPDBResidueInfo:
    """Adapter that mimics RDKit PDBResidueInfo for MDAnalysis atoms."""

    def __init__(self, atom):
        self._atom = atom

    def GetResidueName(self):
        return f"{self._atom.resname:>3s}"

    def GetResidueNumber(self):
        return int(self._atom.resid)

    def GetInsertionCode(self):
        icode = getattr(self._atom, 'icode', ' ')
        return icode if icode else ' '

    def GetChainId(self):
        return self._atom.segid if self._atom.segid else 'A'

    def GetName(self):
        name = self._atom.name
        # PDB atom name is 4 chars, left-padded for 1-char elements
        if len(name) < 4:
            name = f" {name:<3s}"
        return name

    def GetIsHeteroAtom(self):
        return False  # RobustProteinMol already filters HETATM

    def GetOccupancy(self):
        return getattr(self._atom, 'occupancy', 1.0)

    def GetTempFactor(self):
        return getattr(self._atom, 'tempfactor', 0.0)


class RobustAtomWrapper:
    """Wraps an MDAnalysis atom to provide RDKit-compatible interface."""

    def __init__(self, atom):
        self._atom = atom
        self._pdbinfo = RobustPDBResidueInfo(atom)

    def GetPDBResidueInfo(self):
        return self._pdbinfo

    def GetAtomicNum(self):
        return symbol2number.get(self._atom.element.capitalize(), 0)

class RobustMolecule(data.Molecule):
    empty_mol = RobustProteinMol(None)
    dummy_mol = RobustProteinMol(mol=create_cc_universe())
    @classmethod
    @utils.deprecated_alias(node_feature="atom_feature", edge_feature="bond_feature", graph_feature="mol_feature")
    def from_molecule_robust(cls, mol:RobustProteinMol, atom_feature="default", bond_feature="default", mol_feature=None,
                      with_hydrogen=False, kekulize=False):
        """
        Create a molecule from an RDKit object.

        Parameters:
            mol (rdchem.Mol): molecule
            atom_feature (str or list of str, optional): atom features to extract
            bond_feature (str or list of str, optional): bond features to extract
            mol_feature (str or list of str, optional): molecule features to extract
            with_hydrogen (bool, optional): store hydrogens in the molecule graph.
                By default, hydrogens are dropped
            kekulize (bool, optional): convert aromatic bonds to single/double bonds.
                Note this only affects the relation in ``edge_list``.
                For ``bond_type``, aromatic bonds are always stored explicitly.
                By default, aromatic bonds are stored.
        """
        if mol is None:
            mol = cls.empty_mol
        # some RDKit operations are in-place
        # copy the object to avoid undesired behavior in the caller
        mol = copy(mol)
        # if with_hydrogen:
        #     mol = Chem.AddHs(mol)
        # if kekulize:
        #     Chem.Kekulize(mol)

        atom_feature = cls._standarize_option(atom_feature)
        bond_feature = cls._standarize_option(bond_feature)
        mol_feature = cls._standarize_option(mol_feature)

        atom_type = mol.atom_type
        # formal_charge = []
        # explicit_hs = []
        # chiral_tag = []
        # radical_electrons = []
        # atom_map = []
        _atom_feature = []
        dummy_atom = copy(cls.dummy_mol).atoms[0]
        atoms = [atom for atom in mol.atoms] + [dummy_atom]
        try:
            node_position = torch.tensor(mol.node_positions(use_residue_as_node=False))
        except:
            node_position = None
        # for atom in atoms:
            # atom_type.append(atom.GetAtomicNum())
            # formal_charge.append(atom.GetFormalCharge())
            # explicit_hs.append(atom.GetNumExplicitHs())
            # chiral_tag.append(atom.GetChiralTag())
            # radical_electrons.append(atom.GetNumRadicalElectrons())
            # atom_map.append(atom.GetAtomMapNum())
            # feature = []
            # for name in atom_feature:
            #     func = R.get("features.atom.%s" % name)
            #     feature += func(atom)
            # _atom_feature.append(feature)
        atom_type = torch.tensor(atom_type)
        # atom_map = torch.tensor(atom_map)[:-1]
        # formal_charge = torch.tensor(formal_charge)[:-1]
        # explicit_hs = torch.tensor(explicit_hs)[:-1]
        # chiral_tag = torch.tensor(chiral_tag)[:-1]
        # radical_electrons = torch.tensor(radical_electrons)[:-1]
        # if len(atom_feature) > 0:
        #     _atom_feature = torch.tensor(_atom_feature)[:-1]
        # else:
        _atom_feature = None

        edge_list = []
        bond_type = []
        # bond_stereo = []
        # stereo_atoms = []
        _bond_feature = []
        dummy_bond = copy(cls.dummy_mol).GetBondWithIdx(0)
        bonds = [mol.GetBondWithIdx(i) for i in range(mol.GetNumBonds())] + [dummy_bond]
        for bond in bonds:
            btype = bond.GetBondType()
            # stereo = bond.GetStereo()
            # if stereo:
            #     _atoms = [a for a in bond.GetStereoAtoms()]
            # else:
            #     _atoms = [0, 0]
            if btype not in cls.bond2id:
                continue
            btype_id = cls.bond2id[btype]
            h, t = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            edge_list += [[h, t, btype_id], [t, h, btype_id]]
            # always explicitly store aromatic bonds, no matter kekulize or not
            # if bond.GetIsAromatic():
            #     type = cls.bond2id["AROMATIC"]
            # bond_type += [type, type]
            # bond_stereo += [stereo, stereo]
            # stereo_atoms += [_atoms, _atoms]
            feature = []
            for name in bond_feature:
                func = R.get("features.bond.%s" % name)
                feature += func(bond)
            _bond_feature += [feature, feature]
        edge_list = edge_list[:-2]
        bond_type = torch.tensor(bond_type)[:-2]
        # bond_stereo = torch.tensor(bond_stereo)[:-2]
        # stereo_atoms = torch.tensor(stereo_atoms)[:-2]
        if len(bond_feature) > 0:
            _bond_feature = torch.tensor(_bond_feature)[:-2]
        else:
            _bond_feature = None

        _mol_feature = []
        for name in mol_feature:
            func = R.get("features.molecule.%s" % name)
            _mol_feature += func(mol)
        if len(mol_feature) > 0:
            _mol_feature = torch.tensor(_mol_feature)
        else:
            _mol_feature = None

        num_relation = len(cls.bond2id) - 1 if kekulize else len(cls.bond2id)
        return cls(edge_list, atom_type, bond_type,
                #    formal_charge=formal_charge, explicit_hs=explicit_hs,
                #    chiral_tag=chiral_tag, radical_electrons=radical_electrons, atom_map=atom_map,
                #    bond_stereo=bond_stereo, stereo_atoms=stereo_atoms, 
                   node_position=node_position,
                   atom_feature=_atom_feature, bond_feature=_bond_feature, 
                #    mol_feature=_mol_feature,
                   num_node=len(mol.atoms), num_relation=num_relation)




class RobustProtein(data.Protein):
    
    @classmethod
    @utils.deprecated_alias(node_feature="atom_feature", edge_feature="bond_feature", graph_feature="mol_feature")
    def from_pdb_robust(cls, pdb_file, atom_feature="default", bond_feature="default", residue_feature="default",
                 mol_feature=None, kekulize=False):
        """
        Create a protein from a PDB file.

        Parameters:
            pdb_file (str): file name
            atom_feature (str or list of str, optional): atom features to extract
            bond_feature (str or list of str, optional): bond features to extract
            residue_feature (str, list of str, optional): residue features to extract
            mol_feature (str or list of str, optional): molecule features to extract
            kekulize (bool, optional): convert aromatic bonds to single/double bonds.
                Note this only affects the relation in ``edge_list``.
                For ``bond_type``, aromatic bonds are always stored explicitly.
                By default, aromatic bonds are stored.
        """
        if not os.path.exists(pdb_file):
            raise FileNotFoundError("No such file `%s`" % pdb_file)
        mol = RobustProteinMol(pdb_file)
        # if mol is None:
        #     raise ValueError("RDKit cannot read PDB file `%s`" % pdb_file)
        return cls.from_molecule_robust(mol, atom_feature, bond_feature, residue_feature, mol_feature, kekulize)
    
    @classmethod
    @utils.deprecated_alias(node_feature="atom_feature", edge_feature="bond_feature", graph_feature="mol_feature")
    def from_molecule_robust(cls, mol, atom_feature="default", bond_feature="default", residue_feature="default",
                      mol_feature=None, kekulize=False):
        """
        Create a protein from an RDKit object.

        Parameters:
            mol (rdchem.Mol): molecule
            atom_feature (str or list of str, optional): atom features to extract
            bond_feature (str or list of str, optional): bond features to extract
            residue_feature (str, list of str, optional): residue features to extract
            mol_feature (str or list of str, optional): molecule features to extract
            kekulize (bool, optional): convert aromatic bonds to single/double bonds.
                Note this only affects the relation in ``edge_list``.
                For ``bond_type``, aromatic bonds are always stored explicitly.
                By default, aromatic bonds are stored.
        """
        protein = RobustMolecule.from_molecule_robust(mol, atom_feature=atom_feature, bond_feature=bond_feature,
                                         mol_feature=mol_feature, with_hydrogen=False, kekulize=kekulize)
        residue_feature = cls._standarize_option(residue_feature)

        if kekulize:
            Chem.Kekulize(mol)

        residue_type = []
        atom_name = []
        is_hetero_atom = []
        occupancy = []
        b_factor = []
        atom2residue = []
        residue_number = []
        insertion_code = []
        chain_id = []
        _residue_feature = []
        last_residue = None
        atoms = [mol.GetAtomWithIdx(i) for i in range(mol.GetNumAtoms())] + [cls.dummy_atom]
        for atom in atoms:
            pdbinfo = atom.GetPDBResidueInfo()
            number = pdbinfo.GetResidueNumber()
            code = pdbinfo.GetInsertionCode()
            type = pdbinfo.GetResidueName().strip()
            canonical_residue = (number, code, type)
            if canonical_residue != last_residue:
                last_residue = canonical_residue
                if type not in cls.residue2id:
                    if type in NONSTANDARD_RESIDUE_MAP:
                        mapped = NONSTANDARD_RESIDUE_MAP[type]
                        warnings.warn(f"Non-standard residue `{type}` mapped to `{mapped}`")
                        type = mapped
                    else:
                        warnings.warn("Unknown residue `%s`. Treat as glycine" % type)
                        type = "GLY"
                residue_type.append(cls.residue2id[type])
                residue_number.append(number)
                ins_code = pdbinfo.GetInsertionCode()
                ch_id = pdbinfo.GetChainId()
                if ins_code not in cls.alphabet2id:
                    warnings.warn(f"Unknown insertion code `{ins_code}`, replacing with ' '")
                    ins_code = ' '
                if ch_id not in cls.alphabet2id:
                    warnings.warn(f"Unknown chain ID `{ch_id}`, replacing with 'A'")
                    ch_id = 'A'
                insertion_code.append(cls.alphabet2id[ins_code])
                chain_id.append(cls.alphabet2id[ch_id])
                feature = []
                for name in residue_feature:
                    func = R.get("features.residue.%s" % name)
                    feature += func(pdbinfo)
                _residue_feature.append(feature)
            name = pdbinfo.GetName().strip()
            if name not in cls.atom_name2id:
                name = "UNK"
            atom_name.append(cls.atom_name2id[name])
            is_hetero_atom.append(pdbinfo.GetIsHeteroAtom())
            occupancy.append(pdbinfo.GetOccupancy())
            b_factor.append(pdbinfo.GetTempFactor())
            atom2residue.append(len(residue_type) - 1)
        residue_type = torch.tensor(residue_type)[:-1]
        atom_name = torch.tensor(atom_name)[:-1]
        is_hetero_atom = torch.tensor(is_hetero_atom)[:-1]
        occupancy = torch.tensor(occupancy)[:-1]
        b_factor = torch.tensor(b_factor)[:-1]
        atom2residue = torch.tensor(atom2residue)[:-1]
        residue_number = torch.tensor(residue_number)[:-1]
        insertion_code = torch.tensor(insertion_code)[:-1]
        chain_id = torch.tensor(chain_id)[:-1]
        if len(residue_feature) > 0:
            _residue_feature = torch.tensor(_residue_feature)[:-1]
        else:
            _residue_feature = None

        return cls(protein.edge_list, num_node=protein.num_node, residue_type=residue_type,
                   atom_name=atom_name, atom2residue=atom2residue, residue_feature=_residue_feature,
                   is_hetero_atom=is_hetero_atom, occupancy=occupancy, b_factor=b_factor,
                   residue_number=residue_number, insertion_code=insertion_code, chain_id=chain_id,
                   meta_dict=protein.meta_dict, **protein.data_dict)




if __name__ == '__main__':
    from pathlib import Path
    from tqdm import tqdm
    
    test_data_path = Path('../dataset/ec_site_dataset/structures/pdb_download').resolve()
    pdb_fnames = list(test_data_path.glob('*.pdb'))
    pdb_fnames = sorted(pdb_fnames, key=lambda x:str(x))
    batch = []
    for fpath in tqdm(pdb_fnames[:1000]):
        # protein = RobustProtein.from_pdb_robust(fpath)
        # protein._build_xae_molecule()
        mol = RobustProteinMol(fpath)
        bond_orders = mol.bond_orders



        pass