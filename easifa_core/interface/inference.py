import warnings
warnings.filterwarnings("ignore")

import os
import tempfile
from functools import partial
import sys
from pathlib import Path
from rdkit import Chem
import torch
from dgllife.utils import WeaveAtomFeaturizer, CanonicalBondFeaturizer, mol_to_bigraph

sys.path.append(Path(__file__).resolve().parent.parent.as_posix())

from config import EasIFAInferenceConfig

from common.utils import (
    cuda,
    read_model_state,
    colorize,
    )
from common.residue_constants import validate_and_replace_sequence
from model_structure.enzyme_site_model import (
    EnzymeActiveSiteClsESMGearNetModel,
    EnzymeActiveSiteClsESMModel,
    EnzymeActiveSiteClsSeqModel,
    EnzymeActiveSiteClsModel,
    )
from data_loaders.enzyme_rxn_dataloader import (
    enzyme_rxn_collate_extract,
    ReactionFeatures,
)
from data_loaders.rxn_dataloader import (
    atom_types,
    get_adm,
)
from data_loaders.enzyme_dataloader import (
    MyProtein,
    )
try:
    from common.protein_reader import RobustProtein
    _HAS_ROBUST_PROTEIN = True
except ImportError:
    _HAS_ROBUST_PROTEIN = False


class EasIFAPreprocessingError(Exception):
    """Protein or reaction preprocessing failed."""

class EasIFASequenceTooLongError(Exception):
    """Protein sequence exceeds maximum length."""

class EasIFAModelError(Exception):
    """Model forward pass failed."""

mol_to_graph = partial(mol_to_bigraph, add_self_loop=True)
node_featurizer = WeaveAtomFeaturizer(atom_types=atom_types)
edge_featurizer = CanonicalBondFeaturizer(self_loop=True)

def transform_devices(gpu_allocations):
    devices = {}
    for task, device in gpu_allocations.items():
        if (device == 'cpu') or (device == -1):
            devices[task] = torch.device('cpu')
        else:
            devices[task] = torch.device(f'cuda:{device}')
    return devices


class EasIFAInferenceAPI:
    def __init__(
        self,
        args: EasIFAInferenceConfig,
    ) -> None:
        
        self.args = args
        self.max_enzyme_aa_length = args.max_enzyme_aa_length
        self.gpu_allocations = args.gpu_allocations
        self.device_allocations = transform_devices(args.gpu_allocations)
        self.convert_fn = lambda x: x.tolist() if args.pred_tolist else x

        models = {}

        if "all_features" in args.model_to_use:
            models["all_features"] = EnzymeActiveSiteClsModel(
                rxn_model_path=args.model_state_paths["rxn_model_path"],
                num_active_site_type=args.num_active_site_type,
                viz_warning=False,
            )
        if "wo_structures" in args.model_to_use:
            models["wo_structures"] = EnzymeActiveSiteClsSeqModel(
                rxn_model_path=args.model_state_paths["rxn_model_path"],
                num_active_site_type=args.num_active_site_type,
                viz_warning=False,
            )
        if "wo_reactions" in args.model_to_use:
            models["wo_reactions"]  = EnzymeActiveSiteClsESMGearNetModel(
                num_active_site_type=args.num_active_site_type,
                bridge_hidden_dim=args.bridge_hidden_dim,
            )
        if "wo_rxn_structures" in args.model_to_use:
            models["wo_rxn_structures"] = EnzymeActiveSiteClsESMModel(
                num_active_site_type=args.num_active_site_type,
                bridge_hidden_dim=args.bridge_hidden_dim,
            )

        for model_name in args.model_to_use:
            print('\n' + colorize('#' * 60, 'red'))
            print(colorize("# EasIFA MODEL NAME     :", 'green') + f" {model_name}")
            print(colorize("# CHECKPOINT            :", 'yellow') + f" {args.model_state_paths[model_name]}")
            print(colorize("# DEVICE                :", 'blue') + f" {self.device_allocations[model_name]}")
            print(colorize('#' * 60, 'red') + '\n')

            model = self._read_and_load_model_state(models[model_name], args.model_state_paths[model_name])
            model.to(self.device_allocations[model_name])
            model.eval()
            models[model_name] = model


        self.models = models

        self.model_to_inference = None

    def _read_and_load_model_state(self, model, model_save_path):
        model_state, _ = read_model_state(model_save_path=model_save_path)
        model.load_state_dict(model_state)
        return model


    def _calculate_mol_features(self, smi):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            raise EasIFAPreprocessingError(f"RDKit cannot parse SMILES: '{smi}'")
        fgraph = mol_to_graph(
            mol,
            node_featurizer=node_featurizer,
            edge_featurizer=edge_featurizer,
            canonical_atom_order=False,
        )
        dgraph = get_adm(mol)
        return fgraph, dgraph

    def _calculate_rxn_features(self, rxn):
        if '>>' not in rxn:
            raise EasIFAPreprocessingError(
                f"Invalid reaction SMILES format: '{rxn}'. Expected 'reactant>>product' with '>>' separator."
            )
        try:
            react, prod = rxn.split(">>", 1)
            if not react.strip() or not prod.strip():
                raise EasIFAPreprocessingError(
                    f"Reaction SMILES has empty reactant or product: '{rxn}'"
                )
            react_features_tuple = self._calculate_mol_features(react.strip())
            prod_features_tuple = self._calculate_mol_features(prod.strip())
            return ReactionFeatures(
                (react_features_tuple, prod_features_tuple)
            )
        except EasIFAPreprocessingError:
            raise
        except Exception as e:
            raise EasIFAPreprocessingError(f"Failed to process reaction SMILES '{rxn}': {e}") from e
        
    def _judge_data(self, rxn_smiles=None, enzyme_structure_path=None, enzyme_aa_sequence=None):
        if (enzyme_structure_path is None) and (enzyme_aa_sequence is None):
            raise ValueError("Please provide either enzyme_structure_path or enzyme_aa_sequence")
        if (enzyme_structure_path is not None) and (rxn_smiles is not None):
            self.model_to_inference = "all_features"
        elif (enzyme_structure_path is None) and (enzyme_aa_sequence is not None):
            if rxn_smiles is not None:
                self.model_to_inference = "wo_structures"
            else:
                self.model_to_inference = "wo_rxn_structures"
        elif (enzyme_structure_path is not None) and (rxn_smiles is None):
            self.model_to_inference = "wo_reactions"
        else:
            raise ValueError("Please provide either enzyme_structure_path or enzyme_aa_sequence")
        return True

    def _load_protein_from_pdb(self, path):
        """Load protein from PDB, falling back to RobustProtein on failure."""
        path_str = str(path)
        primary_error = None
        try:
            protein = MyProtein.from_pdb(path_str)
            if protein is not None:
                return protein
        except Exception as e:
            primary_error = e
            warnings.warn(f"MyProtein.from_pdb failed for '{path_str}': {e}, trying RobustProtein fallback")

        if _HAS_ROBUST_PROTEIN:
            try:
                protein = RobustProtein.from_pdb_robust(path_str)
                if protein is not None:
                    if hasattr(protein, 'num_residue') and protein.num_residue > 0:
                        return protein
                    raise EasIFAPreprocessingError(
                        f"RobustProtein loaded '{path_str}' but found 0 residues"
                    )
            except EasIFAPreprocessingError:
                raise
            except Exception as e:
                raise EasIFAPreprocessingError(
                    f"Both MyProtein and RobustProtein failed to load PDB '{path_str}': {e}"
                ) from e

        if primary_error is not None:
            raise EasIFAPreprocessingError(
                f"Failed to load protein from PDB '{path_str}': {primary_error}"
            ) from primary_error
        raise EasIFAPreprocessingError(f"Failed to load protein from PDB '{path_str}': MyProtein.from_pdb returned None")

    def _load_protein_from_sequence(self, seq):
        """Load protein from amino acid sequence, validating and cleaning non-standard characters."""
        try:
            cleaned_seq, warnings_list = validate_and_replace_sequence(seq)
            protein = MyProtein.from_sequence(cleaned_seq)
            if protein is None:
                raise EasIFAPreprocessingError(
                    f"MyProtein.from_sequence returned None for sequence (length={len(cleaned_seq)})"
                )
            return protein
        except ValueError as e:
            raise EasIFAPreprocessingError(
                f"Invalid amino acid sequence: {e}"
            ) from e
        except EasIFAPreprocessingError:
            raise
        except Exception as e:
            raise EasIFAPreprocessingError(
                f"Failed to create protein from sequence (length={len(cleaned_seq)}): {e}"
            ) from e

    def _preprocess_one(self, rxn_smiles=None, enzyme_structure_path=None, enzyme_aa_sequence=None):

        self._judge_data(
            rxn_smiles=rxn_smiles,
            enzyme_structure_path=enzyme_structure_path,
            enzyme_aa_sequence=enzyme_aa_sequence
            )
        if enzyme_structure_path is not None:
            protein = self._load_protein_from_pdb(enzyme_structure_path)
        elif enzyme_aa_sequence is not None:
            protein = self._load_protein_from_sequence(enzyme_aa_sequence)

        if protein is None:
            raise EasIFAPreprocessingError("Failed to load protein: protein object is None")

        if rxn_smiles is not None:
            rxn_fclass = self._calculate_rxn_features(rxn_smiles)
        else:
            rxn_fclass = None
        if hasattr(protein, "residue_feature"):
            with protein.residue():
                protein.residue_feature = protein.residue_feature.to_dense()
        # Clean sequence: RobustProtein.to_sequence() produces "A.L.G..." with dot separators
        raw_seq = protein.to_sequence()
        clean_seq = raw_seq.replace('.', '').replace('\n', '').strip()
        item = {
            "protein_graph": protein,
            "protein_sequence": clean_seq,
        }
        if rxn_fclass is not None:
            item["reaction_graph"] = rxn_fclass
        return item

    def _split_pdb_chains(self, pdb_path):
        """Detect and split a multi-chain PDB into individual chain PDB files.

        Returns:
            List of (chain_id, n_residues, temp_pdb_path) sorted by residue count desc,
            or None if the PDB is single-chain or splitting fails.
        """
        try:
            import MDAnalysis as mda
            u = mda.Universe(str(pdb_path))
            protein = u.select_atoms("protein")
            if len(protein) == 0:
                return None

            chain_ids = sorted(set(protein.residues.segids))
            if len(chain_ids) <= 1:
                return None

            chains = []
            for cid in chain_ids:
                chain_atoms = protein.select_atoms(f"segid {cid}")
                n_res = len(chain_atoms.residues)
                if n_res < 10:
                    warnings.warn(
                        f"Skipping short chain '{cid}' ({n_res} residues) in multi-chain PDB."
                    )
                    continue
                tmp = tempfile.NamedTemporaryFile(
                    suffix='.pdb', delete=False, prefix=f'easifa_{cid}_'
                )
                chain_atoms.write(tmp.name)
                tmp.close()
                chains.append((cid, n_res, tmp.name))

            if not chains:
                return None

            chains.sort(key=lambda x: x[1], reverse=True)
            warnings.warn(
                f"Multi-chain PDB detected ({len(chain_ids)} chains). "
                f"Splitting into {len(chains)} chains for individual prediction: "
                + ", ".join(f"'{c[0]}'({c[1]}res)" for c in chains)
            )
            return chains
        except Exception as e:
            warnings.warn(f"Failed to split chains from '{pdb_path}': {e}")
            return None

    def _calculate_one_data(self, rxn_smiles=None, enzyme_structure_path=None, enzyme_aa_sequence=None):
        data_package = self._preprocess_one(
            rxn_smiles=rxn_smiles,
            enzyme_structure_path=enzyme_structure_path,
            enzyme_aa_sequence=enzyme_aa_sequence,
        )
        self.caculated_sequence = data_package["protein_sequence"]
        seq_len = len(self.caculated_sequence)

        if seq_len > self.max_enzyme_aa_length:
            raise EasIFASequenceTooLongError(
                f"Sequence length {seq_len} exceeds maximum allowed length {self.max_enzyme_aa_length}. "
                f"Consider increasing max_enzyme_aa_length or providing a single-chain PDB."
            )
        batch_one_data = enzyme_rxn_collate_extract([data_package])
        if batch_one_data is None:
            raise EasIFAPreprocessingError(
                "Data collation failed: enzyme_rxn_collate_extract returned None"
            )
        return batch_one_data

    def _inference_single(self, rxn_smiles=None, enzyme_structure_path=None,
                          enzyme_aa_sequence=None, _fallback_attempt=False):
        """Run inference on a single protein chain. Returns (pred, prob)."""
        batch_one_data = self._calculate_one_data(rxn_smiles, enzyme_structure_path, enzyme_aa_sequence)
        device = self.device_allocations[self.model_to_inference]
        if self.model_to_inference not in self.models:
            raise ValueError(
                f"EasIFA Model '{self.model_to_inference}' not found in loaded models "
                f"(available: {list(self.models.keys())}). Please initialize the model first."
            )
        model = self.models[self.model_to_inference]
        if device.type == "cuda":
            batch_one_data = cuda(batch_one_data, device=device)
        try:
            if self.model_to_inference in [
                "wo_reactions",
                "wo_rxn_structures"
                ]:
                protein_node_logic = model(batch_one_data)
            else:
                protein_node_logic, _ = model(batch_one_data)
        except Exception as e:
            # Fix 3: If structure-based model fails, fall back to sequence-only prediction
            failed_model = self.model_to_inference
            if not _fallback_attempt and enzyme_structure_path is not None and failed_model in ("all_features", "wo_reactions"):
                seq = self.caculated_sequence
                if rxn_smiles is not None and "wo_structures" in self.models:
                    warnings.warn(
                        f"Model '{failed_model}' forward pass failed ({e}). "
                        f"Falling back to 'wo_structures' (sequence + reaction) model."
                    )
                    return self._inference_single(
                        rxn_smiles=rxn_smiles, enzyme_aa_sequence=seq,
                        _fallback_attempt=True,
                    )
                elif "wo_rxn_structures" in self.models:
                    warnings.warn(
                        f"Model '{failed_model}' forward pass failed ({e}). "
                        f"Falling back to 'wo_rxn_structures' (sequence only) model."
                    )
                    return self._inference_single(
                        enzyme_aa_sequence=seq,
                        _fallback_attempt=True,
                    )
            raise EasIFAModelError(
                f"Model '{failed_model}' forward pass failed "
                f"(sequence_length={len(self.caculated_sequence)}, device={device}): {e}"
            ) from e
        prob = protein_node_logic.softmax(-1)
        pred = torch.argmax(prob, dim=-1)
        pred = self.convert_fn(pred)
        prob = self.convert_fn(prob)
        return pred, prob

    def _inference_multichain(self, chains, rxn_smiles=None):
        """Predict each chain from a multi-chain PDB separately.

        Args:
            chains: list of (chain_id, n_residues, temp_pdb_path)
            rxn_smiles: optional reaction SMILES

        Returns:
            dict: chain_id -> {"pred": [...], "prob": [...], "sequence": str, "n_residues": int}
        """
        results = {}
        errors = []

        for chain_id, n_res, chain_pdb in chains:
            try:
                pred, prob = self._inference_single(
                    rxn_smiles=rxn_smiles,
                    enzyme_structure_path=chain_pdb,
                )
                results[chain_id] = {
                    "pred": pred,
                    "prob": prob,
                    "sequence": self.caculated_sequence,
                    "n_residues": len(pred),
                    "model_used": self.model_to_inference,
                }
            except Exception as e:
                warnings.warn(
                    f"Chain '{chain_id}' ({n_res} residues) prediction failed: {e}. Skipping."
                )
                errors.append((chain_id, n_res, str(e)))
            finally:
                try:
                    os.unlink(chain_pdb)
                except OSError:
                    pass

        if not results:
            error_summary = "; ".join(f"chain {cid}: {msg}" for cid, _, msg in errors)
            raise EasIFAPreprocessingError(
                f"All {len(chains)} chains failed to predict. Errors: {error_summary}"
            )

        if errors:
            warnings.warn(
                f"Multi-chain prediction: {len(results)}/{len(results)+len(errors)} chains succeeded, "
                f"{len(errors)} failed: "
                + ", ".join(f"'{cid}'({nr}res)" for cid, nr, _ in errors)
            )
        return results

    @torch.no_grad()
    def inference(self, rxn_smiles=None, enzyme_structure_path=None, enzyme_aa_sequence=None):
        """Run inference on a protein.

        For multi-chain PDB files, automatically splits into individual chains
        and predicts each one separately.

        Returns:
            Single-chain PDB or sequence input:
                tuple (pred, prob) where pred is List[int] and prob is List[List[float]]
            Multi-chain PDB (>1 protein chain with >=10 residues):
                dict mapping chain_id -> {"pred": [...], "prob": [...],
                    "sequence": str, "n_residues": int, "model_used": str}
        """
        # Multi-chain detection and splitting
        if enzyme_structure_path is not None:
            chains = self._split_pdb_chains(enzyme_structure_path)
            if chains is not None and len(chains) > 1:
                return self._inference_multichain(chains, rxn_smiles)
            elif chains is not None and len(chains) == 1:
                # Only one meaningful chain extracted — use the cleaned single-chain PDB
                chain_id, n_res, chain_pdb = chains[0]
                try:
                    return self._inference_single(
                        rxn_smiles=rxn_smiles,
                        enzyme_structure_path=chain_pdb,
                    )
                finally:
                    try:
                        os.unlink(chain_pdb)
                    except OSError:
                        pass

        # Single-chain PDB or sequence input
        return self._inference_single(
            rxn_smiles=rxn_smiles,
            enzyme_structure_path=enzyme_structure_path,
            enzyme_aa_sequence=enzyme_aa_sequence,
        )

