import warnings
warnings.filterwarnings("ignore")

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
        fgraph = mol_to_graph(
            mol,
            node_featurizer=node_featurizer,
            edge_featurizer=edge_featurizer,
            canonical_atom_order=False,
        )
        dgraph = get_adm(mol)
        return fgraph, dgraph

    def _calculate_rxn_features(self, rxn):
        try:
            react, prod = rxn.split(">>")
            react_features_tuple = self._calculate_mol_features(react)
            prod_features_tuple = self._calculate_mol_features(prod)
            return ReactionFeatures(
                (react_features_tuple, prod_features_tuple)
            )
        except:
            return None
        
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

    def _preprocess_one(self, rxn_smiles=None, enzyme_structure_path=None, enzyme_aa_sequence=None):

        self._judge_data(
            rxn_smiles=rxn_smiles, 
            enzyme_structure_path=enzyme_structure_path, 
            enzyme_aa_sequence=enzyme_aa_sequence
            )
        if enzyme_structure_path is not None:
            protein = MyProtein.from_pdb(str(enzyme_structure_path))
        elif enzyme_aa_sequence is not None:
            protein = MyProtein.from_sequence(enzyme_aa_sequence)

        if rxn_smiles is not None:
            rxn_fclass = self._calculate_rxn_features(rxn_smiles)
        else:
            rxn_fclass = None
        if hasattr(protein, "residue_feature"):
            with protein.residue():
                protein.residue_feature = protein.residue_feature.to_dense()
        item = {
            "protein_graph": protein,
            "protein_sequence": protein.to_sequence(),
        }
        if rxn_fclass is not None:
            item["reaction_graph"] = rxn_fclass
        return item

    def _calculate_one_data(self, rxn_smiles=None, enzyme_structure_path=None, enzyme_aa_sequence=None):
        data_package = self._preprocess_one(
            rxn_smiles=rxn_smiles, 
            enzyme_structure_path=enzyme_structure_path, 
            enzyme_aa_sequence=enzyme_aa_sequence,
        )
        self.caculated_sequence = data_package["protein_sequence"]
        if len(self.caculated_sequence) > self.max_enzyme_aa_length:
            return None
        batch_one_data = enzyme_rxn_collate_extract([data_package])
        return batch_one_data

    @torch.no_grad()
    def inference(self, rxn_smiles=None, enzyme_structure_path=None, enzyme_aa_sequence=None):
        batch_one_data = self._calculate_one_data(rxn_smiles, enzyme_structure_path, enzyme_aa_sequence)
        if batch_one_data is None:
            return
        device = self.device_allocations[self.model_to_inference]
        try:
            model = self.models[self.model_to_inference]
        except:
            raise ValueError(f"EasIFA Model: {self.model_to_inference} not found in models, please initialize the model first")
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
        except:
            print(f"erro in this data")
            return
        prob = protein_node_logic.softmax(-1)
        pred = torch.argmax(prob, dim=-1)
        pred = self.convert_fn(pred)
        prob = self.convert_fn(prob)
        return pred, prob


