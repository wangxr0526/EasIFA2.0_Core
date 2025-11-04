#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Example usage of EasIFA Core for enzyme active site prediction.

This script demonstrates various ways to use the EasIFA inference API.
"""

from pathlib import Path
from easifa_core import EasIFAInferenceAPI, EasIFAInferenceConfig


def example_1_structure_and_reaction():
    """Example 1: Predict with protein structure and reaction."""
    print("\n" + "="*70)
    print("Example 1: Predict with protein structure and reaction")
    print("="*70)
    
    # Setup configuration
    config = EasIFAInferenceConfig(
        max_enzyme_aa_length=1000,
        gpu_allocations={
            "all_features": 'cpu',
            "wo_structures": 'cpu',
            "wo_reactions": 'cpu',
            "wo_rxn_structures": 'cpu',
        }
    )
    
    # Initialize API
    easifa = EasIFAInferenceAPI(config)
    
    # Example data
    structure_path = "test/test_inferece_input/AF-A0A2K5QMP9-F1-model_v4.pdb"  # Update with actual path
    rxn_smiles = "O.OCC1OC(OC2C(O)C(CO)OC(OC3C(O)C(O)OC(CO)C3O)C2O)C(O)C(O)C1O>>O=C[C@H](O)[C@@H](O)[C@H](O)[C@H](O)CO"
    
    # Run inference
    pred, prob = easifa.inference(
        rxn_smiles=rxn_smiles,
        enzyme_structure_path=structure_path
    )
    
    print(f"Model used: {easifa.model_to_inference}")
    print(f"Sequence length: {len(easifa.caculated_sequence)}")
    print(f"Predicted labels (first 20): {pred[:20]}")
    print(f"Number of predicted active sites: {sum(1 for x in pred if x != 0)}")


def example_2_sequence_and_reaction():
    """Example 2: Predict with amino acid sequence and reaction."""
    print("\n" + "="*70)
    print("Example 2: Predict with amino acid sequence and reaction")
    print("="*70)
    
    config = EasIFAInferenceConfig()
    easifa = EasIFAInferenceAPI(config)
    
    # Example data
    sequence = "MSPRPLRALLGAAAAALVSAAALAFPSQAAANDSPFYVNPNMSSAEWVRNNPNDPRTPVIRDRIASVPQGTWFAHHNPGQITGQVDALMSAAQAAGKIPILVVYNAPGRDCGNHSSGGAPSHSAYRSWIDEFAAGLKNRPAYIIVEPDLISLMSSCMQHVQQEVLETMAYAGKALKAGSSQARIYFDAGHSAWHSPAQMASWLQQADISNSAHGIATNTSNYRWTADEVAYAKAVLSAIGNPSLRAVIDTSRNGNGPAGNEWCDPSGRAIGTPSTTNTGDPMIDAFLWIKLPGEADGCIAGAGQFVPQAAYEMAIAAGGTNPNPNPNPTPTPTPTPTPPPGSSGACTATYTIANEWNDGFQATVTVTANQNITGWTVTWTFTDGQTITNAWNADVSTSGSSVTARNVGHNGTLSQGASTEFGFVGSKGNSNSVPTLTCAAS"
    rxn_smiles = "O.OCC1OC(OC2C(O)C(CO)OC(OC3C(O)C(O)OC(CO)C3O)C2O)C(O)C(O)C1O>>O=C[C@H](O)[C@@H](O)[C@H](O)[C@H](O)CO"
    
    pred, prob = easifa.inference(
        rxn_smiles=rxn_smiles,
        enzyme_aa_sequence=sequence
    )
    
    print(f"Model used: {easifa.model_to_inference}")
    print(f"Sequence length: {len(sequence)}")
    print(f"Predicted labels (first 20): {pred[:20]}")
    print(f"Number of predicted active sites: {sum(1 for x in pred if x != 0)}")


def example_3_structure_only():
    """Example 3: Predict with protein structure only (no reaction)."""
    print("\n" + "="*70)
    print("Example 3: Predict with protein structure only (no reaction)")
    print("="*70)
    
    config = EasIFAInferenceConfig()
    easifa = EasIFAInferenceAPI(config)
    
    structure_path = "test/test_inferece_input/AF-A0A2K5QMP9-F1-model_v4.pdb"
    
    pred, prob = easifa.inference(
        enzyme_structure_path=structure_path
    )
    
    print(f"Model used: {easifa.model_to_inference}")
    print(f"Sequence length: {len(easifa.caculated_sequence)}")
    print(f"Predicted labels (first 20): {pred[:20]}")
    print(f"Number of predicted active sites: {sum(1 for x in pred if x != 0)}")


def example_4_sequence_only():
    """Example 4: Predict with amino acid sequence only (no reaction)."""
    print("\n" + "="*70)
    print("Example 4: Predict with amino acid sequence only (no reaction)")
    print("="*70)
    
    config = EasIFAInferenceConfig()
    easifa = EasIFAInferenceAPI(config)
    
    sequence = "MSPRPLRALLGAAAAALVSAAALAFPSQAAANDSPFYVNPNMSSAEWVRNNPNDPRTPVIRDRIASVPQGTWFAHHNPGQITGQVDALMSAAQAAGKIPILVVYNAPGRDCGNHSSGGAPSHSAYRSWIDEFAAGLKNRPAYIIVEPDLISLMSSCMQHVQQEVLETMAYAGKALKAGSSQARIYFDAGHSAWHSPAQMASWLQQADISNSAHGIATNTSNYRWTADEVAYAKAVLSAIGNPSLRAVIDTSRNGNGPAGNEWCDPSGRAIGTPSTTNTGDPMIDAFLWIKLPGEADGCIAGAGQFVPQAAYEMAIAAGGTNPNPNPNPTPTPTPTPTPPPGSSGACTATYTIANEWNDGFQATVTVTANQNITGWTVTWTFTDGQTITNAWNADVSTSGSSVTARNVGHNGTLSQGASTEFGFVGSKGNSNSVPTLTCAAS"
    
    pred, prob = easifa.inference(
        enzyme_aa_sequence=sequence
    )
    
    print(f"Model used: {easifa.model_to_inference}")
    print(f"Sequence length: {len(sequence)}")
    print(f"Predicted labels (first 20): {pred[:20]}")
    print(f"Number of predicted active sites: {sum(1 for x in pred if x != 0)}")


def example_5_custom_config():
    """Example 5: Custom configuration with GPU."""
    print("\n" + "="*70)
    print("Example 5: Custom configuration with GPU")
    print("="*70)
    
    # Custom configuration
    config = EasIFAInferenceConfig(
        model_to_use=["all_features", "wo_structures"],  # Load only selected models
        max_enzyme_aa_length=500,
        gpu_allocations={
            "all_features": 0,        # Use GPU 0
            "wo_structures": 'cpu',   # Use CPU
        }
    )
    
    print("Configuration:")
    print(f"  Models to use: {config.model_to_use}")
    print(f"  Max sequence length: {config.max_enzyme_aa_length}")
    print(f"  GPU allocations: {config.gpu_allocations}")
    
    easifa = EasIFAInferenceAPI(config)
    
    # Your inference code here
    print("\nAPI initialized successfully!")


def example_6_batch_prediction():
    """Example 6: Batch prediction for multiple proteins."""
    print("\n" + "="*70)
    print("Example 6: Batch prediction for multiple proteins")
    print("="*70)
    
    config = EasIFAInferenceConfig()
    easifa = EasIFAInferenceAPI(config)
    
    # Multiple proteins
    proteins = [
        {
            "id": "protein_1",
            "sequence": "MSPRPLRALLGAAAAALVSAAALAFPSQAAANDSPFYVNPNMSS",
            "rxn_smiles": "CCCO>>CO"
        },
        {
            "id": "protein_2", 
            "sequence": "MSPRPLRALLGAAAAALVSAAALAFPSQAAANDSPFYVNPNMSSAEWVRNNPNDPRT",
            "rxn_smiles": None  # No reaction
        },
    ]
    
    results = []
    for protein in proteins:
        pred, prob = easifa.inference(
            rxn_smiles=protein.get("rxn_smiles"),
            enzyme_aa_sequence=protein["sequence"]
        )
        
        results.append({
            "id": protein["id"],
            "model_used": easifa.model_to_inference,
            "predictions": pred,
            "num_sites": sum(1 for x in pred if x != 0)
        })
        
        print(f"\n{protein['id']}:")
        print(f"  Model: {results[-1]['model_used']}")
        print(f"  Active sites: {results[-1]['num_sites']}")


def main():
    """Run all examples."""
    print("\n" + "="*70)
    print("           EasIFA Core - Usage Examples")
    print("="*70)
    print("\nThese examples demonstrate different ways to use EasIFA.")
    print("Note: Update file paths to match your local setup.")
    
    try:
        # Run examples (comment out examples that need specific files)
        example_1_structure_and_reaction()
        example_2_sequence_and_reaction()
        example_3_structure_only()
        example_4_sequence_only()
        example_5_custom_config()
        example_6_batch_prediction()
        
        print("\n" + "="*70)
        print("All examples completed successfully!")
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"\nError running examples: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
