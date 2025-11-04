#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Command-line interface for EasIFA inference.
"""

import argparse
import json
import sys
from pathlib import Path

try:
    from easifa_core import EasIFAInferenceAPI, EasIFAInferenceConfig
except ImportError:
    # If running from source without installation
    import os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from easifa_core import EasIFAInferenceAPI, EasIFAInferenceConfig


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='EasIFA: Enzyme Active Site Prediction',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Predict with structure and reaction:
  %(prog)s --enzyme-structure protein.pdb --rxn-smiles "reactant>>product" --output result.json
  
  # Predict with sequence and reaction:
  %(prog)s --enzyme-sequence "MSPRL..." --rxn-smiles "reactant>>product" --output result.json
  
  # Predict with structure only (no reaction):
  %(prog)s --enzyme-structure protein.pdb --output result.json
  
  # Predict with sequence only (no reaction):
  %(prog)s --enzyme-sequence "MSPRL..." --output result.json
        '''
    )
    
    # Input options
    input_group = parser.add_argument_group('Input Options')
    input_group.add_argument(
        '--enzyme-structure',
        type=str,
        default=None,
        help='Path to enzyme structure file (PDB format)'
    )
    input_group.add_argument(
        '--enzyme-sequence',
        type=str,
        default=None,
        help='Enzyme amino acid sequence (one-letter code)'
    )
    input_group.add_argument(
        '--rxn-smiles',
        type=str,
        default=None,
        help='Reaction SMILES string (format: reactant>>product)'
    )
    
    # Output options
    output_group = parser.add_argument_group('Output Options')
    output_group.add_argument(
        '--output',
        '-o',
        type=str,
        default='easifa_result.json',
        help='Output file path (JSON format) [default: easifa_result.json]'
    )
    output_group.add_argument(
        '--pretty',
        action='store_true',
        help='Pretty print JSON output'
    )
    
    # Model options
    model_group = parser.add_argument_group('Model Options')
    model_group.add_argument(
        '--checkpoint-dir',
        type=str,
        default=None,
        help='Directory containing model checkpoints (overrides default paths)'
    )
    model_group.add_argument(
        '--model-to-use',
        nargs='+',
        choices=['all_features', 'wo_structures', 'wo_reactions', 'wo_rxn_structures'],
        default=None,
        help='Specific model(s) to load [default: all models]'
    )
    model_group.add_argument(
        '--max-length',
        type=int,
        default=1000,
        help='Maximum enzyme sequence length [default: 1000]'
    )
    
    # Device options
    device_group = parser.add_argument_group('Device Options')
    device_group.add_argument(
        '--device',
        type=str,
        default='cpu',
        help='Device to use: cpu or cuda:N (e.g., cuda:0) [default: cpu]'
    )
    device_group.add_argument(
        '--gpu-id',
        type=int,
        default=None,
        help='GPU ID to use (shortcut for --device cuda:N)'
    )
    
    return parser.parse_args()


def validate_inputs(args):
    """Validate input arguments."""
    if args.enzyme_structure is None and args.enzyme_sequence is None:
        print("Error: Either --enzyme-structure or --enzyme-sequence must be provided", file=sys.stderr)
        sys.exit(1)
    
    if args.enzyme_structure is not None:
        structure_path = Path(args.enzyme_structure)
        if not structure_path.exists():
            print(f"Error: Structure file not found: {args.enzyme_structure}", file=sys.stderr)
            sys.exit(1)
    
    if args.enzyme_sequence is not None and len(args.enzyme_sequence) == 0:
        print("Error: Enzyme sequence cannot be empty", file=sys.stderr)
        sys.exit(1)


def setup_config(args):
    """Setup EasIFA configuration from command line arguments."""
    config_kwargs = {
        'max_enzyme_aa_length': args.max_length,
        'pred_tolist': True,
    }
    
    # Setup device
    if args.gpu_id is not None:
        device = args.gpu_id
    elif args.device != 'cpu':
        if args.device.startswith('cuda:'):
            try:
                device = int(args.device.split(':')[1])
            except (IndexError, ValueError):
                device = 'cpu'
        else:
            device = 'cpu'
    else:
        device = 'cpu'
    
    # Override models to load
    if args.model_to_use is not None:
        config_kwargs['model_to_use'] = args.model_to_use
        # Set all devices for selected models
        gpu_allocations = {model: device for model in args.model_to_use}
        config_kwargs['gpu_allocations'] = gpu_allocations
    else:
        # Set device for all models
        gpu_allocations = {
            'all_features': device,
            'wo_structures': device,
            'wo_reactions': device,
            'wo_rxn_structures': device,
        }
        config_kwargs['gpu_allocations'] = gpu_allocations
    
    # Override checkpoint paths if provided
    if args.checkpoint_dir is not None:
        checkpoint_dir = Path(args.checkpoint_dir)
        model_state_paths = {
            'all_features': checkpoint_dir / 'all_features',
            'wo_structures': checkpoint_dir / 'wo_structures',
            'wo_reactions': checkpoint_dir / 'wo_reactions',
            'wo_rxn_structures': checkpoint_dir / 'wo_rxn_structures',
            'rxn_model_path': checkpoint_dir / 'rxn_model',
        }
        config_kwargs['model_state_paths'] = model_state_paths
    
    config = EasIFAInferenceConfig(**config_kwargs)
    return config


def run_inference(args, config):
    """Run inference with the given arguments and configuration."""
    print("Initializing EasIFA...")
    easifa = EasIFAInferenceAPI(config)
    
    print("Running inference...")
    results = easifa.inference(
        rxn_smiles=args.rxn_smiles,
        enzyme_structure_path=args.enzyme_structure,
        enzyme_aa_sequence=args.enzyme_sequence,
    )
    
    if results is None:
        print("Error: Inference failed. Sequence may be too long.", file=sys.stderr)
        sys.exit(1)
    
    pred, prob = results
    
    # Prepare output
    output_data = {
        'model_used': easifa.model_to_inference,
        'input': {
            'enzyme_structure': args.enzyme_structure,
            'enzyme_sequence': args.enzyme_sequence if args.enzyme_sequence else easifa.caculated_sequence,
            'rxn_smiles': args.rxn_smiles,
        },
        'predictions': {
            'labels': pred,
            'probabilities': prob,
        },
        'sequence_length': len(easifa.caculated_sequence),
        'site_type_mapping': {
            '0': 'non-site',
            '1': 'BINDING',
            '2': 'ACT_SITE',
            '3': 'SITE',
        }
    }
    
    return output_data


def save_results(output_data, output_path, pretty=False):
    """Save results to JSON file."""
    indent = 2 if pretty else None
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=indent, ensure_ascii=False)
    print(f"Results saved to: {output_path}")


def main():
    """Main entry point for CLI."""
    args = parse_args()
    
    # Validate inputs
    validate_inputs(args)
    
    # Setup configuration
    config = setup_config(args)
    
    try:
        # Run inference
        output_data = run_inference(args, config)
        
        # Save results
        save_results(output_data, args.output, args.pretty)
        
        # Print summary
        print("\n" + "="*60)
        print("Inference Summary:")
        print("="*60)
        print(f"Model used: {output_data['model_used']}")
        print(f"Sequence length: {output_data['sequence_length']}")
        
        pred_labels = output_data['predictions']['labels']
        if isinstance(pred_labels, list):
            # Count predicted sites by type
            from collections import Counter
            site_counts = Counter(pred_labels)
            print("\nPredicted sites:")
            site_mapping = output_data['site_type_mapping']
            for site_type, count in sorted(site_counts.items()):
                site_name = site_mapping.get(str(site_type), 'unknown')
                print(f"  {site_name}: {count} residues")
        print("="*60)
        
    except Exception as e:
        print(f"Error during inference: {str(e)}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
