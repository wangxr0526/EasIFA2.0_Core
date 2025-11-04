#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Standalone inference script for EasIFA using command-line arguments.

This script can be used directly without installing the package.
"""

import argparse
import json
import sys
from pathlib import Path

# Add parent directory to path for imports
import os
current_dir = Path(__file__).resolve().parent
if str(current_dir.parent) not in sys.path:
    sys.path.insert(0, str(current_dir.parent))

from easifa_core import EasIFAInferenceAPI, EasIFAInferenceConfig


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='EasIFA: Enzyme Active Site Prediction',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Single prediction with structure and reaction:
  python easifa_predict.py --enzyme-structure protein.pdb --rxn-smiles "reactant>>product" --output result.json
  
  # Single prediction with sequence and reaction:
  python easifa_predict.py --enzyme-sequence "MSPRL..." --rxn-smiles "reactant>>product" --output result.json
  
  # Single prediction with structure only (no reaction):
  python easifa_predict.py --enzyme-structure protein.pdb --output result.json
  
  # Single prediction with sequence only (no reaction):
  python easifa_predict.py --enzyme-sequence "MSPRL..." --output result.json
  
  # Batch prediction from JSON file:
  python easifa_predict.py --batch-input batch_input.json --output batch_results.json
  
Batch Input JSON Format:
  [
    {
      "id": "protein_1",
      "enzyme_structure": "path/to/protein1.pdb",  # Optional
      "enzyme_sequence": "MSPRL...",                # Optional (provide structure or sequence)
      "rxn_smiles": "reactant>>product"             # Optional
    },
    {
      "id": "protein_2",
      "enzyme_sequence": "MVKLI...",
      "rxn_smiles": "CC>>CO"
    }
  ]
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
    input_group.add_argument(
        '--batch-input',
        type=str,
        default=None,
        help='Path to JSON file containing batch input data (see documentation for format)'
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
    output_group.add_argument(
        '--verbose',
        '-v',
        action='store_true',
        help='Print verbose output'
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
    # Check if batch mode or single mode
    if args.batch_input is not None:
        # Batch mode: validate batch input file
        batch_path = Path(args.batch_input)
        if not batch_path.exists():
            print(f"Error: Batch input file not found: {args.batch_input}", file=sys.stderr)
            return False
        # Check if other single-mode arguments are provided
        if args.enzyme_structure or args.enzyme_sequence or args.rxn_smiles:
            print("Warning: --batch-input is provided. Single prediction arguments will be ignored.", file=sys.stderr)
        return True
    
    # Single mode: validate single input
    if args.enzyme_structure is None and args.enzyme_sequence is None:
        print("Error: Either --enzyme-structure, --enzyme-sequence, or --batch-input must be provided", file=sys.stderr)
        return False
    
    if args.enzyme_structure is not None:
        structure_path = Path(args.enzyme_structure)
        if not structure_path.exists():
            print(f"Error: Structure file not found: {args.enzyme_structure}", file=sys.stderr)
            return False
    
    if args.enzyme_sequence is not None and len(args.enzyme_sequence) == 0:
        print("Error: Enzyme sequence cannot be empty", file=sys.stderr)
        return False
    
    return True


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


def load_batch_input(batch_file):
    """Load batch input from JSON file."""
    try:
        with open(batch_file, 'r', encoding='utf-8') as f:
            batch_data = json.load(f)
        
        if not isinstance(batch_data, list):
            print("Error: Batch input JSON must be a list of objects", file=sys.stderr)
            return None
        
        # Validate each entry
        for idx, entry in enumerate(batch_data):
            if not isinstance(entry, dict):
                print(f"Error: Entry {idx} is not a valid object", file=sys.stderr)
                return None
            
            if 'id' not in entry:
                print(f"Error: Entry {idx} missing required field 'id'", file=sys.stderr)
                return None
            
            if 'enzyme_structure' not in entry and 'enzyme_sequence' not in entry:
                print(f"Error: Entry {idx} (id: {entry['id']}) must have either 'enzyme_structure' or 'enzyme_sequence'", file=sys.stderr)
                return None
        
        return batch_data
    
    except json.JSONDecodeError as e:
        print(f"Error: Failed to parse JSON file: {e}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"Error: Failed to load batch input file: {e}", file=sys.stderr)
        return None


def run_single_inference(easifa, enzyme_structure=None, enzyme_sequence=None, rxn_smiles=None, verbose=False):
    """Run inference for a single protein."""
    results = easifa.inference(
        rxn_smiles=rxn_smiles,
        enzyme_structure_path=enzyme_structure,
        enzyme_aa_sequence=enzyme_sequence,
    )
    
    if results is None:
        return None
    
    pred, prob = results
    
    # Prepare output
    output_data = {
        'model_used': easifa.model_to_inference,
        'input': {
            'enzyme_structure': enzyme_structure,
            'enzyme_sequence': enzyme_sequence if enzyme_sequence else easifa.caculated_sequence,
            'rxn_smiles': rxn_smiles,
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


def run_inference(args, config):
    """Run inference with the given arguments and configuration."""
    if args.verbose:
        print("Initializing EasIFA...")
    
    easifa = EasIFAInferenceAPI(config)
    
    if args.verbose:
        print("Running inference...")
    
    output_data = run_single_inference(
        easifa,
        enzyme_structure=args.enzyme_structure,
        enzyme_sequence=args.enzyme_sequence,
        rxn_smiles=args.rxn_smiles,
        verbose=args.verbose
    )
    
    if output_data is None:
        print("Error: Inference failed. Sequence may be too long or invalid.", file=sys.stderr)
        return None
    
    return output_data


def run_batch_inference(args, config):
    """Run batch inference from JSON input file."""
    if args.verbose:
        print("Loading batch input...")
    
    batch_data = load_batch_input(args.batch_input)
    if batch_data is None:
        return None
    
    if args.verbose:
        print(f"Loaded {len(batch_data)} entries for batch prediction")
        print("Initializing EasIFA...")
    
    easifa = EasIFAInferenceAPI(config)
    
    results = []
    failed_ids = []
    
    if args.verbose:
        print("Running batch inference...")
    
    for idx, entry in enumerate(batch_data):
        protein_id = entry['id']
        enzyme_structure = entry.get('enzyme_structure')
        enzyme_sequence = entry.get('enzyme_sequence')
        rxn_smiles = entry.get('rxn_smiles')
        
        if args.verbose:
            print(f"  Processing {idx + 1}/{len(batch_data)}: {protein_id}")
        
        try:
            output_data = run_single_inference(
                easifa,
                enzyme_structure=enzyme_structure,
                enzyme_sequence=enzyme_sequence,
                rxn_smiles=rxn_smiles,
                verbose=False
            )
            
            if output_data is None:
                print(f"  Warning: Failed to process {protein_id} (sequence too long or invalid)", file=sys.stderr)
                failed_ids.append(protein_id)
                continue
            
            # Add protein ID to output
            output_data['id'] = protein_id
            results.append(output_data)
            
        except Exception as e:
            print(f"  Error processing {protein_id}: {str(e)}", file=sys.stderr)
            failed_ids.append(protein_id)
            continue
    
    if args.verbose:
        print(f"\nBatch inference completed:")
        print(f"  Successful: {len(results)}/{len(batch_data)}")
        if failed_ids:
            print(f"  Failed IDs: {', '.join(failed_ids)}")
    
    return {
        'batch_results': results,
        'total': len(batch_data),
        'successful': len(results),
        'failed': len(failed_ids),
        'failed_ids': failed_ids
    }


def save_results(output_data, output_path, pretty=False, verbose=False):
    """Save results to JSON file."""
    indent = 2 if pretty else None
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=indent, ensure_ascii=False)
    
    if verbose:
        print(f"Results saved to: {output_path}")


def print_summary(output_data):
    """Print inference summary."""
    print("\n" + "="*70)
    print("                     EasIFA Inference Summary")
    print("="*70)
    print(f"Model used:      {output_data['model_used']}")
    print(f"Sequence length: {output_data['sequence_length']} residues")
    
    pred_labels = output_data['predictions']['labels']
    if isinstance(pred_labels, list):
        # Count predicted sites by type
        from collections import Counter
        site_counts = Counter(pred_labels)
        print("\nPredicted active site residues:")
        site_mapping = output_data['site_type_mapping']
        
        for site_type_id in sorted(site_counts.keys()):
            if site_type_id == 0:  # Skip non-sites
                continue
            site_name = site_mapping.get(str(site_type_id), 'unknown')
            count = site_counts[site_type_id]
            percentage = (count / output_data['sequence_length']) * 100
            print(f"  {site_name:12s}: {count:4d} residues ({percentage:.1f}%)")
        
        total_sites = sum(1 for x in pred_labels if x != 0)
        total_percentage = (total_sites / output_data['sequence_length']) * 100
        print(f"  {'Total sites':12s}: {total_sites:4d} residues ({total_percentage:.1f}%)")
        print(f"  {'Non-sites':12s}: {site_counts.get(0, 0):4d} residues ({(site_counts.get(0, 0) / output_data['sequence_length']) * 100:.1f}%)")
    
    print("="*70)


def print_batch_summary(batch_results):
    """Print batch inference summary."""
    print("\n" + "="*70)
    print("                  EasIFA Batch Inference Summary")
    print("="*70)
    print(f"Total proteins:    {batch_results['total']}")
    print(f"Successful:        {batch_results['successful']}")
    print(f"Failed:            {batch_results['failed']}")
    
    if batch_results['failed_ids']:
        print(f"\nFailed protein IDs:")
        for failed_id in batch_results['failed_ids']:
            print(f"  - {failed_id}")
    
    print("\nPer-protein summary:")
    for idx, result in enumerate(batch_results['batch_results'][:5]):  # Show first 5
        print(f"\n  {idx + 1}. ID: {result['id']}")
        print(f"     Model: {result['model_used']}")
        print(f"     Sequence length: {result['sequence_length']} residues")
        pred_labels = result['predictions']['labels']
        total_sites = sum(1 for x in pred_labels if x != 0)
        print(f"     Active sites: {total_sites} residues")
    
    if len(batch_results['batch_results']) > 5:
        print(f"\n  ... and {len(batch_results['batch_results']) - 5} more proteins")
    
    print("="*70)


def main():
    """Main entry point for the script."""
    args = parse_args()
    
    # Validate inputs
    if not validate_inputs(args):
        sys.exit(1)
    
    # Setup configuration
    try:
        config = setup_config(args)
    except Exception as e:
        print(f"Error setting up configuration: {str(e)}", file=sys.stderr)
        sys.exit(1)
    
    try:
        # Check if batch mode or single mode
        if args.batch_input is not None:
            # Batch inference
            output_data = run_batch_inference(args, config)
            
            if output_data is None:
                sys.exit(1)
            
            # Save results
            save_results(output_data, args.output, args.pretty, args.verbose)
            
            # Print summary
            print_batch_summary(output_data)
        else:
            # Single inference
            output_data = run_inference(args, config)
            
            if output_data is None:
                sys.exit(1)
            
            # Save results
            save_results(output_data, args.output, args.pretty, args.verbose)
            
            # Print summary
            print_summary(output_data)
        
    except KeyboardInterrupt:
        print("\nInterrupted by user", file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        print(f"\nError during inference: {str(e)}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
