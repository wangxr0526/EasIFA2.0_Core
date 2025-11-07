#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Command-line interface for EasIFA inference.
"""

import argparse
import json
import sys
import os
from pathlib import Path

try:
    from easifa_core import EasIFAInferenceAPI, EasIFAInferenceConfig
except ImportError:
    # If running from source without installation
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from easifa_core import EasIFAInferenceAPI, EasIFAInferenceConfig


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='EasIFA: Enzyme Active Site Prediction',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Batch prediction from JSON file:
  %(prog)s --batch-input batch_tasks.json --output batch_results.json
  
  # Predict with structure and reaction:
  %(prog)s --enzyme-structure protein.pdb --rxn-smiles "reactant>>product" --output result.json
  
  # Predict with sequence and reaction:
  %(prog)s --enzyme-sequence "MSPRL..." --rxn-smiles "reactant>>product" --output result.json
  
  # Predict with structure only (no reaction):
  %(prog)s --enzyme-structure protein.pdb --output result.json
  
  # Predict with sequence only (no reaction):
  %(prog)s --enzyme-sequence "MSPRL..." --output result.json

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

Note: All file paths in batch input JSON are resolved relative to the current working directory.
        '''
    )
    
    # Input options
    input_group = parser.add_argument_group('Input Options')
    input_group.add_argument(
        '--batch-input',
        type=str,
        default=None,
        help='Path to batch input JSON file (see BATCH_PREDICTION_GUIDE.md for format)'
    )
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
    # Batch mode
    if args.batch_input is not None:
        batch_path = Path(args.batch_input)
        if not batch_path.exists():
            print(f"Error: Batch input file not found: {args.batch_input}", file=sys.stderr)
            sys.exit(1)
        # In batch mode, other input args should not be provided
        if args.enzyme_structure is not None or args.enzyme_sequence is not None or args.rxn_smiles is not None:
            print("Error: When using --batch-input, do not provide --enzyme-structure, --enzyme-sequence, or --rxn-smiles", file=sys.stderr)
            sys.exit(1)
        return
    
    # Single prediction mode
    if args.enzyme_structure is None and args.enzyme_sequence is None:
        print("Error: Either --batch-input, --enzyme-structure, or --enzyme-sequence must be provided", file=sys.stderr)
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
    if args.verbose:
        print("Initializing EasIFA...")
    
    easifa = EasIFAInferenceAPI(config)
    
    if args.verbose:
        print("Running inference...")
    
    results = easifa.inference(
        rxn_smiles=args.rxn_smiles,
        enzyme_structure_path=args.enzyme_structure,
        enzyme_aa_sequence=args.enzyme_sequence,
    )
    
    if results is None:
        print("Error: Inference failed. Sequence may be too long or invalid.", file=sys.stderr)
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


def save_results(output_data, output_path, pretty=False, verbose=False):
    """Save results to JSON file."""
    indent = 2 if pretty else None
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=indent, ensure_ascii=False)
    
    if verbose:
        print(f"Results saved to: {output_path}")


def load_batch_input(batch_file, verbose=False):
    """
    Load batch input from JSON file.
    Note: All file paths in the JSON are resolved relative to the current working directory.
    """
    try:
        with open(batch_file, 'r', encoding='utf-8') as f:
            batch_data = json.load(f)
        
        if not isinstance(batch_data, list):
            print("Error: Batch input JSON must be a list of prediction tasks", file=sys.stderr)
            return None
        
        # Validate each entry
        for idx, task in enumerate(batch_data):
            if not isinstance(task, dict):
                print(f"Error: Task {idx} is not a valid object", file=sys.stderr)
                return None
            
            if 'id' not in task:
                print(f"Error: Task {idx} missing required field 'id'", file=sys.stderr)
                return None
            
            if 'enzyme_structure' not in task and 'enzyme_sequence' not in task:
                print(f"Error: Task {idx} (id: {task['id']}) must have either 'enzyme_structure' or 'enzyme_sequence'", file=sys.stderr)
                return None
        
        if verbose:
            print(f"Note: All file paths in batch input are resolved relative to: {os.getcwd()}")
        
        return batch_data
    
    except json.JSONDecodeError as e:
        print(f"Error: Failed to parse JSON file: {e}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"Error: Failed to load batch input file: {e}", file=sys.stderr)
        return None


def run_batch_inference(args, config):
    """Run batch inference from JSON input file."""
    if args.verbose:
        print("Loading batch input...")
    
    batch_data = load_batch_input(args.batch_input, verbose=args.verbose)
    if batch_data is None:
        sys.exit(1)
    
    print(f"Loaded {len(batch_data)} prediction tasks from {args.batch_input}")
    
    if args.verbose:
        print("Initializing EasIFA...")
    
    easifa = EasIFAInferenceAPI(config)
    
    results = []
    failed_tasks = []
    
    if args.verbose:
        print("Running batch inference...")
    
    # Store current working directory for path resolution
    cwd = os.getcwd()
    
    for idx, task in enumerate(batch_data, 1):
        task_id = task.get('id', f'task_{idx}')
        print(f"\n[{idx}/{len(batch_data)}] Processing: {task_id}")
        
        try:
            # Extract inputs (use same keys as in JSON)
            rxn_smiles = task.get('rxn_smiles')
            enzyme_structure = task.get('enzyme_structure')
            enzyme_sequence = task.get('enzyme_sequence')
            
            # Resolve relative paths relative to current working directory
            if enzyme_structure and not os.path.isabs(enzyme_structure):
                enzyme_structure = os.path.join(cwd, enzyme_structure)
                if args.verbose:
                    print(f"  Resolved structure path: {enzyme_structure}")
            
            # Run inference
            result = easifa.inference(
                rxn_smiles=rxn_smiles,
                enzyme_structure_path=enzyme_structure,
                enzyme_aa_sequence=enzyme_sequence,
            )
            
            if result is None:
                print(f"  ✗ Failed: Sequence may be too long or invalid input")
                failed_tasks.append({
                    'id': task_id,
                    'error': 'Inference returned None (sequence too long or invalid input)'
                })
                continue
            
            pred, prob = result
            
            # Store result (preserve original keys from input)
            task_result = {
                'id': task_id,
                'model_used': easifa.model_to_inference,
                'input': {
                    'enzyme_structure': task.get('enzyme_structure'),  # Store original path from JSON
                    'enzyme_sequence': enzyme_sequence if enzyme_sequence else easifa.caculated_sequence,
                    'rxn_smiles': rxn_smiles,
                },
                'predictions': {
                    'labels': pred,
                    'probabilities': prob,
                },
                'sequence_length': len(easifa.caculated_sequence),
            }
            
            results.append(task_result)
            print(f"  ✓ Success: {len(easifa.caculated_sequence)} residues, model: {easifa.model_to_inference}")
            
        except Exception as e:
            print(f"  ✗ Error: {str(e)}")
            failed_tasks.append({
                'id': task_id,
                'error': str(e)
            })
    
    # Prepare output
    output_data = {
        'total_tasks': len(batch_data),
        'successful': len(results),
        'failed': len(failed_tasks),
        'results': results,
        'failed_tasks': failed_tasks,
        'site_type_mapping': {
            '0': 'non-site',
            '1': 'BINDING',
            '2': 'ACT_SITE',
            '3': 'SITE',
        }
    }
    
    return output_data


def main():
    """Main entry point for CLI."""
    args = parse_args()
    
    # Validate inputs
    validate_inputs(args)
    
    # Setup configuration
    config = setup_config(args)
    
    try:
        # Check if batch mode
        if args.batch_input is not None:
            # Run batch inference
            output_data = run_batch_inference(args, config)
            
            # Save results
            save_results(output_data, args.output, args.pretty, args.verbose)
            
            # Print summary
            print("\n" + "="*70)
            print("Batch Inference Summary:")
            print("="*70)
            print(f"Total tasks: {output_data['total_tasks']}")
            print(f"Successful: {output_data['successful']}")
            print(f"Failed: {output_data['failed']}")
            if output_data['failed'] > 0:
                print("\nFailed tasks:")
                for failed in output_data['failed_tasks']:
                    print(f"  - {failed['id']}: {failed['error']}")
            print("="*70)
        else:
            # Run single inference
            output_data = run_inference(args, config)
            
            # Save results
            save_results(output_data, args.output, args.pretty, args.verbose)
            
            # Print summary
            print("\n" + "="*70)
            print("Inference Summary:")
            print("="*70)
            print(f"Model used: {output_data['model_used']}")
            print(f"Sequence length: {output_data['sequence_length']} residues")
            
            pred_labels = output_data['predictions']['labels']
            if isinstance(pred_labels, list):
                # Count predicted sites by type
                from collections import Counter
                site_counts = Counter(pred_labels)
                print("\nPredicted active site residues:")
                site_mapping = output_data['site_type_mapping']
                
                for site_type_id in sorted(site_counts.keys()):
                    if site_type_id == 0:  # Skip non-sites for detail display
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
        
    except KeyboardInterrupt:
        print("\nInterrupted by user", file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        print(f"Error during inference: {str(e)}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
