"""
Convert PyTorch .pth model files to safetensors format.
This script will convert all model.pth files in the checkpoints directory to model.safetensors
"""
import os
import torch
from pathlib import Path
from safetensors.torch import save_file

def convert_pth_to_safetensors(pth_path, safetensors_path=None):
    """
    Convert a .pth file to .safetensors format.
    
    Args:
        pth_path: Path to the .pth file
        safetensors_path: Output path for .safetensors file (optional)
    """
    if safetensors_path is None:
        safetensors_path = pth_path.replace('.pth', '.safetensors')
    
    print(f"Loading {pth_path}...")
    try:
        # Load the PyTorch state dict
        state_dict = torch.load(pth_path, map_location='cpu')
        
        # Remove 'module.' prefix if present (from DataParallel)
        cleaned_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('module.'):
                cleaned_state_dict[key.replace('module.', '')] = value
            else:
                cleaned_state_dict[key] = value
        
        # Save as safetensors
        print(f"Saving to {safetensors_path}...")
        save_file(cleaned_state_dict, safetensors_path)
        print(f"✓ Successfully converted: {safetensors_path}")
        return True
    except Exception as e:
        print(f"✗ Error converting {pth_path}: {e}")
        return False

def find_and_convert_all_checkpoints(checkpoint_root):
    """
    Find all model.pth files in checkpoint directory and convert them.
    
    Args:
        checkpoint_root: Root directory containing checkpoints
    """
    checkpoint_root = Path(checkpoint_root)
    pth_files = list(checkpoint_root.rglob('model.pth'))
    
    print(f"Found {len(pth_files)} model.pth files to convert\n")
    
    success_count = 0
    for pth_file in pth_files:
        safetensors_file = pth_file.parent / 'model.safetensors'
        print(f"\n{'='*80}")
        print(f"Converting: {pth_file.relative_to(checkpoint_root)}")
        
        if convert_pth_to_safetensors(str(pth_file), str(safetensors_file)):
            success_count += 1
            # Get file sizes for comparison
            pth_size = pth_file.stat().st_size / (1024**2)  # MB
            safetensors_size = safetensors_file.stat().st_size / (1024**2)  # MB
            print(f"  Original size: {pth_size:.2f} MB")
            print(f"  New size: {safetensors_size:.2f} MB")
    
    print(f"\n{'='*80}")
    print(f"Conversion complete: {success_count}/{len(pth_files)} files successfully converted")
    
    return success_count == len(pth_files)

if __name__ == '__main__':
    import sys
    
    # Get checkpoint directory
    if len(sys.argv) > 1:
        checkpoint_dir = sys.argv[1]
    else:
        # Default to checkpoints directory in project root
        script_dir = Path(__file__).parent
        checkpoint_dir = script_dir / 'checkpoints'
    
    if not Path(checkpoint_dir).exists():
        print(f"Error: Checkpoint directory not found: {checkpoint_dir}")
        sys.exit(1)
    
    print(f"Converting all model.pth files in: {checkpoint_dir}\n")
    
    success = find_and_convert_all_checkpoints(checkpoint_dir)
    
    if success:
        print("\n✓ All conversions successful!")
        sys.exit(0)
    else:
        print("\n✗ Some conversions failed. Please check the errors above.")
        sys.exit(1)
