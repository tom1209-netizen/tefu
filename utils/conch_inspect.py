import torch
import os
import argparse
import sys


def inspect_checkpoint(checkpoint_path, log_file="model_structure_log.txt"):
    print(f"\nLoading checkpoint: {checkpoint_path}")
    print(f"Writing detailed log to: {log_file}")
    print("=" * 60)

    if not os.path.exists(checkpoint_path):
        print(f"Error: File not found at {checkpoint_path}")
        return

    try:
        # Load the checkpoint (cpu is safer/faster for inspection)
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        # Handle cases where the state_dict is nested
        if isinstance(checkpoint, dict):
            if 'state_dict' in checkpoint:
                print("Found 'state_dict' key, using inner dictionary.")
                state_dict = checkpoint['state_dict']
            elif 'model' in checkpoint:
                print("Found 'model' key, using inner dictionary.")
                state_dict = checkpoint['model']
            else:
                # Heuristic: check if values are tensors
                first_val = next(iter(checkpoint.values()))
                if isinstance(first_val, torch.Tensor):
                    print("Dictionary appears to be the state_dict itself.")
                    state_dict = checkpoint
                else:
                    print(
                        "Could not automatically determine state_dict structure. Logging top-level keys.")
                    state_dict = checkpoint
        else:
            print("Checkpoint is not a dictionary?")
            return

        # Open log file
        with open(log_file, "w") as f:
            f.write(f"Checkpoint: {checkpoint_path}\n")
            f.write(f"Total Parameters: {len(state_dict)}\n")
            f.write("=" * 80 + "\n")
            f.write(f"{'Layer Name':<70} | {'Shape'}\n")
            f.write("-" * 80 + "\n")

            # Write every single layer
            for key, value in state_dict.items():
                if hasattr(value, 'shape'):
                    shape_str = str(list(value.shape))
                else:
                    shape_str = "No Shape (Scalar/Other)"

                f.write(f"{key:<70} | {shape_str}\n")

        print(f"\n[SUCCESS] Model structure dumped to {log_file}")
        print("Please open that file and search for 'text' or 'proj' to find your missing layer.")

    except Exception as e:
        print(f"Failed to load checkpoint: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Default to the path you used in your config
    parser.add_argument(
        "--path", type=str, default="./pytorch_model.bin", help="Path to .bin file")
    parser.add_argument(
        "--out", type=str, default="conch_structure.txt", help="Output log file")
    args = parser.parse_args()

    inspect_checkpoint(args.path, args.out)
