"""This script generates Mezo-compatible graphs from its inference graph"""

import onnx
import numpy as np
import onnx_graphsurgeon as gs
from onnx import helper, numpy_helper
from pathlib import Path
import argparse

def generate_mezo_graphs(
    onnx_path: str,
    epsilon: float = 0.01,
    seed: float = 42.0,
    random_type: str = "normal",
) -> None:
    """
    Generates two ONNX models by inserting nodes to perturb weights for Conv, MatMul, and Gemm layers,
    following the MeZO (Zeroth-Order Optimization) methodology.

    This function inserts statically-seeded random operators. The unique seed for each
    operator serves as an identifier that a custom hardware runtime can override with
    a dynamic, runtime-provided seed.

    Args:
        onnx_path: Path to the original ONNX model.
        epsilon: The magnitude of the perturbation.
        seed: A base seed to generate unique, deterministic seeds for each operator.
        random_type: The type of random distribution to use ('normal' or 'uniform').
    """
    # Load original ONNX model
    p = Path(onnx_path)
    output_path_plus = p.parent / f"{p.stem}_plus.onnx"
    output_path_minus = p.parent / f"{p.stem}_minus.onnx"

    # --- 1. Identify target weights and biases ---
    model = onnx.load(onnx_path)
    weights_and_biases = {
        init.name
        for init in model.graph.initializer
        if "weight" in init.name or "bias" in init.name
    }

    if not weights_and_biases:
        print("Warning: No weights or biases containing 'weight' or 'bias' in their names were found to perturb.")
        return

    print(f"Found {len(weights_and_biases)} weight/bias tensors to perturb.")

    # --- 2. Helper function to modify the graph using onnx-surgeon ---
    def modify_graph(op_type, output_path):
        graph = gs.import_onnx(onnx.load(onnx_path))
        
        # Add epsilon as a constant tensor
        epsilon_tensor = gs.Constant(name="epsilon", values=np.array(epsilon, dtype=np.float32))
        
        nodes_to_modify = [node for node in graph.nodes if node.op in ["Conv", "MatMul", "Gemm"]]
        
        base_seed = int(seed)
        perturbation_counter = 0

        for node in nodes_to_modify:
            for i, weight_tensor in enumerate(node.inputs):
                if isinstance(weight_tensor, gs.Constant) and weight_tensor.name in weights_and_biases:
                    dtype = weight_tensor.dtype
                    
                    # Generate a unique but deterministic static seed for this operation.
                    # This acts as an ID for the hardware runtime to override.
                    unique_seed = float(base_seed + perturbation_counter)
                    
                    # Create a statically-seeded random noise tensor
                    random_output_tensor = gs.Variable(name=f"{weight_tensor.name}_random", dtype=dtype, shape=weight_tensor.shape)
                    
                    random_op_name = "RandomNormal" if random_type == "normal" else "RandomUniform"
                    random_node = gs.Node(
                        op=random_op_name,
                        name=f"{random_output_tensor.name}_op",
                        attrs={
                            "seed": unique_seed,
                            "dtype": helper.np_dtype_to_tensor_dtype(dtype),
                            "shape": weight_tensor.shape,
                        },
                        outputs=[random_output_tensor]
                    )
                    graph.nodes.append(random_node)

                    # Scale the noise by epsilon
                    scaled_noise_tensor = gs.Variable(name=f"{weight_tensor.name}_scaled_noise", dtype=dtype, shape=weight_tensor.shape)
                    casted_epsilon_tensor = gs.Variable(name=f"epsilon_casted_{perturbation_counter}", dtype=dtype)
                    cast_node = gs.Node(op="Cast", inputs=[epsilon_tensor], outputs=[casted_epsilon_tensor], attrs={"to": helper.np_dtype_to_tensor_dtype(dtype)})
                    graph.nodes.append(cast_node)

                    scale_node = gs.Node(op="Mul", name=f"{scaled_noise_tensor.name}_op", inputs=[random_output_tensor, casted_epsilon_tensor], outputs=[scaled_noise_tensor])
                    graph.nodes.append(scale_node)

                    # Create Add/Sub node
                    perturbed_weight_tensor = gs.Variable(name=f"{weight_tensor.name}_{op_type}", dtype=dtype, shape=weight_tensor.shape)
                    op_node = gs.Node(op=op_type, name=f"{perturbed_weight_tensor.name}_op", inputs=[weight_tensor, scaled_noise_tensor], outputs=[perturbed_weight_tensor])
                    graph.nodes.append(op_node)

                    # Update the original node's input
                    node.inputs[i] = perturbed_weight_tensor
                    perturbation_counter += 1

        # Clean up the graph
        graph.cleanup().toposort()
        
        # Export the modified graph
        onnx.save(gs.export_onnx(graph), output_path)

    # --- 3. Apply perturbations and save models ---
    modify_graph("Add", output_path_plus)
    modify_graph("Sub", output_path_minus)

    print(f"Saved perturbed models to:\n- {output_path_plus}\n- {output_path_minus}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Mezo-compatible perturbed ONNX models")
    parser.add_argument("--onnx_path", type=str, required=True, help="Path to the original ONNX model.")
    parser.add_argument("--epsilon", type=float, default=0.01, help="Magnitude of the perturbation.")
    parser.add_argument("--seed", type=float, default=42.0, help="Base random seed for reproducibility.")
    parser.add_argument("--random_type", type=str, default="normal", choices=["normal", "uniform"], help="Type of random distribution for noise.")
    args = parser.parse_args()
    generate_mezo_graphs(args.onnx_path, args.epsilon, args.seed, args.random_type)