# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List, Tuple

from Deeploy.DeeployTypes import NetworkContext, NodeTemplate, OperatorRepresentation


class _RandomNormalTemplate(NodeTemplate):

    def __init__(self, templateStr):
        super().__init__(templateStr)

    def alignToContext(self, ctxt: NetworkContext,
                       operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, Dict, List[str]]:
        # Add the node's unique ID to help create a unique seed.
        operatorRepresentation['node_id'] = operatorRepresentation['nodeIdx']
        return ctxt, operatorRepresentation, []


referenceTemplate = _RandomNormalTemplate("""
// RandomNormal (Name: ${nodeName}, Op: ${nodeOp})
// Approximates a normal distribution using a fast triangular distribution.
BEGIN_SINGLE_CORE
    // Create a robustly unique seed using an LCG step on the runtime seed and node ID.
    // 'runtime_seed' must be provided by the calling context.
    uint32_t initial_val = runtime_seed + ${node_id};
    uint32_t seed = (initial_val * 1664525u) + 1013904223u;

    // The standard deviation of (u1-u2) is 1/sqrt(6). We need to scale it.
    // scale_factor = target_std / (1/sqrt(6)) = target_std * sqrt(6)
    const float scale_factor = ${scale} * 2.44948974278f;

    for (uint32_t i = 0; i < ${size}; i++) {
        // Generate u1 from [0, 1]
        seed ^= seed << 13;
        seed ^= seed >> 17;
        seed ^= seed << 5;
        float u1 = (float)seed / (float)0xFFFFFFFF;

        // Generate u2 from [0, 1]
        seed ^= seed << 13;
        seed ^= seed >> 17;
        seed ^= seed << 5;
        float u2 = (float)seed / (float)0xFFFFFFFF;

        // Triangular sample is in [-1, 1]. Scale it to match target std dev.
        float triangular_sample = u1 - u2;
        
        // Apply scale and mean
        ${output}[i] = triangular_sample * scale_factor + ${mean};
    }
END_SINGLE_CORE
""")