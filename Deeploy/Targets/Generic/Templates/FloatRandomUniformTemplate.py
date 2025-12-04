# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List, Tuple

from Deeploy.DeeployTypes import NetworkContext, NodeTemplate, OperatorRepresentation


class _RandomUniformTemplate(NodeTemplate):

    def __init__(self, templateStr):
        super().__init__(templateStr)

    def alignToContext(self, ctxt: NetworkContext,
                       operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, Dict, List[str]]:
        # Add the node's unique ID to help create a unique seed.
        operatorRepresentation['node_id'] = operatorRepresentation['nodeIdx']
        return ctxt, operatorRepresentation, []


referenceTemplate = _RandomUniformTemplate("""
// RandomUniform (Name: ${nodeName}, Op: ${nodeOp})
// Implements an Xorshift PRNG.
BEGIN_SINGLE_CORE
    // Create a robustly unique seed using an LCG step on the runtime seed and node ID.
    // 'runtime_seed' must be provided by the calling context.
    uint32_t initial_val = runtime_seed + ${node_id};
    uint32_t seed = (initial_val * 1664525u) + 1013904223u;
    
    float range = ${high} - ${low};

    for (uint32_t i = 0; i < ${size}; i++) {
        // Xorshift PRNG
        seed ^= seed << 13;
        seed ^= seed >> 17;
        seed ^= seed << 5;
        
        // Normalize to [0, 1] and then scale to [low, high]
        float u = (float)seed / (float)0xFFFFFFFF;
        ${output}[i] = ${low} + u * range;
    }
END_SINGLE_CORE
""")