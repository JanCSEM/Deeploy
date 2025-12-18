# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List, Tuple

from Deeploy.DeeployTypes import NetworkContext, NodeTemplate, OperatorRepresentation


class _FloatPerturbNormalTemplate(NodeTemplate):

    def __init__(self, templateStr):
        super().__init__(templateStr)

    def alignToContext(self, ctxt: NetworkContext,
                       operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, Dict, List[str]]:
        # Add the node's unique ID to help create a unique seed_${nodeName}.

        data_in = ctxt.lookup(operatorRepresentation['data_in'])
        data_out = ctxt.lookup(operatorRepresentation['data_out'])
        operatorRepresentation['node_id'] = operatorRepresentation['nodeIdx']
        return ctxt, operatorRepresentation, []


referenceTemplate = _FloatPerturbNormalTemplate("""
// PerturbNormal (Name: ${nodeName}, Op: ${nodeOp})
BEGIN_SINGLE_CORE
    uint32_t initial_val_${nodeName} = ${seed} + ${node_id};
    uint32_t seed_${nodeName} = (initial_val_${nodeName} * 1664525u) + 1013904223u;

    const float scale_factor_${nodeName} = ${eps} * 2.44948974278f;

    for (uint32_t i = 0; i < ${size}; i++) {
        // Generate u1 from [0, 1]
        seed_${nodeName} ^= seed_${nodeName} << 13;
        seed_${nodeName} ^= seed_${nodeName} >> 17;
        seed_${nodeName} ^= seed_${nodeName} << 5;
        float u1_${nodeName} = (float)seed_${nodeName} / (float)0xFFFFFFFF;

        // Generate u2 from [0, 1]
        seed_${nodeName} ^= seed_${nodeName} << 13;
        seed_${nodeName} ^= seed_${nodeName} >> 17;
        seed_${nodeName} ^= seed_${nodeName} << 5;
        float u2_${nodeName} = (float)seed_${nodeName} / (float)0xFFFFFFFF;

        float triangular_sample_${nodeName} = u1_${nodeName} - u2_${nodeName};
        
        ${data_out}[i] = triangular_sample_${nodeName} * scale_factor_${nodeName} + ${data_in}[i];
    }
END_SINGLE_CORE
""")