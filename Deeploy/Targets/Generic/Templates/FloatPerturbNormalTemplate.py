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
    ApplyTriangularPerturbation((const float32_t *)${data_in},
                                (float32_t *)${data_out},
                                seed + ${node_id},
                                ${eps}f,
                                perturbation_sign, // globally defined in DeedeployTest main
                                ${size});
END_SINGLE_CORE
""")

updateTemplate = _FloatPerturbNormalTemplate("""
// UpdateNormal (Name: ${nodeName}, Op: ${nodeOp})
BEGIN_SINGLE_CORE
    UpdateWeightsTriangle((float32_t *)${data_in},
                                loss,
                                seed + ${node_id},
                                ${eps}f,
                                lr, // globally defined
                                ${size});
END_SINGLE_CORE
""")