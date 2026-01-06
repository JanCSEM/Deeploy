/*
 * SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "DeeployBasicMath.h"

// Sample from Unifom distribution U[-0.5,0.5]
float32_t UniformSample(uint32_t *state);
// Sample from triangular distribution Tr[-1, 1]
float32_t TriangularSample(uint32_t *state);

// Applies triangular perturbation to the weights and applies rescaling to match Gaussian(0, 1) l2 norm.
void ApplyTriangularPerturbation(const float32_t *__restrict__ psrc,
                            float32_t *__restrict__ pdst,
                            uint32_t seed,
                            float32_t epsilon,
                            int32_t dir,
                            uint32_t size);

// Updates the weights in place according to the MeZO update rule with triangular noise.
// Only supports qMeZO with q = 1 for now.
void UpdateWeightsTriangle(float32_t *__restrict__ pweights,
                            float32_t loss,
                            uint32_t seed,
                            float32_t epsilon,
                            float32_t lr,
                            uint32_t size);

// Applies uniform perturbation to the weights and applies rescaling to match Gaussian(0, 1) l2 norm.
void ApplyUniformPerturbation(float32_t *__restrict__ pnoise,
                            uint32_t seed,
                            int32_t dir,
                            float32_t epsilon,
                            uint32_t size);

// Updates the weights in place according to the MeZO update rule with uniform noise.
// Only supports qMeZO with q = 1 for now.
void UpdateWeightsUniform(float32_t *__restrict__ pweights,
                            float32_t loss,
                            uint32_t seed,
                            float32_t epsilon,
                            float32_t lr,
                            uint32_t size);

/* Xorshift32 implementation. Most basic software PRNG*/
uint32_t Xorshift32(uint32_t state);