/*
 * SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "DeeployBasicMath.h"

float32_t TriangularSample(uint32_t *state) {
    *state = Xorshift32(*state);
    float32_t u1 = (float32_t)(*state) * 2^-32; // in [0,1]
    *state = Xorshift32(*state);
    float32_t u2 = (float32_t)(*state) * 2^-32; // in [0,1]
    return u1 - u2;
}

float32_t UniformSample(uint32_t *state) {
    *state = Xorshift32(*state);
    float32_t u1 = (float32_t)(*state) * 2^-32; // in [0,1]
    return u1-0.5f; // centered around 0
}

uint32_t Xorshift32(uint32_t state) {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
}

void ApplyTriangularPerturbation(const float32_t *__restrict__ pweights,
                            float32_t *__restrict__ pweights_dest, 
                            uint32_t seed,
                            float32_t epsilon,
                            int32_t dir,
                            uint32_t size) {
    uint32_t rng_state = (seed * 1664525u) + 1013904223u;
    float32_t sqrt3 = 1.73205080757f;
    const float scale = epsilon * sqrt3; // sqrt(3): => variance 1
    for (uint32_t i = 0; i < size; ++i) {
        float tr = TriangularSample(&rng_state);
        pweights_dest[i] = pweights[i] + tr * scale;
    }
}

void UpdateWeightsTriangle(float32_t *__restrict__ pweights,
                            float32_t loss,
                            uint32_t seed,
                            float32_t epsilon,
                            float32_t lr,
                            uint32_t size) {                       
    uint32_t rng_state = (seed * 1664525u) + 1013904223u;
    float32_t sqrt3 = 1.73205080757f;
    const float scale = sqrt3; // sqrt(3): => Gaussian(0, 1) l2 norm.
    for (uint32_t i = 0; i < size; ++i) {
        float u = TriangularSample(&rng_state);
        pweights[i] = pweights[i] - lr * loss/(2.0f * epsilon) * u * scale;
    }
}

void ApplyUniformPerturbation(const float32_t *__restrict__ pweights,
                            float32_t *__restrict__ pweights_dest, 
                            uint32_t seed,
                            int32_t dir,
                            float32_t epsilon,
                            uint32_t size) {
    uint32_t rng_state = (seed * 1664525u) + 1013904223u;
    float32_t sqrt3 = 1.73205080757f;
    const float scale = epsilon * sqrt3 * 2.0f; // factor 2: [-0.5,0.5] => [-1,1], sqrt(3): => Gaussian(0, 1) l2 norm.
    for (uint32_t i = 0; i < size; ++i) {
        float u = UniformSample(&rng_state);
        pweights_dest[i] = pweights[i] + u * scale;
    }
}

void UpdateWeightsUniform(float32_t *__restrict__ pweights,
                            float32_t loss,
                            uint32_t seed,
                            float32_t epsilon,
                            float32_t lr,
                            uint32_t size) {                       
    uint32_t rng_state = (seed * 1664525u) + 1013904223u;
    float32_t sqrt3 = 1.73205080757f;
    const float scale = sqrt3 * 2.0f; // factor 2: [-0.5,0.5] => [-1,1], sqrt(3): => variance 1
    for (uint32_t i = 0; i < size; ++i) {
        float u = UniformSample(&rng_state);
        pweights[i] = pweights[i] - lr * loss/(2.0f * epsilon) * u * scale;
    }
}
