/*
 * SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "DeeployBasicMath.h"
#include <math.h>



// TODO: 1) loop unrolling for ILP perf
// TODO: 2) Perturbation directly integrated in GEMM or Conv kernels.
/* --------------------------- RNG ---------------------------------- */

uint32_t Xorshift32(uint32_t state) {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
}

/* --------------------------- Samplers ---------------------------------- */

float32_t TriangularSample(uint32_t *state) {
    *state = Xorshift32(*state);
    float32_t u1 = (float32_t)(*state) / (float32_t)0xFFFFFFFF; // in [0,1]
    // mutate state to avoid same seed for u2.
    *state = Xorshift32(*state);
    float32_t u2 = (float32_t)(*state) / (float32_t)0xFFFFFFFF; // in [0,1]
    return u1 - u2;
}

float32_t UniformSample(uint32_t *state) {
    *state = Xorshift32(*state);
    float32_t u1 = (float32_t)(*state) / (float32_t)0xFFFFFFFF; // in [0,1]
    return u1-0.5f; // centered around 0
}

float32_t GaussianSample(uint32_t *state) {
    // Box-Muller transform
    *state = Xorshift32(*state);
    float32_t u1 = (float32_t)(*state) / (float32_t)0xFFFFFFFF; // in (0,1]
    // mutate state to avoid same seed for u2.
    *state = Xorshift32(*state);
    float32_t u2 = (float32_t)(*state) / (float32_t)0xFFFFFFFF; // in [0,1]
    return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * PI_F * u2);
}

typedef struct {
    uint32_t state;
    uint32_t bits;
    int bitpos;
} RademacherRNG;

void RademacherRNG_init(RademacherRNG *rng, uint32_t seed) {
    rng->state = seed;
    rng->bits = 0;
    rng->bitpos = 32; // force refill on first use
}

float32_t RademacherSample(RademacherRNG *rng) {
    if (rng->bitpos >= 32) {
        rng->state = Xorshift32(rng->state);
        rng->bits = rng->state;
        rng->bitpos = 0;
    }
    float32_t val = (rng->bits & 1) ? 1.0f : -1.0f;
    rng->bits >>= 1;
    rng->bitpos++;
    return val;
}

/* ------------------------- Perturbation Functions -------------------------------- */

void ApplyTriangularPerturbation(const float32_t *__restrict__ pweights,
                            float32_t *__restrict__ pweights_dest,
                            uint32_t seed,
                            float32_t epsilon,
                            uint32_t dir,
                            uint32_t size) {
    uint32_t rng_state = (seed * 1664525u) + 1013904223u;
    float32_t sqrt6 = 2.44948974278f;
    float32_t scale = epsilon * sqrt6; // sqrt(6): => variance 1
    if (dir == 0) {scale *= -1.0f;}
    for (uint32_t i = 0; i < size; ++i) {
        float32_t tr = TriangularSample(&rng_state);
        pweights_dest[i] = pweights[i] + tr * scale;
    }
}

void ApplyUniformPerturbation(const float32_t *__restrict__ pweights,
                            float32_t *__restrict__ pweights_dest,
                            uint32_t seed,
                            float32_t epsilon,
                            uint32_t dir,
                            uint32_t size) {
    uint32_t rng_state = (seed * 1664525u) + 1013904223u;
    float32_t sqrt3 = 1.73205080757f;
    float32_t scale = epsilon * sqrt3 * 2.0f; // factor 2: [-0.5,0.5] => [-1,1], sqrt(3): => Gaussian(0, 1) l2 norm.
    if (dir == 0) {scale *= -1.0f;}
    for (uint32_t i = 0; i < size; ++i) {
        float32_t u = UniformSample(&rng_state);
        pweights_dest[i] = pweights[i] + u * scale;
    }
}


void ApplyGaussianPerturbation(const float32_t *__restrict__ pweights,
                            float32_t *__restrict__ pweights_dest,
                            uint32_t seed,
                            float32_t epsilon,
                            uint32_t dir,
                            uint32_t size) {
    uint32_t rng_state = (seed * 1664525u) + 1013904223u;
    float32_t sqrt3 = 1.73205080757f;
    float32_t scale = epsilon; // gaussian naturally has variance 1
    if (dir == 0) {scale *= -1.0f;}
    for (uint32_t i = 0; i < size; ++i) {
        float32_t u = GaussianSample(&rng_state);
        pweights_dest[i] = pweights[i] + u * scale;
    }
}

void ApplyRademacherPerturbation(const float32_t *__restrict__ pweights,
                            float32_t *__restrict__ pweights_dest,
                            uint32_t seed,
                            float32_t epsilon,
                            uint32_t dir,
                            uint32_t size) {
    uint32_t rng_state = (seed * 1664525u) + 1013904223u;
    float32_t sqrt3 = 1.73205080757f;
    float32_t scale = epsilon; // rademacher naturally has variance 1
    if (dir == 0) {scale *= -1.0f;}
    for (uint32_t i = 0; i < size; ++i) {
        float32_t u = RademacherSample(&rng_state);
        pweights_dest[i] = pweights[i] + u * scale;
    }
}

/* --------------------------- Update functions ---------------------------------- */

void UpdateWeightsTriangle(float32_t *__restrict__ pweights,
                            float32_t loss,
                            uint32_t seed,
                            float32_t epsilon,
                            float32_t lr,
                            uint32_t size) {
    uint32_t rng_state = (seed * 1664525u) + 1013904223u;
    float32_t sqrt6 = 2.44948974278f;
    const float32_t scale = sqrt6; // sqrt(6): => Gaussian(0, 1) l2 norm.
    for (uint32_t i = 0; i < size; ++i) {
        float32_t tr = TriangularSample(&rng_state);
        pweights[i] = pweights[i] - lr * loss/(2.0f * epsilon) * tr * scale;
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
    const float32_t scale = sqrt3 * 2.0f; // factor 2: [-0.5,0.5] => [-1,1], sqrt(3): => variance 1
    for (uint32_t i = 0; i < size; ++i) {
        float32_t u = UniformSample(&rng_state);
        pweights[i] = pweights[i] - lr * loss/(2.0f * epsilon) * u * scale;
    }
}

void UpdateWeightsGaussian(float32_t *__restrict__ pweights,
                            float32_t loss,
                            uint32_t seed,
                            float32_t epsilon,
                            float32_t lr,
                            uint32_t size) {
    uint32_t rng_state = (seed * 1664525u) + 1013904223u;
    float32_t sqrt3 = 1.73205080757f;
    const float32_t scale = sqrt3 * 2.0f; // factor 2: [-0.5,0.5] => [-1,1], sqrt(3): => variance 1
    for (uint32_t i = 0; i < size; ++i) {
        float32_t u = GaussianSample(&rng_state);
        pweights[i] = pweights[i] - lr * loss/(2.0f * epsilon) * u * scale;
    }
}

void UpdateWeightsRademacher(float32_t *__restrict__ pweights,
                            float32_t loss,
                            uint32_t seed,
                            float32_t epsilon,
                            float32_t lr,
                            uint32_t size) {
    uint32_t rng_state = (seed * 1664525u) + 1013904223u;
    float32_t sqrt3 = 1.73205080757f;
    const float32_t scale = sqrt3 * 2.0f; // factor 2: [-0.5,0.5] => [-1,1], sqrt(3): => variance 1
    for (uint32_t i = 0; i < size; ++i) {
        float32_t u = RademacherSample(&rng_state);
        pweights[i] = pweights[i] - lr * loss/(2.0f * epsilon) * u * scale;
    }
}