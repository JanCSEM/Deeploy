/*
* SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
*
* SPDX-License-Identifier: Apache-2.0
*/

#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <float.h>

#include "Network.h"
#include "testinputs.h"
#include "testoutputs.h"

#if defined(MEZO_TRAINING) && !defined(INFERENCE)

#ifndef MEZO_NUM_EPOCHS
#define MEZO_NUM_EPOCHS 5u
#endif
#ifndef MEZO_NUM_TRAINING_SAMPLES
#define MEZO_NUM_TRAINING_SAMPLES 60000u
#endif
#ifndef MEZO_EPSILON
#define MEZO_EPSILON 1.0e-2f
#endif
#ifndef MEZO_LEARNING_RATE
#define MEZO_LEARNING_RATE 0.00004f
#endif
#ifndef MEZO_UPDATE_INTERVAL
#define MEZO_UPDATE_INTERVAL 64u
#endif
#if MEZO_UPDATE_INTERVAL == 0
#error "MEZO_UPDATE_INTERVAL must be greater than zero"
#endif



// A static buffer to hold one input sample.
// Ensure this size is large enough for your model's input.
// TODO: Dynamically allocate based on model input size.
static float32_t sample_buffer[784]; // 1*28*28 for MNIST

/**
 * @brief Reads one full input sample from stdin into a temporary buffer.
 * @return 1 on success, 0 on failure (e.g., EOF).
 */
static int StreamNextSample(void) {
  const size_t bytes_to_read = DeeployNetwork_inputs_bytes[0];
  if (bytes_to_read > sizeof(sample_buffer)) {
      printf("Error: sample_buffer is too small for the network input!\n");
      return 0;
  }
  const size_t bytes_read = fread(sample_buffer, 1, bytes_to_read, stdin);
  if (bytes_read == 0 && feof(stdin)) return 0; // End of stream
  if (bytes_read != bytes_to_read) {
    printf("Error: Failed to read full sample from stdin. Read %zu/%zu bytes.\n", bytes_read, bytes_to_read);
    return 0; // I/O error
  }
  return 1; // Success
}

/**
 * @brief Copies the sample from the temporary buffer to the network's input.
 */
static void LoadSampleFromBuffer(void) {
    memcpy(DeeployNetwork_inputs[0], sample_buffer, DeeployNetwork_inputs_bytes[0]);
}

static float32_t ComputeCrossEntropyLoss(const float32_t *logits, uint32_t num_classes, uint32_t target_index) {
  
  if (num_classes == 0u) {
    return 0.0f;
  }
  float32_t max_logit = logits[0];
  for (uint32_t i = 1; i < num_classes; ++i) {
    if (logits[i] > max_logit) {
      max_logit = logits[i];
    }
  }
  float32_t sum = 0.0f;
  for (uint32_t i = 0; i < num_classes; ++i) {
    sum += expf(logits[i] - max_logit);
  }
  float32_t log_prob = logits[target_index] - max_logit - logf(sum);
  // printf("Max logit vs target: %.6f vs %d\n" , (float32_t)max_logit, (int)target_index);
  return -log_prob;
}

int main(void) {
  printf("Initializing network...\r\n");
  InitNetwork(0, 1);

  printf("Starting MEZO training (%u epochs, %u samples/epoch, update interval %u)\r\n",
         (unsigned int)MEZO_NUM_EPOCHS,
         (unsigned int)MEZO_NUM_TRAINING_SAMPLES,
         (unsigned int)MEZO_UPDATE_INTERVAL);

  for (uint32_t epoch = 0; epoch < MEZO_NUM_EPOCHS; ++epoch) {
    printf("Epoch %u/%u\r\n", (unsigned int)(epoch + 1), (unsigned int)MEZO_NUM_EPOCHS);

    uint32_t samples_in_window = 0;
    float32_t loss_plus_accum = 0.0f;
    float32_t loss_minus_accum = 0.0f;
    float32_t total_epoch_loss_plus = 0.0f;
    float32_t total_epoch_loss_minus = 0.0f;
    uint32_t noise_seed = (epoch + 1u) * 1315423911u;

    for (uint32_t sample = 0; sample < MEZO_NUM_TRAINING_SAMPLES; ++sample) {
      if (!StreamNextSample()) {
        printf("Stopping training: No more data in stream.\n");
        goto end_of_epoch;
      }
      uint32_t current_target;
      if (fread(&current_target, sizeof(uint32_t), 1, stdin) != 1) {
          printf("Stopping training: Failed to read target label.\n");
          goto end_of_epoch;
      }

      // Positive perturbation
      LoadSampleFromBuffer();
      RunNetworkPerturbed(0, 1, noise_seed, 1); // forward pass with positive perturbation
      float32_t lplus = ComputeCrossEntropyLoss((float32_t*)DeeployNetwork_output_0, 10, current_target);
      LoadSampleFromBuffer();
      RunNetworkPerturbed(0, 1, noise_seed, 0); // forward pass with negative perturbation
      float32_t lmin = ComputeCrossEntropyLoss((float32_t*)DeeployNetwork_output_0, 10, current_target);

      loss_plus_accum += lplus;
      loss_minus_accum += lmin;
      total_epoch_loss_plus += lplus;
      total_epoch_loss_minus += lmin;
      samples_in_window++;

      if (samples_in_window == MEZO_UPDATE_INTERVAL) {
        const float32_t avg_plus = loss_plus_accum / (float32_t)samples_in_window;
        const float32_t avg_minus = loss_minus_accum / (float32_t)samples_in_window;
        const float32_t loss = avg_plus - avg_minus;
        // printf("  Sample %u: loss+ = %.6f | loss_ = %.6f\r\n",
        //      (unsigned int)(sample + 1), (double)avg_plus, (double)avg_minus);
        UpdateWeightsFiniteDiff(0, 1, MEZO_LEARNING_RATE, noise_seed, loss);

        // printf("    -> Weights updated (avg loss+ %.6f, avg loss- %.6f)\r\n",
        //        avg_plus, avg_minus);
        samples_in_window = 0;
        loss_plus_accum = 0.0f;
        loss_minus_accum = 0.0f;
      }
    }
    printf("Epoch %u completed. Avg Loss+ %.6f | Avg Loss- %.6f\r\n",
           (unsigned int)(epoch + 1),
           (double)(total_epoch_loss_plus / (float32_t)MEZO_NUM_TRAINING_SAMPLES),
           (double)(total_epoch_loss_minus / (float32_t)MEZO_NUM_TRAINING_SAMPLES));
    total_epoch_loss_minus = 0.0f;
    total_epoch_loss_plus = 0.0f;
    end_of_epoch:;
  }

  return 0;
}

#else  // MEZO_TRAINING not defined or INFERENCE defined

int main() {

  printf("Initializing network...\r\n");

  InitNetwork(0, 1);

  for (uint32_t buf = 0; buf < DeeployNetwork_num_inputs; buf++) {
    memcpy(DeeployNetwork_inputs[buf], testInputVector[buf],
           DeeployNetwork_inputs_bytes[buf]);
  }

  printf("Running network...\r\n");
  RunNetwork(0, 1);

  int32_t tot_err = 0;
  uint32_t tot = 0;
  OUTPUTTYPE diff;
  OUTPUTTYPE expected, actual;

  for (uint32_t buf = 0; buf < DeeployNetwork_num_outputs; buf++) {
    tot += DeeployNetwork_outputs_bytes[buf] / sizeof(OUTPUTTYPE);
    for (uint32_t i = 0;
         i < DeeployNetwork_outputs_bytes[buf] / sizeof(OUTPUTTYPE); i++) {
      expected = ((OUTPUTTYPE *)testOutputVector[buf])[i];
      actual = ((OUTPUTTYPE *)DeeployNetwork_outputs[buf])[i];
      diff = expected - actual;

#if ISOUTPUTfloat32_t == 1
      if ((diff < -1e-4) || (diff > 1e-4)) {
        tot_err += 1;
        printf("Expected: %10.6f  ", (float32_t)expected);
        printf("Actual: %10.6f  ", (float32_t)actual);
        printf("Diff: %10.6f at Index %12u in Output %u\r\n", (float32_t)diff, i,
               buf);
      }
#else
      if (diff != 0) {
        tot_err += 1;
        printf("Expected: %4d  ", expected);
        printf("Actual: %4d  ", actual);
        printf("Diff: %4d at Index %12u in Output %u\r\n", diff, i, buf);
      }
#endif
    }
  }

  printf("Errors: %d out of %d \r\n", tot_err, tot);

  return tot_err;
}

#endif  // training vs inference