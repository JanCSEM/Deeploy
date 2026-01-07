import numpy as np
import subprocess
import sys
import os
import threading
import queue
import time

# Define the number of epochs to match the C code's expectation
MEZO_NUM_EPOCHS = 5

def reader_thread(pipe, q):
    """Reads lines from a pipe and puts them into a queue."""
    try:
        for line in iter(pipe.readline, b''):
            q.put(line)
    finally:
        pipe.close()
        q.put(None) # Sentinel value to signal the end

def printer_thread(q, file_handle):
    """Reads lines from a queue and prints them to a file handle (stdout/stderr)."""
    while True:
        line = q.get()
        if line is None: # End of stream
            break
        print(line.decode('utf-8', errors='replace').strip(), file=file_handle, flush=True)

def stream_dataset_to_target(npz_path, target_executable):
    """
    Streams a dataset (images and labels) from an .npz file to a target's stdin.
    """
    # --- Step 1: Load the dataset ---
    try:
        with np.load(npz_path) as data:
            inputs = data['images']
            labels = data['labels']
            inputs = np.ascontiguousarray(inputs, dtype=np.float32)
            labels = np.ascontiguousarray(labels, dtype=np.uint32)
        print(f"Successfully loaded {len(inputs)} samples from '{npz_path}'.")
    except Exception as e:
        print(f"Fatal Error loading dataset: {e}", file=sys.stderr)
        sys.exit(1)

    # --- Step 2: Launch the C executable ---
    print(f"Attempting to launch executable: '{target_executable}'...")
    try:
        process = subprocess.Popen(
            [target_executable], 
            stdin=subprocess.PIPE, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE
        )
    except Exception as e:
        print(f"\nFatal Error launching executable: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Executable launched successfully. Starting stream for {MEZO_NUM_EPOCHS} epochs...")

    # --- Step 3: Start reader and printer threads to prevent deadlock ---
    q_stdout = queue.Queue()
    t_reader_stdout = threading.Thread(target=reader_thread, args=(process.stdout, q_stdout))
    t_printer_stdout = threading.Thread(target=printer_thread, args=(q_stdout, sys.stdout))

    q_stderr = queue.Queue()
    t_reader_stderr = threading.Thread(target=reader_thread, args=(process.stderr, q_stderr))
    t_printer_stderr = threading.Thread(target=printer_thread, args=(q_stderr, sys.stderr))

    for t in [t_reader_stdout, t_printer_stdout, t_reader_stderr, t_printer_stderr]:
        t.daemon = True
        t.start()

    # --- Step 4: Stream data for each epoch ---
    try:
        for epoch in range(MEZO_NUM_EPOCHS):
            # The C code prints the epoch summary, so we don't need to here.
            for i, (sample, label) in enumerate(zip(inputs, labels)):
                flat_sample = sample.flatten()
                process.stdin.write(flat_sample.tobytes())
                process.stdin.write(label.tobytes())
            process.stdin.flush()

    except BrokenPipeError:
        print(f"\nError: Target process terminated unexpectedly.", file=sys.stderr)
    except Exception as e:
        print(f"\nAn error occurred during streaming: {e}", file=sys.stderr)
    finally:
        if not process.stdin.closed:
            try:
                process.stdin.close()
            except BrokenPipeError:
                pass

    # --- Step 5: Wait for process and all threads to finish ---
    process.wait()
    for t in [t_reader_stdout, t_printer_stdout, t_reader_stderr, t_printer_stderr]:
        t.join()

    if process.returncode != 0:
        print(f"\nC program exited with non-zero status: {process.returncode}", file=sys.stderr)
    else:
        print("\nStream finished successfully.")


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python stream_script.py <path_to_dataset.npz> <path_to_target_executable>", file=sys.stderr)
        sys.exit(1)
    
    npz_file = sys.argv[1]
    executable_file = sys.argv[2]
    stream_dataset_to_target(npz_file, executable_file)