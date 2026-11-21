
import json
import re
import urllib.request
from pathlib import Path

MODEL = "qwen2.5-coder:3b"
API_URL = "http://localhost:11434/api/generate"

PROMPT = """
Write a complete, correct C program for matrix multiplication using OpenMP.

Requirements:
1. Use N = 1000.
2. Declare A, B and C as global double arrays of size N x N.
3. Initialize every element of A to 1.0, B to 2.0 and C to 0.0.
4. Implement the conventional i-j-k matrix multiplication algorithm.
5. Parallelize the outer i loop using OpenMP.
6. Use omp_get_wtime() to measure only the matrix multiplication
   computation time, excluding initialization and verification.
7. Verify every element of C against the expected value 2*N.
8. Print:
   - Execution time in seconds using the label "Execution time:"
   - Verification value for C[0][0] using the label
     "Verification value:"
   - Whether the full matrix verification passed.
9. Include all necessary headers.
10. Produce compilable C code.

Do not use loop interchange, tiling, manual SIMD intrinsics,
or other advanced optimizations. This is the one-shot baseline.

Return ONLY the C source code. Do not include Markdown fences
or explanations.
"""

def main():
    payload = {
        "model": MODEL,
        "prompt": PROMPT,
        "stream": False,
        "options": {
            "temperature": 0.2,
            "seed": 42
        }
    }

    request = urllib.request.Request(
        API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    print("Sending matrix multiplication task to Qwen...")

    with urllib.request.urlopen(request, timeout=600) as response:
        result = json.loads(response.read().decode("utf-8"))

    generated = result["response"].strip()

    # Remove Markdown fences if the model includes them.
    match = re.search(r"```(?:c)?\s*(.*?)```", generated, re.DOTALL)
    if match:
        generated = match.group(1).strip()

    output_dir = Path("kernels/matrix_multiplication")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "llm_generated_openmp.c"
    output_file.write_text(generated + "\n", encoding="utf-8")

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    (results_dir / "llm_one_shot_raw.txt").write_text(
        result["response"], encoding="utf-8"
    )

    print("\nGeneration completed.")
    print(f"Generated C file: {output_file}")
    print(f"Raw model response: {results_dir / 'llm_one_shot_raw.txt'}")
    print(f"Model: {MODEL}")
    print(f"Generation duration: {result.get('total_duration', 0) / 1e9:.2f} seconds")


if __name__ == "__main__":
    main()
