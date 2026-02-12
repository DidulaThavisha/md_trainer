#!/usr/bin/env python3
"""
Ballerina Model Improvement Pipeline

Generates Ballerina solutions from an LLM, evaluates them against test cases,
provides iterative feedback, and produces SFT-optimized learning notes for
problems that required correction.
"""

import json
import zlib
import base64
import pickle
import subprocess
import os
import re
import sys
import time
import shutil
import tempfile
import argparse
from enum import Enum
from typing import Optional

from unsloth import FastLanguageModel
import torch

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_NAME = "didula-wso2/exp_23_emb_grpo_checkpoint_220_16bit_vllm"
DATA_FILE = "ballerina_grpo_X3.json"
MAX_RETRIES = 4
LOG_DIR = "log"
NOTES_DIR = "notes"
PROGRESS_FILE = "progress.json"
MAX_SEQ_LENGTH = 8192
BAL_TIMEOUT = 30  # seconds

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(NOTES_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Error Categories
# ---------------------------------------------------------------------------
class ErrorCategory(str, Enum):
    COMPILE_ERROR = "COMPILE_ERROR"
    RUNTIME_ERROR = "RUNTIME_ERROR"
    WRONG_ANSWER = "WRONG_ANSWER"
    TIMEOUT = "TIMEOUT"
    SUCCESS = "SUCCESS"


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------
def decompress_lcb_private_tests(text: str):
    """Decompress LiveCodeBench-style compressed test cases."""
    try:
        return json.loads(
            pickle.loads(zlib.decompress(base64.b64decode(text.encode("utf-8"))))
        )
    except Exception as e:
        print(f"  [WARN] Error decompressing tests: {e}")
        return []


def load_data(filepath: str) -> list:
    with open(filepath, "r") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Progress tracking
# ---------------------------------------------------------------------------
def load_progress(filepath: str) -> dict:
    """Load progress state; returns empty dict if no file exists."""
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            return json.load(f)
    return {}


def save_progress(filepath: str, progress: dict):
    with open(filepath, "w") as f:
        json.dump(progress, f, indent=2)


# ---------------------------------------------------------------------------
# Code extraction
# ---------------------------------------------------------------------------
def extract_code(response: str) -> str:
    """Extract Ballerina code from a markdown-fenced response."""
    match = re.search(r"```ballerina(.*?)```", response, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Try generic code block
    match = re.search(r"```(.*?)```", response, re.DOTALL)
    if match:
        return match.group(1).strip()
    return response.strip()


# ---------------------------------------------------------------------------
# Ballerina execution (proper project structure)
# ---------------------------------------------------------------------------
def run_ballerina_code(code: str, input_str: str) -> tuple[int, str, str]:
    """
    Execute Ballerina code as a standalone file.
    
    Creates a temp directory, writes main.bal, and runs `bal run main.bal`.
    We avoid creating a full project (Ballerina.toml) to prevent nesting issues
    and 'already within a Ballerina package' errors.
    """
    tmpdir = tempfile.mkdtemp(prefix="bal_eval_")
    try:
        # Write only the .bal file
        file_path = os.path.join(tmpdir, "main.bal")
        with open(file_path, "w") as f:
            f.write(code)

        process = subprocess.Popen(
            ["bal", "run", "main.bal"],
            cwd=tmpdir,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        stdout, stderr = process.communicate(input=input_str, timeout=BAL_TIMEOUT)
        return process.returncode, stdout, stderr

    except subprocess.TimeoutExpired:
        process.kill()
        return -1, "", "Execution Timed Out"
    except Exception as e:
        return -1, "", str(e)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Error categorization & feedback cleaning
# ---------------------------------------------------------------------------
def clean_stderr(stderr: str) -> str:
    """
    Strip noisy paths, stack traces, and build artifacts from Ballerina
    compiler/runtime output.  Keep only the meaningful error lines.
    """
    lines = stderr.strip().splitlines()
    cleaned = []
    for line in lines:
        # Skip blank lines and build progress
        if not line.strip():
            continue
        if line.strip().startswith("Compiling") or line.strip().startswith("Running"):
            continue
        # Strip absolute paths from error messages (keep filename:line:col)
        line = re.sub(r"/[^\s:]+/([^/\s:]+\.bal)", r"\1", line)
        # Skip Java stack traces
        if line.strip().startswith("at ") or "java." in line:
            continue
        cleaned.append(line)
    return "\n".join(cleaned).strip()


def categorize_error(
    returncode: int, stderr: str, stdout: str, expected: str
) -> ErrorCategory:
    """Determine the type of failure."""
    if returncode == -1 and "Timed Out" in stderr:
        return ErrorCategory.TIMEOUT
    if returncode != 0:
        # Distinguish compile vs runtime by checking stderr content
        lower = stderr.lower()
        if any(
            kw in lower
            for kw in ["compilation", "syntax", "undefined", "incompatible", "missing"]
        ):
            return ErrorCategory.COMPILE_ERROR
        return ErrorCategory.RUNTIME_ERROR

    # returncode == 0 but no output produced — likely a silent runtime error.
    # Ballerina `bal run` can return 0 even when the program panics/errors
    # if the compilation itself succeeded.  Check stderr for clues.
    cleaned_stderr = clean_stderr(stderr)
    if not stdout.strip() and cleaned_stderr:
        # There's error info in stderr even though returncode was 0
        lower = cleaned_stderr.lower()
        if any(
            kw in lower
            for kw in ["compilation", "syntax", "undefined", "incompatible", "missing"]
        ):
            return ErrorCategory.COMPILE_ERROR
        return ErrorCategory.RUNTIME_ERROR

    if stdout.strip() != expected.strip():
        return ErrorCategory.WRONG_ANSWER
    return ErrorCategory.SUCCESS


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def _truncate(text: str, max_len: int = 500) -> str:
    """Truncate long text to avoid blowing up the prompt context."""
    if len(text) <= max_len:
        return text
    return text[:max_len] + f"\n... (truncated, {len(text)} chars total)"


def evaluate_solution(
    code: str, test_cases: list
) -> tuple[bool, str, ErrorCategory]:
    """
    Run code against all test cases.  Only need to run test case 1 to detect
    compilation errors (they'll fail on every test case identically).
    Returns (success, feedback_message, error_category).
    """
    for i, test in enumerate(test_cases):
        inp = test.get("input", "")
        expected_out = test.get("output", "").strip()

        returncode, stdout, stderr = run_ballerina_code(code, inp)
        cleaned_stderr = clean_stderr(stderr)
        category = categorize_error(returncode, stderr, stdout, expected_out)

        if category == ErrorCategory.TIMEOUT:
            return (
                False,
                f"Timeout on Test Case {i+1}. Your solution is too slow — "
                f"consider a more efficient algorithm.",
                category,
            )

        if category in (ErrorCategory.COMPILE_ERROR, ErrorCategory.RUNTIME_ERROR):
            return (
                False,
                f"{category.value} on Test Case {i+1}:\n{cleaned_stderr}",
                category,
            )

        if category == ErrorCategory.WRONG_ANSWER:
            feedback_parts = [
                f"Wrong Answer on Test Case {i+1}.",
                f"Input:\n{_truncate(inp)}",
                f"Expected Output:\n{_truncate(expected_out)}",
                f"Actual Output:\n{_truncate(stdout.strip()) if stdout.strip() else '(empty — no output produced)'}",
            ]
            # Include stderr if present — it often has warnings/errors
            # that explain WHY the output is wrong or empty
            if cleaned_stderr:
                feedback_parts.append(
                    f"Stderr/Warnings:\n{_truncate(cleaned_stderr)}"
                )
            return (False, "\n".join(feedback_parts), category)

    return True, "All test cases passed.", ErrorCategory.SUCCESS


# ---------------------------------------------------------------------------
# Feedback prompt construction
# ---------------------------------------------------------------------------
def build_feedback_prompt(
    feedback: str,
    category: ErrorCategory,
    attempt: int,
    max_retries: int,
    python_ref: str,
) -> str:
    """
    Build a category-specific feedback prompt.
    Gives Python hint on the last retry for wrong-answer cases.
    """
    if category == ErrorCategory.COMPILE_ERROR:
        prompt = (
            "Your Ballerina code failed to compile.\n"
            f"Error:\n{feedback}\n\n"
            "Please fix the syntax/type errors and provide the corrected code."
        )
    elif category == ErrorCategory.RUNTIME_ERROR:
        prompt = (
            "Your Ballerina code compiled but crashed at runtime.\n"
            f"Error:\n{feedback}\n\n"
            "Please fix the runtime error and provide the corrected code."
        )
    elif category == ErrorCategory.TIMEOUT:
        prompt = (
            "Your Ballerina code timed out — it's too slow.\n"
            "Please re-think the algorithm for better time complexity "
            "and provide an optimized solution."
        )
    else:  # WRONG_ANSWER
        prompt = (
            "Your Ballerina code compiles and runs, but produces the wrong output.\n"
            f"{feedback}\n\n"
            "Please carefully re-check your logic and provide a corrected solution."
        )

    # On the last retry, provide the Python reference as a hint
    if attempt >= max_retries - 1 and python_ref:
        prompt += (
            "\n\nHere is a correct Python implementation for reference. "
            "Translate this logic carefully to Ballerina:\n"
            f"```python\n{python_ref}\n```"
        )

    return prompt


# ---------------------------------------------------------------------------
# Learning note generation (SFT-optimized)
# ---------------------------------------------------------------------------
LEARNING_NOTE_PROMPT = """\
You are a Ballerina programming expert. A model attempted a coding problem and \
failed before eventually solving it. Your job is to produce a **learning note** \
that will be used to fine-tune (SFT) the model so it avoids these mistakes in the future.

=== PROBLEM ===
{problem_description}

=== PYTHON REFERENCE ===
```python
{python_reference}
```

=== ATTEMPT HISTORY ===
{conversation_history}

=== INSTRUCTIONS FOR THE LEARNING NOTE ===

Write the note as a Markdown document optimised for SFT training.  
**Key principle: Models learn from concrete code examples, NOT from prose explanations.**

Structure the note with EXACTLY these sections:

### 1. Error Analysis
For EACH failed attempt, show the **wrong code snippet** and the **corrected code snippet** \
side-by-side as fenced Ballerina code blocks.  Tag each pair with the error category \
(COMPILE_ERROR / RUNTIME_ERROR / WRONG_ANSWER / TIMEOUT).

### 2. Ballerina Syntax & Idiom Drill
Provide **at least 5 diverse, standalone Ballerina code examples** that demonstrate the \
correct syntax/pattern the model got wrong.  Vary the examples across:
- Different input types (int, string, array, map, record)
- Different control flow (if/else, match, foreach, while)
- Edge cases (empty input, single element, large values)
Each example must be a complete, runnable Ballerina snippet.

### 3. Python → Ballerina Translation Patterns
Show **side-by-side** Python and Ballerina code for the key constructs used in this \
problem. Cover:
- I/O (reading stdin, printing)
- Data structures (list→array, dict→map, set)
- String manipulation
- Type conversions

### 4. Problem-Solving Template
Provide a step-by-step Ballerina code skeleton for this type of problem that the model \
can follow as a pattern. Include comments at each step.

### 5. Final Correct Solution
The complete, working Ballerina solution with inline comments explaining each key decision.

IMPORTANT RULES:
- Every section MUST contain Ballerina code blocks.  Minimize prose(Not eradicate, but minimize).
- Do NOT say "remember to" or "make sure to" — instead SHOW the correct code.  
- Prefer MANY short code examples over few long explanations.
- The note must be self-contained: a reader seeing ONLY this note should learn the pattern.
- Focus on how to **teach the model** to get the best out of it. The teaching style should be optimized for SFT (clear patterns, diverse examples).

"""


def generate_learning_note(
    model,
    tokenizer,
    conversation_history: str,
    problem_description: str,
    python_reference: str,
) -> str:
    """Generate an SFT-optimized learning note using the model."""
    prompt = LEARNING_NOTE_PROMPT.format(
        problem_description=problem_description,
        python_reference=python_reference,
        conversation_history=conversation_history,
    )

    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    ).to("cuda")
    outputs = model.generate(inputs, max_new_tokens=4096, use_cache=True)
    return tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Ballerina Model Improvement Pipeline")
    parser.add_argument(
        "--max-problems", type=int, default=None, help="Max problems to process (for testing)"
    )
    parser.add_argument(
        "--start-from", type=int, default=0, help="Problem index to start from (0-based)"
    )
    parser.add_argument(
        "--reset-progress", action="store_true", help="Ignore saved progress and start fresh"
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    print(f"Loading model: {MODEL_NAME}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)

    # Fix pad_token == eos_token issue (prevents attention mask warning)
    if tokenizer.pad_token is None or tokenizer.pad_token == tokenizer.eos_token:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

    # ------------------------------------------------------------------
    # Load data & progress
    # ------------------------------------------------------------------
    data = load_data(DATA_FILE)
    progress = {} if args.reset_progress else load_progress(PROGRESS_FILE)

    total = len(data)
    end_idx = min(total, args.start_from + args.max_problems) if args.max_problems else total

    stats = {"first_pass": 0, "recovered": 0, "failed": 0, "skipped": 0, "total": 0}

    print(f"Processing problems {args.start_from} to {end_idx - 1} ({end_idx - args.start_from} total).")

    for problem_idx in range(args.start_from, end_idx):
        problem = data[problem_idx]
        pid = str(problem_idx)
        stats["total"] += 1

        # Skip already-completed problems
        if pid in progress and progress[pid].get("status") in ("solved", "failed"):
            print(f"[{problem_idx+1}/{total}] Skipping (already processed).")
            stats["skipped"] += 1
            continue

        print(f"\n{'='*60}")
        print(f"[{problem_idx+1}/{total}] Processing...")

        description = problem["prompt"]
        test_cases = decompress_lcb_private_tests(problem.get("answer", ""))
        python_ref = problem.get("python", "")

        if not test_cases:
            print("  Skipping — missing/invalid test cases.")
            progress[pid] = {"status": "skipped", "reason": "no_test_cases"}
            save_progress(PROGRESS_FILE, progress)
            continue

        # --------------------------------------------------------------
        # Feedback loop
        # --------------------------------------------------------------
        conversation_log = []
        solved = False

        for attempt in range(MAX_RETRIES):
            # Build a SINGLE-TURN prompt for each attempt.
            # Multi-turn accumulation overflows max_seq_length by attempt 3-4,
            # causing the model to lose feedback entirely.
            if attempt == 0:
                prompt_content = f"Solve this problem in Ballerina:\n{description}"
            else:
                # Summarize previous attempts into a compact single-turn prompt
                prev = conversation_log[-1]
                prev_code = prev["code"]
                prev_feedback = prev["feedback"]
                prev_category = prev["category"]

                fb_section = build_feedback_prompt(
                    prev_feedback, ErrorCategory(prev_category),
                    attempt, MAX_RETRIES, python_ref
                )

                prompt_content = (
                    f"Solve this problem in Ballerina:\n{description}\n\n"
                    f"--- Your Previous Attempt (attempt {attempt}) ---\n"
                    f"```ballerina\n{prev_code}\n```\n\n"
                    f"--- Feedback ---\n{fb_section}\n\n"
                    f"Provide a completely corrected Ballerina solution."
                )

            messages = [{"role": "user", "content": prompt_content}]

            inputs = tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
            ).to("cuda")

            # Create proper attention mask
            attention_mask = torch.ones_like(inputs).to("cuda")

            token_count = inputs.shape[-1]
            print(f"  [Attempt {attempt+1}] Input tokens: {token_count}")
            if token_count > MAX_SEQ_LENGTH:
                print(f"  ⚠️  WARNING: {token_count} tokens exceeds max_seq_length={MAX_SEQ_LENGTH}!")

            gen_kwargs = dict(max_new_tokens=2048, use_cache=True, attention_mask=attention_mask)
            if attempt > 0:
                gen_kwargs.update(temperature=0.7, do_sample=True)

            outputs = model.generate(inputs, **gen_kwargs)
            response = tokenizer.decode(
                outputs[0][len(inputs[0]):], skip_special_tokens=True
            )

            code = extract_code(response)
            success, feedback, category = evaluate_solution(code, test_cases)

            conversation_log.append(
                {
                    "attempt": attempt + 1,
                    "response": response,
                    "code": code,
                    "feedback": feedback,
                    "category": category.value,
                }
            )

            if success:
                print(f"  ✅ Solved on attempt {attempt + 1}")
                solved = True
                break
            else:
                summary = feedback[:200].replace("\n", " ")
                print(f"  ❌ Attempt {attempt + 1} [{category.value}]: {summary}")
                # Also log the actual output for debugging
                actual_out = feedback.split("Actual Output:\n")[-1] if "Actual Output:" in feedback else "(see above)"
                print(f"     Actual output: {actual_out[:150]}")

        # --------------------------------------------------------------
        # Post-processing
        # --------------------------------------------------------------
        if solved and len(conversation_log) == 1:
            # First-attempt success — nothing to learn
            stats["first_pass"] += 1
            progress[pid] = {"status": "solved", "attempts": 1}
            print("  (First-pass success — no learning note needed.)")

        elif solved and len(conversation_log) > 1:
            # Recovered after failures — generate learning note
            stats["recovered"] += 1
            print("  Generating learning note...")

            note = generate_learning_note(
                model,
                tokenizer,
                json.dumps(conversation_log, indent=2),
                description,
                python_ref,
            )

            # Save note
            note_filename = os.path.join(
                NOTES_DIR, f"problem_{problem_idx+1}_learning_note.md"
            )
            with open(note_filename, "w") as f:
                f.write(note)
            print(f"  Saved note → {note_filename}")

            # Save log
            log_filename = os.path.join(
                LOG_DIR, f"problem_{problem_idx+1}_log.json"
            )
            with open(log_filename, "w") as f:
                json.dump(conversation_log, f, indent=2)
            print(f"  Saved log  → {log_filename}")

            progress[pid] = {
                "status": "solved",
                "attempts": len(conversation_log),
                "note": note_filename,
                "log": log_filename,
            }
        else:
            # Failed all retries
            stats["failed"] += 1
            # Still save the log for analysis
            log_filename = os.path.join(
                LOG_DIR, f"problem_{problem_idx+1}_failed_log.json"
            )
            with open(log_filename, "w") as f:
                json.dump(conversation_log, f, indent=2)
            print(f"  Saved failed log → {log_filename}")

            progress[pid] = {
                "status": "failed",
                "attempts": len(conversation_log),
                "log": log_filename,
                "last_category": conversation_log[-1]["category"],
            }

        save_progress(PROGRESS_FILE, progress)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("Pipeline Complete!")
    print(f"  Total processed : {stats['total']}")
    print(f"  Skipped (cached): {stats['skipped']}")
    print(f"  First-pass pass : {stats['first_pass']}")
    print(f"  Recovered       : {stats['recovered']}")
    print(f"  Failed          : {stats['failed']}")
    print(f"Progress saved to {PROGRESS_FILE}")


if __name__ == "__main__":
    main()
