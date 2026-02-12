
import json
import zlib
import base64
import pickle
import subprocess
import os
import re
import time
from unsloth import FastLanguageModel
import torch

# Configuration
MODEL_NAME = "didula-wso2/exp_23_emb_grpo_checkpoint_220_16bit_vllm"
DATA_FILE = "ballerina_grpo_X3.json"
MAX_RETRIES = 3
LOG_DIR = "log"
NOTES_DIR = "notes"

# Ensure directories exist
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(NOTES_DIR, exist_ok=True)

def decompress_lcb_private_tests(text: str):
    try:
        return json.loads(
            pickle.loads(zlib.decompress(base64.b64decode(text.encode("utf-8"))))
        )
    except Exception as e:
        print(f"Error decompressing tests: {e}")
        return []

def load_data(filepath):
    with open(filepath, "r") as f:
        return json.load(f)

def extract_code(response: str):
    match = re.search(r"```ballerina(.*?)```", response, re.DOTALL)
    if match:
        return match.group(1).strip()
    return response.strip() # Fallback if no code block

def run_ballerina_code(code: str, input_str: str):
    filename = "temp_solution.bal"
    with open(filename, "w") as f:
        f.write(code)

    try:
        # Using subprocess to run 'bal run'
        # We need to pass input via stdin
        process = subprocess.Popen(
            ["bal", "run", filename],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        stdout, stderr = process.communicate(input=input_str, timeout=10) # Timeout to prevent infinite loops
        return process.returncode, stdout, stderr
    except subprocess.TimeoutExpired:
        process.kill()
        return -1, "", "Execution Timed Out"
    except Exception as e:
        return -1, "", str(e)
    finally:
        if os.path.exists(filename):
            os.remove(filename)

def evaluate_solution(code: str, test_cases: list):
    for i, test in enumerate(test_cases):
        inp = test.get("input", "")
        expected_out = test.get("output", "").strip()
        
        returncode, stdout, stderr = run_ballerina_code(code, inp)
        
        if returncode != 0:
            return False, f"Compile/Runtime Error on Test Case {i+1}:\n{stderr}"
        
        if stdout.strip() != expected_out:
            return False, f"Wrong Answer on Test Case {i+1}.\nInput:\n{inp}\nExpected Output:\n{expected_out}\nActual Output:\n{stdout.strip()}"
            
    return True, "Success"

def generate_learning_note(model, tokenizer, conversation_history, problem_description, python_reference):
    prompt = f"""
    You have successfully solved the problem after some failed attempts. 
    Problem: {problem_description}
    
    Python Reference Logic:
    {python_reference}

    Conversation History (Failures -> Success):
    {conversation_history}
    
    Your Task: write a **comprehensive learning note** as a Markdown file.
    Target Audience: A future version of yourself (an AI model) being fine-tuned (SFT).
    
    Instructions:
    1. Analyze the mistakes made in previous attempts.
    2. Explain *why* the initial approach was wrong and *how* to think correctly for this specific problem type in Ballerina.
    3. Provide a **variety of code samples** showing the correct syntax/logic in different scenarios. Do NOT just explain. code is crucial.
    4. Focus on how to **teach the model** to get the best out of it. The teaching style should be optimized for SFT (clear patterns, diverse examples).
    5. If the logic was complex, explain how to translate the Python reference logic to Ballerina efficiently.
    """
    
    # Simple generation call (abstracted for clarity)
    inputs = tokenizer([prompt], return_tensors = "pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens = 2048, use_cache = True)
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

def main():
    print(f"Loading model: {MODEL_NAME}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = MODEL_NAME,
        max_seq_length = 4096,
        dtype = None,
        load_in_4bit = True,
    )
    FastLanguageModel.for_inference(model)

    data = load_data(DATA_FILE)
    
    for problem_idx, problem in enumerate(data):
        print(f"Processing Problem {problem_idx + 1}...")
        description = problem["prompt"]
        test_cases = decompress_lcb_private_tests(problem.get("answer", ""))
        python_ref = problem.get("python", "")
        
        if not test_cases:
            print("Skipping problem due to missing/invalid test cases.")
            continue

        # Initial Attempt
        messages = [{"role": "user", "content": f"Solve this problem in Ballerina:\n{description}"}]
        
        conversation_log = []
        solved = False
        
        for attempt in range(MAX_RETRIES):
            # Generate Code
            inputs = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to("cuda")
            outputs = model.generate(inputs, max_new_tokens = 1024, use_cache = True)
            response = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
            
            code = extract_code(response)
            success, feedback = evaluate_solution(code, test_cases)
            
            conversation_log.append({"attempt": attempt + 1, "response": response, "feedback": feedback})
            messages.append({"role": "assistant", "content": response})
            
            if success:
                print(f"Problem {problem_idx + 1} Solved on attempt {attempt + 1}!")
                solved = True
                break
            else:
                print(f"Attempt {attempt + 1} Failed: {feedback[:100]}...") # Print summary
                
                # feedback strategy
                feedback_prompt = f"Your solution failed.\nFeedback:\n{feedback}\nPlease fix the code."
                
                # If it's a logic error and we are on the last retry, give the python hint
                if "Wrong Answer" in feedback and attempt == MAX_RETRIES - 2:
                     feedback_prompt += f"\nHere is a correct Python implementation as a reference logic:\n{python_ref}\nTranslate this logic carefully to Ballerina."
                
                messages.append({"role": "user", "content": feedback_prompt})

        # Post-Processing
        if solved and len(conversation_log) > 1: # Only if we had failures and recovered
            print("Generating Learning Note...")
            note = generate_learning_note(model, tokenizer, json.dumps(conversation_log, indent=2), description, python_ref)
            
            # Save Note
            note_filename = os.path.join(NOTES_DIR, f"problem_{problem_idx+1}_learning_note.md")
            with open(note_filename, "w") as f:
                f.write(note)
            print(f"Saved note to {note_filename}")

            # Save Log
            log_filename = os.path.join(LOG_DIR, f"problem_{problem_idx+1}_log.json")
            with open(log_filename, "w") as f:
                json.dump(conversation_log, f, indent=2)
            print(f"Saved log to {log_filename}")

if __name__ == "__main__":
    main()
