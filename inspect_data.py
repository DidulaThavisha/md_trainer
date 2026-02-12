import json
import zlib
import base64
import pickle

def decompress_lcb_private_tests(text: str):
    return json.loads(
        pickle.loads(zlib.decompress(base64.b64decode(text.encode("utf-8"))))
    )

with open("ballerina_grpo_X3.json", "r") as f:
    data = json.load(f)

first_problem = data[0]
print("Problem Title:", first_problem.get("prompt").split("\n")[0])
if "answer" in first_problem:
    try:
        test_cases = decompress_lcb_private_tests(first_problem["answer"])
        print("Test Cases Type:", type(test_cases))
        print("Test Cases Content (First 2):", test_cases[:2] if isinstance(test_cases, list) else test_cases)
    except Exception as e:
        print("Error decompressing:", e)
else:
    print("No 'answer' field found.")
