import re

def clean_stderr(stderr: str) -> str:
    lines = stderr.strip().splitlines()
    cleaned = []
    for line in lines:
        stripped_line = line.strip()
        if not stripped_line:
            continue
        if stripped_line.startswith("Compiling") or stripped_line.startswith("Running"):
            continue

        line = re.sub(r"/[^\s:]+/([^/\s:]+\.bal)", r"\1", line)

        if stripped_line.startswith("at "):
            print(f"DEBUG: checking line '{line}'")
            if ".bal" in line:
                print("  DEBUG: kept (has .bal)")
                cleaned.append(line)
            else:
                print("  DEBUG: skipped (no .bal)")
            continue
        
        if "java.lang" in line or "io.ballerina.runtime" in line:
            continue

        cleaned.append(line)
    return "\n".join(cleaned).strip()

stderr = "error: something\n\tat java.lang.Thread.run(Thread.java:750)\n\tat io.ballerina.runtime.internal.BalRuntime.start(BalRuntime.java:100)"
print(f"Input:\n{stderr}\n")
cleaned = clean_stderr(stderr)
print(f"Output:\n{cleaned}")
