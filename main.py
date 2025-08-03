import subprocess
import sys
import os

def run_script(script_name):
    module_name = script_name.replace(".py", "")
    try:
        # Import the module dynamically
        module = __import__(f"src.{module_name}", fromlist=[None])
        exec(f"import src.{module_name}")
    except Exception as e:
        print(f"Error running {script_name}: {e}")

if __name__ == "__main__":
    for i in range(1, 8):
        script = f"chapter_{i}.py"
        print(f"Running {script}...")
        run_script(script)