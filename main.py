# Shivang Chaudhary
# Course: MSc Artificial Intelligence
# Year: 2023-24
# This is the main file to run the main GEC model.
# Once results are recorded after running the file, 'Gecmodel_errEval.py' must be run to evaluate the -
# - recorded predictions and obtain final F0.5 scores.

import subprocess
import Gecmodel


def run_file(script):
    # Running the file using 'subprocess'.
    try:
        result = subprocess.run(['python', script], check=True, capture_output=True, text=True)
        print(f"Output of {script}:\n{result.stdout}")
    except subprocess.CalledProcessError as e:
        print(f"Error while running {script}:\n{e.stderr}")


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # Imported the GEC - script and putting in as parameter to run it.
    script = Gecmodel
    run_file(script)