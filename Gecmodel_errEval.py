# Author: Shivang Chaudhary
# Course: MSc Artificial Intelligence
# Year: 2023-24
# The following is the file to produce scores of the recorded sentences files.
# IT REQUIRES THE M2 FORMAT OF THE PREDICTED AND REFERENCE TEXT FILES TO PRODUCE THE FINAL SCORES.

import subprocess


def create_blank_file(filepath):
    """Creates a blank file at the given path if it doesn't exist.
  Args:
      filepath: The path to the file.
  """
    with open(filepath, "w") as file:
        pass


# Verifying the file paths ---
filepath1 = "Err_files_evals/results.txt"
create_blank_file(filepath1)

filepath2 = "Err_files_evals/ref_m2.txt"
create_blank_file(filepath2)
# ------


# Running the ERRANT commands
subprocess.run(
    ["errant_parallel", "-orig", "Err_files_evals/clang8.incorrect", "-cor", "Err_files_evals/predictions.txt", "-out",
     "Err_files_evals/results.txt"],
    capture_output=True,  # Capture stdout and stderr
    text=True  # Return the output as a string
)
# subprocess.run(["errant_m2", "-silver", "Err_files_evals/references.txt", "-out", "Err_files_evals/ref_m2.txt"],  capture_output=True, text=True)
subprocess.run(
    ["errant_parallel", "-orig", "Err_files_evals/references.txt", "-cor", "Err_files_evals/predictions.txt", "-out",
     "Err_files_evals/ref_m2.txt"],
    capture_output=True,  # Capture stdout and stderr
    text=True  # Return the output as a string
)
output = subprocess.run(["errant_compare", "-hyp", "Err_files_evals/results.txt", "-ref", "Err_files_evals/ref_m2.txt"],
                        capture_output=True,  # Capture stdout and stderr
                        text=True  # Return the output as a string
                        )

# -------- Saving the final score
output_file_path = 'Err_files_evals/Err_score.txt'
# Writing the output to file,
with open(output_file_path, "w") as file:
    file.write("Standard Output:\n")
    file.write(output.stdout)
    file.write("\n\nStandard Error:\n")
    file.write(output.stderr)
# ---------
