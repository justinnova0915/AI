import time
import sys

for i in range(11):
    # The \r moves the cursor to the start of the line
    # The spaces ensure the new output overwrites the old, especially if the new string is shorter
    output = f"\rProgress: {i*10}% complete{' ' * 10}" 
    print(output, end='', flush=True)
    time.sleep(0.5)

print("\nProcess finished.") # Print a final newline after the loop is done
