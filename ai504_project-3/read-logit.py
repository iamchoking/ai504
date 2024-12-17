import numpy as np

def read_npy_file(file_path, num_lines=100):
    # Load the .npy file
    data = np.load(file_path)
    
    # Check if the data has at least 'num_lines' entries
    if len(data) < num_lines:
        print(f"The file contains less than {num_lines} lines. Showing all available lines.")
        num_lines = len(data)
    
    # Print the first 'num_lines' of the data
    print(data[:num_lines])

# Example usage:
file_path = '20243406.npy'  # Replace with the path to your .npy file
read_npy_file(file_path)