import numpy as np

# Load the .npz file
data = np.load('parameters.npz')

# Check the names of the arrays stored
print(data.files)

# Convert and save each array to a separate .txt file
for name in data.files:
    array = data[name]
    print(f"Saving {name} with shape {array.shape} to {name}.txt")
    np.savetxt(f"{name}.txt", array, fmt='%s')  # fmt='%s' works for both numbers and strings

