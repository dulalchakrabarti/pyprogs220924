# Sample dictionaries
dict1 = {'a': 1, 'b': 2, 'c': 3, 'd': 4}
dict2 = {'b': 5, 'c': 6, 'e': 7, 'f': 8}

# Method 1: Using set intersection
common_keys = dict1.keys() & dict2.keys()

# Create a list of values for common keys
result = [(key, dict1[key], dict2[key]) for key in common_keys]

# Print results
print("Common keys and their values:")
for key, val1, val2 in result:
    print(f"Key: {key}, dict1 value: {val1}, dict2 value: {val2}")

# Alternative Method 2: Using dictionary comprehension
result_dict = {key: (dict1[key], dict2[key]) for key in dict1 if key in dict2}

# Print results
print("\nUsing dictionary comprehension:")
for key, (val1, val2) in result_dict.items():
    print(f"Key: {key}, dict1 value: {val1}, dict2 value: {val2}")
