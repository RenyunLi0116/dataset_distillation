def check_duplicates(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
    
    # Remove newline characters and split lines by delimiter
    processed_lines = [line.strip().split(', ') for line in lines]
    
    # Use a set to find duplicates
    unique_lines = set(tuple(line) for line in processed_lines)
    
    # Check for duplicates
    if len(unique_lines) == len(processed_lines):
        print("No duplicates found.")
    else:
        print("Duplicates found.")

# Example usage
check_duplicates('high_loss_c2w_stage_100.txt')
