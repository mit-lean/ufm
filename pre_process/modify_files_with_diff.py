import sys
import json
def apply_differences_to_new_file(original_file_path, modified_file_path, diff_file_path):
    # load differences from diff json file provided
    with open(diff_file_path, 'r') as diff_file:
        differences = json.load(diff_file)
    # modify file with differences
    new_content = []    
    for line in differences:
        if line.startswith('- '):  # Line to be removed
            continue
        elif line.startswith('+ '):  # Line to be added
            new_content.append(line[2:])
        elif line.startswith('  '):  # Unchanged line
            new_content.append(line[2:])
    with open(modified_file_path, 'w') as new_file:
        new_file.writelines(new_content)
    print(f"Saved modified path in {modified_file_path} using the provided diff json.")
apply_differences_to_new_file(sys.argv[1], sys.argv[2], sys.argv[3])