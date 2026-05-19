import os

# The folders you want to scan
TARGET_DIRS = ['src', 'scripts']
OUTPUT_FILE = 'codebase_dump.txt'

def is_text_file(filepath):
    """Basic check to skip binaries or unreadable files."""
    try:
        with open(filepath, 'tr', encoding='utf-8') as check_file:
            check_file.read(1024)
            return True
    except:
        return False

def main():
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as outfile:
        for directory in TARGET_DIRS:
            if not os.path.exists(directory):
                outfile.write(f"=== DIRECTORY NOT FOUND: {directory} ===\n\n")
                continue
                
            for root, _, files in os.walk(directory):
                for file in files:
                    filepath = os.path.join(root, file)
                    
                    # Skip cache and compiled files
                    if '__pycache__' in filepath or filepath.endswith('.pyc'):
                        continue
                        
                    if not is_text_file(filepath):
                        continue
                        
                    # Write the visual separator and file path
                    outfile.write("\n" + "=" * 80 + "\n")
                    outfile.write(f"FILE: {filepath}\n")
                    outfile.write("=" * 80 + "\n\n")
                    
                    # Write the file's contents
                    try:
                        with open(filepath, 'r', encoding='utf-8') as infile:
                            outfile.write(infile.read())
                    except Exception as e:
                        outfile.write(f"[Error reading file: {e}]\n")
                    
                    outfile.write("\n")
                    
    print(f"Done! Codebase dumped to '{OUTPUT_FILE}'.")

if __name__ == '__main__':
    main()