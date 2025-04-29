import json

def fix_json_file(input_file, output_file):
    with open(input_file, 'r') as f:
        content = f.read()
    
    # Split the content at the problematic ] [
    parts = content.split('] [')
    
    # Parse each part as JSON
    all_data = []
    for part in parts:
        # Clean up the part
        part = part.strip()
        if part.startswith('['):
            part = part[1:]
        if part.endswith(']'):
            part = part[:-1]
        
        # Parse the JSON objects
        try:
            data = json.loads('[' + part + ']')
            all_data.extend(data)
        except json.JSONDecodeError as e:
            print(f"Error parsing part: {e}")
            continue
    
    # Write the fixed JSON
    with open(output_file, 'w') as f:
        json.dump(all_data, f, indent=4)

if __name__ == '__main__':
    fix_json_file('data/train.json', 'data/train.json.fixed') 