from colorama import Fore, Style, init
import pprint

# Initialize colorama
init()

def print_yaml_colored(obj, indent=0):
    """
    Print any Python object in YAML-style format with colors.
    - Keys are cyan
    - Values have colors based on their type:
      - Strings: green
      - Numbers: yellow
      - Booleans/None: magenta
      - Lists/Dicts: recursively formatted
    """
    if isinstance(obj, dict):
        for key, value in obj.items():
            print(" " * indent + f"{Fore.CYAN}{key}{Style.RESET_ALL}:", end="")
            
            if isinstance(value, (dict, list)):
                print()
                print_yaml_colored(value, indent + 2)
            else:
                format_value(value)
                
    elif isinstance(obj, list):
        for item in obj:
            print(" " * indent + f"{Fore.CYAN}-{Style.RESET_ALL}", end="")
            if isinstance(item, (dict, list)):
                print()
                print_yaml_colored(item, indent + 2)
            else:
                format_value(item)
    else:
        format_value(obj)

def format_value(value):
    if isinstance(value, str):
        print(f" {Fore.GREEN}{value}{Style.RESET_ALL}")
    elif isinstance(value, (int, float)):
        print(f" {Fore.YELLOW}{value}{Style.RESET_ALL}")
    elif isinstance(value, (bool, type(None))):
        print(f" {Fore.MAGENTA}{value}{Style.RESET_ALL}")
    else:
        print(f" {value}")

# Example usage
example = {
    "person": {
        "name": "John Doe",
        "age": 30,
        "is_active": True,
        "skills": ["Python", "JavaScript", "Docker"],
        "address": {
            "city": "New York",
            "zip": 10001
        }
    },
    "projects": [
        {"name": "Project A", "status": "completed"},
        {"name": "Project B", "status": "in progress"}
    ],
    "notes": None
}

print_yaml_colored(example)