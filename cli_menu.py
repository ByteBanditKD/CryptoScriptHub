import os
import subprocess
from pathlib import Path

# Define the root directory (where this script resides)
ROOT_DIR = Path(__file__).parent

# Define the folder structure
FOLDERS = {
    "1": {"name": "support-resistance", "desc": "Scripts for identifying support and resistance levels"},
    "2": {"name": "scanners", "desc": "Various Scanner Tools"},
    "3": {"name": "price-prediction", "desc": "Machine learning models for price forecasting"},
    "4": {"name": "trend-analysis", "desc": "Scripts for analyzing market trends"},
}

def clear_screen():
    """Clear the command-line screen."""
    os.system('cls' if os.name == 'nt' else 'clear')

def list_scripts(folder):
    """List all .py files in the specified folder."""
    folder_path = ROOT_DIR / folder
    scripts = [f for f in folder_path.glob("*.py") if f.is_file()]
    return scripts

def display_menu():
    """Display the main menu."""
    clear_screen()
    print("=== Project Script Runner ===")
    for key, info in FOLDERS.items():
        print(f"{key}. {info['name']} - {info['desc']}")
    print("0. Exit")
    print("========================")

def run_script(folder, script_path):
    """Run the selected Python script."""
    try:
        subprocess.run(["python", str(script_path)], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running script: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    input("\nPress Enter to return to the menu...")

def main():
    while True:
        display_menu()
        choice = input("Select an option (0-4): ").strip()

        if choice == "0":
            print("Exiting...")
            break
        elif choice in FOLDERS:
            folder = FOLDERS[choice]["name"]
            scripts = list_scripts(folder)

            if not scripts:
                print(f"No Python scripts found in {folder}!")
                input("Press Enter to continue...")
                continue

            clear_screen()
            print(f"\n=== Scripts in {folder} ===")
            for i, script in enumerate(scripts, 1):
                print(f"{i}. {script.name}")
            print("0. Back")
            print("========================")

            script_choice = input("Select a script to run (0 to go back): ").strip()
            if script_choice == "0":
                continue
            try:
                script_idx = int(script_choice) - 1
                if 0 <= script_idx < len(scripts):
                    run_script(folder, scripts[script_idx])
                else:
                    print("Invalid script selection!")
                    input("Press Enter to continue...")
            except ValueError:
                print("Please enter a valid number!")
                input("Press Enter to continue...")
        else:
            print("Invalid option!")
            input("Press Enter to continue...")

if __name__ == "__main__":
    main()