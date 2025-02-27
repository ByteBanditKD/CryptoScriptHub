import tkinter as tk
from tkinter import messagebox, scrolledtext
import subprocess
from pathlib import Path

# Define the root directory and folder structure
ROOT_DIR = Path(__file__).parent
FOLDERS = {
    "Support-Resistance": {"path": "support-resistance", "desc": "Scripts for identifying support and resistance levels"},
    "Scanners": {"path": "scanners", "desc": "Scanner Analysis tools"},
    "Price-Prediction": {"path": "price-prediction", "desc": "Machine learning models for price forecasting"},
    "Trend-Analysis": {"path": "trend-analysis", "desc": "Scripts for analyzing market trends"},
}

class ScriptRunnerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Project Script Runner")
        self.root.geometry("600x700")
        self.root.resizable(True, True)

        # Title
        tk.Label(root, text="Project Script Runner", font=("Arial", 16, "bold")).pack(pady=10)

        # Folder frame
        self.folder_frame = tk.Frame(root)
        self.folder_frame.pack(pady=10)

        for folder_name, info in FOLDERS.items():
            btn = tk.Button(self.folder_frame, text=f"{folder_name}\n({info['desc']})", 
                           command=lambda f=info["path"]: self.show_scripts(f),
                           width=30, height=3, wraplength=250, justify="center", bg="#e0e0e0")
            btn.pack(pady=5)

        # Script frame (hidden initially)
        self.script_frame = tk.Frame(root)
        self.script_buttons = []

        # Back button (hidden initially)
        self.back_button = tk.Button(root, text="Back to Folders", command=self.show_folders, width=20, bg="#d0d0d0")
        self.back_button.pack(pady=10)
        self.back_button.pack_forget()

        # Output area
        self.output_label = tk.Label(root, text="Script Output:", font=("Arial", 12))
        self.output_label.pack(pady=5)
        self.output_text = scrolledtext.ScrolledText(root, height=15, width=70, wrap=tk.WORD, font=("Consolas", 10))
        self.output_text.pack(pady=5)
        self.output_text.config(state="disabled")

        # Clear output button
        self.clear_button = tk.Button(root, text="Clear Output", command=self.clear_output, width=20, bg="#d0d0d0")
        self.clear_button.pack(pady=5)

        # Exit button
        tk.Button(root, text="Exit", command=root.quit, width=20, bg="#ff4d4d", fg="white").pack(pady=20)

    def show_folders(self):
        """Show the folder selection screen."""
        self.script_frame.pack_forget()
        self.back_button.pack_forget()
        self.folder_frame.pack(pady=10)
        for btn in self.script_buttons:
            btn.destroy()
        self.script_buttons.clear()

    def show_scripts(self, folder):
        """Show available scripts in the selected folder."""
        self.folder_frame.pack_forget()
        self.script_frame.pack(pady=10)
        self.back_button.pack(pady=10)

        scripts = [f for f in (ROOT_DIR / folder).glob("*.py") if f.is_file()]
        if not scripts:
            messagebox.showinfo("No Scripts", f"No Python scripts found in {folder}!")
            self.show_folders()
            return

        tk.Label(self.script_frame, text=f"Scripts in {folder}", font=("Arial", 12)).pack(pady=5)
        for script in scripts:
            btn = tk.Button(self.script_frame, text=script.name, 
                           command=lambda s=script: self.run_script(s),
                           width=30, height=2, bg="#e0e0e0")
            btn.pack(pady=5)
            self.script_buttons.append(btn)

    def run_script(self, script_path):
        """Run the script and display output in the GUI."""
        self.output_text.config(state="normal")
        self.output_text.delete(1.0, tk.END)
        self.output_text.insert(tk.END, f"Running {script_path.name}...\n\n")
        self.output_text.config(state="disabled")
        self.root.update()  # Force UI update to show "Running" message

        try:
            # Run the script and capture output
            result = subprocess.run(["python", str(script_path)], 
                                  cwd=script_path.parent, 
                                  capture_output=True, 
                                  text=True, 
                                  timeout=60)  # Optional: 60-second timeout
            
            self.output_text.config(state="normal")
            if result.stdout:
                self.output_text.insert(tk.END, "Output:\n" + result.stdout + "\n")
            if result.stderr:
                self.output_text.insert(tk.END, "Errors:\n" + result.stderr + "\n")
            if result.returncode == 0:
                self.output_text.insert(tk.END, "Script completed successfully.")
            else:
                self.output_text.insert(tk.END, f"Script failed with return code {result.returncode}.")
            self.output_text.config(state="disabled")
            self.output_text.see(tk.END)  # Scroll to the end of output
        except subprocess.TimeoutExpired:
            self.output_text.config(state="normal")
            self.output_text.insert(tk.END, "Error: Script timed out after 60 seconds.\n")
            self.output_text.config(state="disabled")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to run {script_path.name}: {e}")

    def clear_output(self):
        """Clear the output text box."""
        self.output_text.config(state="normal")
        self.output_text.delete(1.0, tk.END)
        self.output_text.config(state="disabled")

if __name__ == "__main__":
    root = tk.Tk()
    app = ScriptRunnerApp(root)
    root.mainloop()