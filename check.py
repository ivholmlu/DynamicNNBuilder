import tkinter as tk
from tkinter import filedialog
import subprocess
import toml


# Function to generate and save the TOML file
def generate_toml():
    try:
        # Collecting data from the GUI
        settings = {
            "learning_rate": float(learning_rate_entry.get()),
            "batch_size": int(batch_size_entry.get()),
            "iterations": int(iterations_entry.get())
        }

        layers = []
        for layer_entries in layer_entries_list:
            layer = {
                "type": layer_entries[0].get(),
                "dim_in": int(layer_entries[1].get()),
                "dim_out": int(layer_entries[2].get()),
                "activation": layer_entries[3].get(),
                "rank": int(layer_entries[4].get())
            }
            layers.append(layer)

        # Creating the TOML structure
        toml_data = {
            "settings": settings,
            "layer": layers
        }

        # Saving the TOML file
        file_path = filedialog.asksaveasfilename(defaultextension=".toml")
        if file_path:
            with open(file_path, "w") as toml_file:
                toml.dump(toml_data, toml_file)
            
            # Running the command with the created file
            subprocess.run(["python", "main.py", "-f", file_path])
    except Exception as e:
        print("An error occurred:", e)

# Tkinter GUI setup
root = tk.Tk()
root.title("TOML File Generator")

# Entries for [settings]
tk.Label(root, text="Learning Rate").grid(row=0, column=0)
learning_rate_entry = tk.Entry(root)
learning_rate_entry.grid(row=0, column=1)

tk.Label(root, text="Batch Size").grid(row=1, column=0)
batch_size_entry = tk.Entry(root)
batch_size_entry.grid(row=1, column=1)

tk.Label(root, text="Iterations").grid(row=2, column=0)
iterations_entry = tk.Entry(root)
iterations_entry.grid(row=2, column=1)

# Entries for [[layer]]
layer_entries_list = []
for i in range(3): # Assuming 3 layers
    tk.Label(root, text=f"Layer {i+1} Type").grid(row=3+i*5, column=0)
    type_entry = tk.Entry(root)
    type_entry.grid(row=3+i*5, column=1)

    tk.Label(root, text=f"Layer {i+1} Dim In").grid(row=4+i*5, column=0)
    dim_in_entry = tk.Entry(root)
    dim_in_entry.grid(row=4+i*5, column=1)

    tk.Label(root, text=f"Layer {i+1} Dim Out").grid(row=5+i*5, column=0)
    dim_out_entry = tk.Entry(root)
    dim_out_entry.grid(row=5+i*5, column=1)

    tk.Label(root, text=f"Layer {i+1} Activation").grid(row=6+i*5, column=0)
    activation_entry = tk.Entry(root)
    activation_entry.grid(row=6+i*5, column=1)

    tk.Label(root, text=f"Layer {i+1} Rank").grid(row=7+i*5, column=0)
    rank_entry = tk.Entry(root)
    rank_entry.grid(row=7+i*5, column=1)

    layer_entries = (type_entry, dim_in_entry, dim_out_entry, activation_entry, rank_entry)
    layer_entries_list.append(layer_entries)

# Button to generate the TOML file
generate_button = tk.Button(root, text="Generate TOML File", command=generate_toml)
generate_button.grid(row=20, column=0, columnspan=2)

root.mainloop()
