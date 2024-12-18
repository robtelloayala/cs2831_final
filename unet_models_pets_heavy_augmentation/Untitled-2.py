#%%
import pandas as pd
import os
import glob

# Specify the directory containing your CSV files
script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()

# Set CSV_DIR to the script directory
CSV_DIR = script_dir
# Initialize a list to store the results
results = []

# Use glob to find all CSV files in the directory
csv_files = glob.glob(os.path.join(CSV_DIR, "*.csv"))

# Iterate over each CSV file
for csv_file in csv_files:
    # Extract the base name without extension for display
    csv_name = os.path.splitext(os.path.basename(csv_file))[0]
    
    try:
        # Read the CSV file
        df = pd.read_csv(csv_file)
        
        # Check if required columns exist
        if 'val_loss' not in df.columns or 'val_accuracy' not in df.columns:
            print(f"Skipping {csv_file}: Required columns not found.")
            continue
        
        # Find the row with the minimum val_loss
        min_loss_row = df.loc[df['val_loss'].idxmin()]
        
        # Extract val_loss and val_accuracy
        val_loss = min_loss_row['val_loss']
        val_accuracy = min_loss_row['val_accuracy']
        
        # Append the result as a dictionary
        results.append({
            'Model': csv_name,
            'Val Loss': val_loss,
            'Val Accuracy': val_accuracy
        })
    
    except Exception as e:
        print(f"Error processing {csv_file}: {e}")

# Create a DataFrame from the results
results_df = pd.DataFrame(results)

# Optionally, sort the DataFrame by Val Loss
results_df = results_df.sort_values(by='Val Loss').reset_index(drop=True)

# Display the DataFrame (optional)
print(results_df)
