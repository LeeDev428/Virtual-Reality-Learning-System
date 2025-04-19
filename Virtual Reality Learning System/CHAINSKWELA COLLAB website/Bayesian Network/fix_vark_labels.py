import pandas as pd

# Load your dataset
df = pd.read_csv('VARK_output1.csv')

# Style order
style_map = ['Visual', 'Aural', 'Read/Write', 'Kinesthetic']

def fix_label(row):
    counts = [row['Count_V'], row['Count_A'], row['Count_R'], row['Count_K']]
    max_val = max(counts)
    # Find all indexes with the max value
    max_idxs = [i for i, v in enumerate(counts) if v == max_val]
    # Assign only the dominant styles (multi-label only if tied)
    return ','.join([style_map[i] for i in max_idxs])

# Apply the fix to every row
df['Label Learning Style'] = df.apply(fix_label, axis=1)

# Save the fixed dataset (overwrite or as new file)
df.to_csv('VARK_output1_fixed.csv', index=False)
print("Fixed dataset saved as VARK_output1_fixed.csv")
