import pandas as pd

# Paths to your dataset
captions_file = "./results.csv/results.csv"

# Load the CSV file
df = pd.read_csv(captions_file, delimiter='|')

# Ensure column names are stripped of whitespace
df.columns = df.columns.str.strip()

# Check for problematic captions
problematic_rows = []
for idx, caption in enumerate(df['comment']):
    if not isinstance(caption, str) or pd.isna(caption) or caption.strip() == "":
        print(f"Row {idx} has a problematic caption: {caption}")
        problematic_rows.append((idx, caption))

# Print problematic rows
if problematic_rows:
    print(f"Found {len(problematic_rows)} problematic captions:")
    for idx, caption in problematic_rows:
        print(f"Row {idx}: {caption}")
else:
    print("No problematic captions found.")

# Optionally, save problematic rows to a CSV file for further inspection
problematic_df = pd.DataFrame(problematic_rows, columns=["Row Index", "Caption"])
problematic_df.to_csv("problematic_captions.csv", index=False)
print("Problematic captions saved to 'problematic_captions.csv'.")