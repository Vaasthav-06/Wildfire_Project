import pandas as pd
import numpy as np
import os
from sklearn.utils import shuffle

# --- CONFIG ---
CHUNKS_DIR = 'data/chunks'
PROCESSED_DIR = 'data/processed'
os.makedirs(PROCESSED_DIR, exist_ok=True)

# 1. MERGE THE 3 PARTS
print("🔄 Merging enriched chunks...")
files = [f for f in os.listdir(CHUNKS_DIR) if f.startswith('enriched_part_')]
df_list = []

for f in files:
    try:
        # on_bad_lines='skip' ensures one bad row doesn't crash the script
        temp = pd.read_csv(os.path.join(CHUNKS_DIR, f), on_bad_lines='skip')
        df_list.append(temp)
        print(f"   -> Loaded {f}: {len(temp)} rows")
    except:
        print(f"   ⚠️ Could not load {f}")

if not df_list:
    print("❌ No enriched files found! Make sure Phase 2 is done.")
    exit()

fire_df = pd.concat(df_list, ignore_index=True)

# Clean rows where API failed (NaNs)
initial_len = len(fire_df)
fire_df.dropna(subset=['temp', 'humidity', 'wind'], inplace=True)
fire_df['fire_detected'] = 1 # Label = 1 (FIRE)

print(f"✅ Cleaned {initial_len - len(fire_df)} bad rows.")
print(f"🔥 Total Fire Samples: {len(fire_df)}")

# 2. GENERATE SYNTHETIC SAFE POINTS (The "0" Class)
# We shuffle the weather data to create "Fake" safe conditions at real locations.
# This gives us a perfectly balanced dataset without 100,000 extra API calls.
print("⚖️ Generating Synthetic Safe Points...")

safe_df = fire_df.copy()
# Shuffle weather columns independently
safe_df['temp'] = np.random.permutation(safe_df['temp'].values)
safe_df['humidity'] = np.random.permutation(safe_df['humidity'].values)
safe_df['wind'] = np.random.permutation(safe_df['wind'].values)
safe_df['fire_detected'] = 0 # Label = 0 (SAFE)

# 3. COMBINE & EXPORT
master_df = pd.concat([fire_df, safe_df], ignore_index=True)
master_df = shuffle(master_df, random_state=42)

OUT_FILE = f'{PROCESSED_DIR}/training_data_final.csv'
master_df.to_csv(OUT_FILE, index=False)

print(f"\n🎉 Dataset Ready for Training!")
print(f"   - Total Rows: {len(master_df)}")
print(f"   - Fire (1): {len(fire_df)}")
print(f"   - Safe (0): {len(safe_df)}")
print(f"💾 Saved to: {OUT_FILE}")