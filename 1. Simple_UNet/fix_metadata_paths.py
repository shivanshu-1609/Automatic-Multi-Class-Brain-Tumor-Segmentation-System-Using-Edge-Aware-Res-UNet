 
import csv
import os
import shutil

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DIR = os.path.join(BASE_DIR, "processed_brats_2d")
METADATA_CSV = os.path.join(PROCESSED_DIR, "metadata.csv")
BACKUP_CSV = METADATA_CSV + ".bak"

# Step 1: Peek at the first row to detect the OLD root automatically

with open(METADATA_CSV, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    rows = list(reader)
    fieldnames = reader.fieldnames

if not rows:
    print("metadata.csv is empty — nothing to fix.")
    exit(0)

sample_image_path = rows[0].get("image_path", "")
print(f"Sample path in CSV : {sample_image_path}")
print(f"Current project dir: {PROCESSED_DIR}")


# Step 2: Detect old root vs new root

# The new correct parent of processed_brats_2d is BASE_DIR.
# Extract the old parent from the sample path.
marker = "processed_brats_2d"
if marker not in sample_image_path:
    print(f"ERROR: Could not find '{marker}' in sample path. Is the CSV already correct?")
    exit(1)

# Everything before "processed_brats_2d" in the sample is the OLD root.
old_root = sample_image_path[: sample_image_path.index(marker)]   # e.g. "C:\...\Modified_Unet_AISS\"
new_root = PROCESSED_DIR + os.sep                                  # e.g. "C:\...\Modified_Unet_ATISS\processed_brats_2d\"

# Build the full old prefix that includes the marker directory
old_prefix = old_root + marker + os.sep
new_prefix = PROCESSED_DIR + os.sep

if old_prefix == new_prefix:
    print("Paths are already correct — no changes needed.")
    exit(0)

print(f"\nOLD prefix: {old_prefix}")
print(f"NEW prefix: {new_prefix}")


# Step 3: Back up and rewrite

shutil.copy2(METADATA_CSV, BACKUP_CSV)
print(f"\nBacked up original to: {BACKUP_CSV}")

fixed_rows = []
for row in rows:
    for field in ("image_path", "mask_path"):
        if field in row and row[field]:
            row[field] = row[field].replace(old_prefix, new_prefix)
    fixed_rows.append(row)

with open(METADATA_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(fixed_rows)

# Step 4: Verify a few fixed paths actually exist

print("\nVerifying first 5 fixed paths...")
ok = 0
fail = 0
for row in fixed_rows[:5]:
    p = row.get("image_path", "")
    exists = os.path.exists(p)
    status = "OK" if exists else "MISSING"
    print(f"  [{status}] {p}")
    if exists:
        ok += 1
    else:
        fail += 1

print(f"\nFixed {len(fixed_rows)} rows. Verified: {ok} OK, {fail} missing.")
if fail == 0:
    print("\nAll paths resolved correctly. You can now run test_brats.py.")
else:
    print("\nWARNING: Some paths are still missing. Check that processed_brats_2d is intact.")
