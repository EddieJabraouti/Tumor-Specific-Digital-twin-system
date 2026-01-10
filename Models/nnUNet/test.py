from pathlib import Path

data_path = "/Users/eddiejabrouti/.cache/kagglehub/datasets/awsaf49/brats2020-training-data/versions/3"

# Check the nested path
nested_path = Path(data_path) / "BraTS2020_training_data" / "content" / "data"

print("=" * 60)
print(f"Contents of: {nested_path}")
print("=" * 60)

if nested_path.exists():
    items = list(nested_path.iterdir())
    print(f"Found {len(items)} items\n")
    
    for item in sorted(items)[:10]:
        if item.is_dir():
            print(f"📁 {item.name}/")
            # Show what's inside patient folders
            subitems = list(item.iterdir())[:3]
            for subitem in subitems:
                print(f"   📄 {subitem.name}")
            if len(list(item.iterdir())) > 3:
                print(f"   ... and {len(list(item.iterdir())) - 3} more files")
        else:
            print(f"📄 {item.name}")
    
    if len(items) > 10:
        print(f"\n... and {len(items) - 10} more items")
else:
    print("Directory does not exist!")
    
# Also try to find any patient folders
print("\n" + "=" * 60)
print("Searching for BraTS patient folders:")
print("=" * 60)

patient_folders = list(Path(data_path).rglob("BraTS*_Training_*"))
if patient_folders:
    print(f"Found {len(patient_folders)} patient folders")
    for folder in patient_folders[:5]:
        print(f"  {folder.relative_to(data_path)}")
else:
    print("No patient folders found!")

