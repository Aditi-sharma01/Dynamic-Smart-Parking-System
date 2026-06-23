from pathlib import Path

# Mapping:
# occupied -> occupied_slot (0)
# space-empty -> open_slot (1)
# space-occupied -> occupied_slot (0)
# unoccupied -> open_slot (1)

CLASS_MAP = {
    0: 0,  # occupied -> occupied_slot
    1: 1,  # space-empty -> open_slot
    2: 0,  # space-occupied -> occupied_slot
    3: 1   # unoccupied -> open_slot
}

label_dirs = [
    "parking_dataset/train/labels",
    "parking_dataset/valid/labels",
    "parking_dataset/test/labels"
]

for label_dir in label_dirs:
    label_dir = Path(label_dir)

    for txt_file in label_dir.glob("*.txt"):
        new_lines = []

        with open(txt_file, "r") as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()

            if len(parts) < 5:
                continue

            old_class = int(parts[0])

            if old_class in CLASS_MAP:
                parts[0] = str(CLASS_MAP[old_class])
                new_lines.append(" ".join(parts))

        with open(txt_file, "w") as f:
            f.write("\n".join(new_lines))

print("Class conversion completed!")