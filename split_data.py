import os
import re
import shutil
from pathlib import Path
from typing import List

# === Configuration ===
DEFAULT_HOUR_SPLIT: int = 24
DEFAULT_SOURCE_FOLDER: str = "mycelium"
DEFAULT_OUTPUT_FOLDER: str = "mycelium_labeled"
DEFAULT_TESTS_TO_INCLUDE: List[int] = [1, 2, 3, 4, 5, 6]


def split_mycelium_by_hours(
    source_folder: str = DEFAULT_SOURCE_FOLDER,
    output_folder: str = DEFAULT_OUTPUT_FOLDER,
    tests_to_include: List[int] = DEFAULT_TESTS_TO_INCLUDE,
    hour_split: int = DEFAULT_HOUR_SPLIT
) -> None:
    # Clear output folder if it exists
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder, exist_ok=True)

    files: List[str] = [
        f for f in os.listdir(source_folder)
        if re.match(r'test\d+_h\d+_\d+\.jpg', f)
    ]

    file_hours: List[tuple[str, int]] = []

    # Extract test number and hour from each file and filter by test number
    for f in files:
        test_match = re.search(r'test(\d+)_h(\d+)_', f)
        if test_match:
            test_num: int = int(test_match.group(1))
            hour: int = int(test_match.group(2))
            if test_num in tests_to_include:
                file_hours.append((f, hour))

    if not file_hours:
        print("No files matched the specified tests.")
        return

    # Determine min and max hours to build buckets
    all_hours: List[int] = [h for _, h in file_hours]
    min_hour: int = min(all_hours)
    max_hour: int = max(all_hours)

    print(f"Splitting from hour {min_hour} to {max_hour} every {hour_split} hours")

    # Sort files into time buckets
    for f, hour in file_hours:
        bucket: int = (hour - min_hour) // hour_split
        bucket_path: str = os.path.join(output_folder, str(bucket))
        os.makedirs(bucket_path, exist_ok=True)
        shutil.copy2(os.path.join(source_folder, f), os.path.join(bucket_path, f))

    print("Done.")


split_mycelium_by_hours()