import os
import sys
import glob
import re
from pathlib import Path
from collections import defaultdict

lib_path = Path.cwd().parent
sys.path.insert(0, str(lib_path))


def classify_file_category(path):

    relative_path = Path(path).relative_to(lib_path)
    filename = "/".join(relative_path.parts[1:]) or relative_path.as_posix()

    if filename.startswith("linear"):
        return "linear"
    if filename.startswith(("torch", "nn")):
        return "nn"
    return "general"


def fetch_option_flags(flags):
    flag_list = []

    for flag in flags:
        flag_list.append(
            {
                "name": flag["name"].replace("\\", ""),
                "instruction": flag["name"].split("-")[-1],
                "description": flag["description"],
            }
        )

    return flag_list


def fetch_all_files():
    main_files = [
        os.path.join(lib_path, "main.py"),
        os.path.join(lib_path, "linear_trainer.py"),
        os.path.join(lib_path, "torch_trainer.py"),
    ]
    lib_files = glob.glob(os.path.join(lib_path, "libmultilabel/**/*.py"), recursive=True)
    file_set = set(map(os.path.abspath, main_files + lib_files))
    return file_set


def find_config_usages_in_file(file_path, allowed_keys, category_set):
    pattern = re.compile(r"\bconfig\.([a-zA-Z_][a-zA-Z0-9_]*)")

    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    if file_path.endswith("main.py"):
        for idx in range(len(lines)):
            if lines[idx].startswith("def main("):
                lines = lines[idx:]
                break
    all_str = " ".join(lines)
    matches = set(pattern.findall(all_str)) & allowed_keys

    category = classify_file_category(file_path)
    for key in matches:
        category_set[category].add(key)


def move_duplicates_together(data):
    duplicates = (data["general"] & data["linear"]) | (data["general"] & data["nn"]) | (data["linear"] & data["nn"])
    data["general"].update(duplicates)
    data["linear"] -= duplicates
    data["nn"] -= duplicates

    return data


def classify(raw_flags):
    category_set = {"general": set(), "linear": set(), "nn": set()}

    flags = fetch_option_flags(raw_flags)
    allowed_keys = set(flag["instruction"] for flag in flags)
    file_set = fetch_all_files()

    for file_path in file_set:
        find_config_usages_in_file(file_path, allowed_keys, category_set)

    category_set = move_duplicates_together(category_set)

    result = defaultdict(list)
    for flag in raw_flags:
        instr = flag["name"].replace("\\", "").split("-")[-1]
        flag_name = flag["name"].replace("--", r"\-\-")

        matched = False
        for category, keys in category_set.items():
            if instr in keys:
                result[category].append({"name": flag_name, "description": flag["description"]})
                matched = True
                break

        if not matched:
            result["general"].append({"name": flag_name, "description": flag["description"]})

    return result
