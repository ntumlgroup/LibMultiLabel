import os
import sys
import glob
import re
from pathlib import Path
from collections import defaultdict

current_dir = os.path.dirname(os.path.abspath(__file__))
lib_path = os.path.abspath(os.path.join(current_dir, "..", ".."))
sys.path.insert(0, lib_path)


def classify_file_category(path):

    relative_path = Path(path).relative_to(lib_path)
    return_path = relative_path.as_posix()
    filename = Path(*relative_path.parts[1:]).as_posix() if len(relative_path.parts) > 1 else return_path

    if filename.startswith("linear"):
        category = "linear"
    elif filename.startswith("torch") or filename.startswith("nn"):
        category = "nn"
    else:
        category = "general"
    return category, return_path


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


def find_config_usages_in_file(file_path, allowed_keys):
    pattern = re.compile(r"\bconfig\.([a-zA-Z_][a-zA-Z0-9_]*)")
    detailed_results = {}
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except (IOError, UnicodeDecodeError):
        return []

    _, path = classify_file_category(file_path)

    if file_path.endswith("main.py"):
        for idx in range(len(lines)):
            if lines[idx].startswith("def main("):
                lines = lines[idx:]
                main_start = idx
                break
        for i, line in enumerate(lines[1:]):
            if line and line[0] not in (" ", "\t") and line.strip() != "":
                lines = lines[:i]
                break

    for i, line in enumerate(lines, start=1):
        matches = pattern.findall(line)
        for key in matches:
            if key in allowed_keys:
                if key not in detailed_results:
                    detailed_results[key] = {"file": path, "lines": []}
                if file_path.endswith("main.py"):
                    detailed_results[key]["lines"].append(str(i + main_start))
                else:
                    detailed_results[key]["lines"].append(str(i))

    return detailed_results


def move_duplicates_together(data, keep):
    all_keys = list(data.keys())
    duplicates = set()

    for i, key1 in enumerate(all_keys):
        for key2 in all_keys[i + 1 :]:
            duplicates |= data[key1] & data[key2]

    data[keep] |= duplicates

    for key in all_keys:
        if key != keep:
            data[key] -= duplicates

    return data


def classify(raw_flags):

    category_set = {"general": set(), "linear": set(), "nn": set()}
    flags = fetch_option_flags(raw_flags)
    allowed_keys = set(flag["instruction"] for flag in flags)
    file_set = fetch_all_files()
    usage_map = defaultdict(list)
    collected = {}

    for file_path in file_set:
        detailed_results = find_config_usages_in_file(file_path, allowed_keys)
        if detailed_results:
            usage_map[file_path] = set(detailed_results.keys())
            for k, v in detailed_results.items():
                if k not in collected:
                    collected[k] = []
                collected[k].append(v)

    for path, keys in usage_map.items():
        category, path = classify_file_category(path)
        category_set[category] = category_set[category].union(keys)

    category_set = move_duplicates_together(category_set, "general")

    for flag in flags:
        for k, v in category_set.items():
            for i in v:
                if flag["instruction"] == i:
                    flag["category"] = k
        if "category" not in flag:
            flag["category"] = "general"

    result = {}
    for flag in flags:
        if flag["category"] not in result:
            result[flag["category"]] = []

        result[flag["category"]].append(
            {"name": flag["name"].replace("--", r"\-\-"), "description": flag["description"]}
        )

    result["details"] = []
    for k, v in collected.items():
        result["details"].append({"name": k, "file": v[0]["file"], "location": ", ".join(v[0]["lines"])})
        if len(v) > 1:
            for i in v[1:]:
                result["details"].append({"name": "", "file": i["file"], "location": ", ".join(i["lines"])})

    return result
