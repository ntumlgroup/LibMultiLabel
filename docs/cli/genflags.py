import sys
import os

sys.path.insert(1, os.path.join(sys.path[0], "..", ".."))

import main

from classifier import classify


class FakeParser(dict):
    def __init__(self):
        self.flags = []

    def add_argument(
        self, *args, type=None, default=None, help=None, action=None, choices=None, nargs=None, const=None
    ):
        flag = {"name": " ".join(list(args)).replace("--", r"\-\-")}

        help = help.replace("%(default)s", str(default))
        help = help.replace("%(const)s", str(const))
        if choices:
            flag["description"] = f'{help}. One of {{{", ".join(choices)}}}.'
        else:
            flag["description"] = f"{help}."

        self.flags.append(flag)


parser = FakeParser()
# Keep this line in sync with the same one in main.py:get_config()
parser.add_argument("-c", "--config", help="Path to configuration file")
main.add_all_arguments(parser)

classified = classify(parser.flags)


def width_title(key, title):
    return max(map(lambda f: len(f[key]), classified[title]))


def print_table(title, flags, intro):
    print()
    print(intro)
    print()

    wn = width_title("name", title)
    wd = width_title("description", title)

    print("=" * wn, "=" * wd)
    print("Name".ljust(wn), "Description".ljust(wd))
    print("=" * wn, "=" * wd)
    for flag in flags:
        print(flag["name"].ljust(wn), flag["description"].ljust(wd))
    print("=" * wn, "=" * wd)
    print()


print_table(
    "general",
    classified["general"],
    intro="**General options**:\n\
Common configurations shared across both linear and neural network trainers.",
)
print_table(
    "linear",
    classified["linear"],
    intro="**Linear options**:\n\
Configurations specific to linear trainer.",
)
print_table(
    "nn",
    classified["nn"],
    intro="**Neural network options**:\n\
Configurations specific to torch (neural networks) trainer.",
)
