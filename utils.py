from collections import namedtuple
correct_diacs = {
    "ş": "ș",
    "Ş": "Ș",
    "ţ": "ț",
    "Ţ": "Ț",
}

def clean_diacs(s: str):
    return "".join([correct_diacs[c] if c in correct_diacs else c for c in list(s)])