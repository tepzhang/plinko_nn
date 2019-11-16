def merge_dicts(a: dict, b: dict):
    d = {}
    for k, v in a.items():
        d[k] = v
    for k, v in b.items():
        d[k] = v
    return d
