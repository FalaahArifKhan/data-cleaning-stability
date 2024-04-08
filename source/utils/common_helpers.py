import hashlib


def generate_guid(ordered_hierarchy_lst: list):
    identifier = '|'.join([str(val) for val in ordered_hierarchy_lst])
    return hashlib.md5(identifier.encode()).hexdigest()
