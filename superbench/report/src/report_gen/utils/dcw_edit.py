def remove_prefix_case_insensitive(s, prefix):
    if s.lower().startswith(prefix.lower()):
        return s[len(prefix):]
    return s