def time_to_seconds(t: str) -> int:
    parts = t.split(":")
    if len(parts) == 2:
        mins, secs = map(int, parts)
        return mins * 60 + secs
    return 0
