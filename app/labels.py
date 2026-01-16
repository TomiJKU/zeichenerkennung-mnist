def label_to_char(label: int) -> str:
    # 0-9
    if 0 <= label <= 9:
        return str(label)
    # 10-35: A-Z
    if 10 <= label <= 35:
        return chr(ord("A") + (label - 10))
    # 36-61: a-z
    if 36 <= label <= 61:
        return chr(ord("a") + (label - 36))
    return "?"


def char_to_label(s: str) -> int:
    s = s.strip()
    if len(s) != 1:
        raise ValueError("Bitte genau ein Zeichen eingeben (0-9, A-Z, a-z).")

    c = s[0]
    if "0" <= c <= "9":
        return int(c)
    if "A" <= c <= "Z":
        return 10 + (ord(c) - ord("A"))
    if "a" <= c <= "z":
        return 36 + (ord(c) - ord("a"))

    raise ValueError("Erlaubt sind: 0-9, A-Z, a-z")
