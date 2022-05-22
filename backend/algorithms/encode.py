import sys
import string
import random
import unicodedata

ALPHABET = list(string.ascii_lowercase) + [" ", "."]
LETTER_TO_IDX = dict(map(reversed, enumerate(ALPHABET)))


def clean_text(text):
    # try and approximate unicode with ascii
    text = unicodedata.normalize("NFKD", text).encode("ascii",
                                                      "ignore").decode()

    text = text.lower()  # make lowercase
    text = text.replace("?", ".").replace("!", ".")
    for c in "/-\n\r":
        text = text.replace(c, " ")
    text = "".join(filter(ALPHABET.__contains__,
                          text))  # filter to alphabet chars

    text = text.lstrip(" .")  # filter out leading spaces and periods
    if text == "":
        raise ValueError("text needs to have at least one letter")

    ret = ""
    for x in text:
        # ret is a valid string after every iteration
        if x == ".":
            ret = ret.rstrip(". ") + ". "
        elif x == " ":
            ret = ret.rstrip(" ") + " "
        else:
            ret += x

    ret = ret.rstrip(" ")  # strip trailing spaces
    return ret


def encode(plaintext):
    cipherbet = ALPHABET.copy()
    random.shuffle(cipherbet)

    ciphertext = "".join(cipherbet[LETTER_TO_IDX[c]] for c in plaintext)
    return ciphertext


def encode_with_breakpoint(plaintext):
    bpoint = random.randint(0, len(plaintext))
    print(f"Breakpoint at position {bpoint}")

    return encode(plaintext[:bpoint]) + encode(plaintext[bpoint:])
