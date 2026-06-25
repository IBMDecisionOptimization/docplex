__author__ = 'couronne'


def find_first_diff_char(s1, s2):
    """
    returns the index of the first differing character between s1 and s2

    Args:
        s1: the first string
        s2: the second string

    Returns:
        a numerical index, the first for which the strings differ, or -1 if they are identical
    """
    i = 0
    len_s1 = len(s1)
    len_s2 = len(s2)

    while True:
        if i >= len_s1:
            if i >= len_s2:
                return -1
            else:
                return i
        elif i >= len_s2:
            if i >= len_s1:
                return -1
            else:
                return i
        else:
            c1 = s1[i]
            c2 = s2[i]
            if c1 != c2:
                return i
        i += 1
    return -1

def check_strings_identical(case, s1, s2):
    first_diff = find_first_diff_char(s1, s2)
    if first_diff >= 0:
        msg = "strings differ pos: {0}, s1={1}, s2={2}".format(first_diff, s1[first_diff:], s2[first_diff:])
        case.assertEqual(s1, s2, msg)

