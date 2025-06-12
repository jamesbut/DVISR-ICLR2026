# Check whether string is a representation of a float
def is_float(string):
    try:
        float(string)
        return True
    except ValueError:
        return False
