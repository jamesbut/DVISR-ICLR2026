# Define a function via a string

import re


class Function():

    def __init__(self, func: str, delimiter = '`'):

        self._delimiter = delimiter
        self._orig_func = func

        # Replace variables with calls to kwargs
        self._kwargs_func = self._append_kwargs_to_func_str(func,
                                                            self._delimiter)

        scope = {}
        func_str = "import numpy as np\ndef func(**kwargs): return " \
                   + self._kwargs_func

        exec(func_str, scope)

        self._func = scope['func']

    # Get the number of distinct variables in the function
    def num_vars(self):

        # Extract all 'var' values
        matches = re.findall(r"kwargs\['(.*?)'\]", self._kwargs_func)
        # Count unique ones
        return len(set(matches))

    # Convert to sympy expression and return string
    def get_sympy(self):

        from sympy import sympify

        # Create string suitable for sympy
        f_str = str(self)

        # Remove kwargs substrings
        f_str = re.sub(r"kwargs\['(.*?)'\]", r"\1", f_str)

        expr = sympify(f_str)
        return str(expr)

    def __call__(self, **kwargs):
        return self._func(**kwargs)

    def _append_kwargs_to_func_str(self, func: str, delimiter: str):

        # Pattern to match content between delimiter
        pattern = r"" + delimiter + "(.*?)" + delimiter

        # Function to replace each match
        def replacement_function(match):

            # Extract the content inside quotes
            original_content = match.group(1)

            # Wrap the original content in kwargs['']
            return f"kwargs['{original_content}']"

        # Replace using the pattern and replacement function
        result = re.sub(pattern, replacement_function, func)

        return result

    def __dict__(self):

        d = {
            'delimiter': self._delimiter,
            'function': self._orig_func
        }

        return d

    def __str__(self):
        return self._kwargs_func


def surround_sub_strings_with_delimiters(x: str, sub_strings: list[str],
                                         delim: str = '`'):
    """
    Surrounds instances of the sub strings in the main string x with a
    delimiter.
    This is mainly to prepare a raw string to be provided to the above Fuction
    class.

    Parameters:
    x (str): The main string that will be modified.
    sub_strings (list[str]): List of substrings that delimiters should
        surround in x.
    delim (str): Delimiter that will surround the substrings.

    Returns:
    x: Modified x with delimiter surrounding all instances of substrings.
    """
    for sub in sub_strings:
        x = x.replace(sub, f'{delim}{sub}{delim}')
    return x
