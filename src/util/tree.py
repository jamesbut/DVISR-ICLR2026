# Tree utils

import copy


# Get parent of the final token in the list tokens
def get_parent(tokens):

    stack = []  # (token, operands_left)
    last_parent = None

    for i, token in enumerate(tokens):

        if token['type'] == 'const':

            stack[-1] = (stack[-1][0], stack[-1][1] - 1)
            while stack and stack[-1][1] == 0:
                stack.pop()
                if stack:
                    stack[-1] = (stack[-1][0], stack[-1][1] - 1)
            last_parent = stack[-1][0] if stack else None

        else:

            # Define arity based on the operator
            arity = 2 if token['type'] == 'bin_op' else 1
            stack.append((token, arity))

            # If operator is last token it is the parent
            if stack and i == len(tokens) - 1:
                last_parent = stack[-1][0]

    return last_parent


# Get sibling of the final token in the list tokens
def get_sibling(tokens):

    # Stack of tuples (token, arity, child)
    operators = []
    sibling = None

    def decr_op_arity(operators):
        if operators:
            operators[-1] = (operators[-1][0],
                             operators[-1][1] - 1,
                             operators[-1][2])

    for i, t in enumerate(tokens):

        if t['type'] == 'const':

            decr_op_arity(operators)

            # Pop operators off the stack as they are consumed
            while operators and operators[-1][1] == 0:
                operators.pop()
                decr_op_arity(operators)

            # Sibling will be child of last binary operator on the stack
            sibling = operators[-1][2] if operators else None

        else:

            # Define operator arity
            arity = 2 if t['type'] == 'bin_op' else 1

            # Determine operator child
            child = tokens[i + 1] if i < len(tokens) - 1 else None

            operators.append((t, arity, child))

            # Keep track of child of most recent binary operator pushed onto
            # the stack
            sibling = operators[-1][2] if arity == 2 else None

    return sibling


# Determines whether the next token in the sequence of tokens will be a
# descedent of any ops in ancestor_ops
def is_descendent(tokens, ancestor_ops):

    # TODO: Add check that the arity of all ancestor_ops must be 1 (un_op)

    dangling = 0
    threshold = None

    for t in tokens:

        if t['type'] == 'bin_op':
            dangling += 1
        if t['type'] == 'const':
            dangling -= 1

        # Turn "on" if ancestor op is found
        # Remain "on" until branch completes
        if threshold is None:
            for op in ancestor_ops:
                if t['op'] == op:
                    threshold = dangling - 1
                    break

        # Turn "off" once the branch completes
        else:
            if dangling == threshold:
                threshold = None

    return threshold is not None
