# Tree utils

import copy


# Get parent of the final token in the list tokens
def get_parent(tokens):

    stack = []  # (token, operands_left)
    last_parent = None

    for i, token in enumerate(tokens):

        if token['type'] == 'const':

            if stack:
                # Capture parent before reduction
                last_parent = stack[-1][0]

                stack[-1] = (stack[-1][0], stack[-1][1] - 1)
                while stack and stack[-1][1] == 0:
                    stack.pop()
                    if stack:
                        stack[-1] = (stack[-1][0], stack[-1][1] - 1)
        else:

            # Define arity based on the operator
            arity = 2 if token['type'] == 'bin_op' else 1
            stack.append((token, arity))

            # If operator is last token, parent is prior operator
            if stack and len(stack) > 1 and i == len(tokens) - 1:
                last_parent = stack[-2][0]

    return copy.deepcopy(last_parent)


# Get sibling of the final token in the list tokens
def get_sibling(tokens):

    if not tokens:
        return None

    # List of tuples (op, arity, index)
    operators = []
    child_map = {}

    def decr_op_arity(operators):
        if operators:
            operators[-1] = (operators[-1][0],
                             operators[-1][1] - 1,
                             operators[-1][2])

    # Build child map
    for i, t in enumerate(tokens):

        if t['type'] == 'const':

            # Parent is most recent operator
            child_map.setdefault(operators[-1][2], []).append(i)
            decr_op_arity(operators)

            # Pop operators off the stack as they are consumed
            while operators and operators[-1][1] == 0:

                # As operators are popped set parent as the previous operator
                # in the stack
                if len(operators) > 1:
                    child_map.setdefault(
                        operators[-2][2], []
                    ).append(operators[-1][2])

                operators.pop()
                decr_op_arity(operators)

        else:

            # Define arity based on the operator
            arity = 2 if t['type'] == 'bin_op' else 1
            operators.append((t['op'], arity, i))

            # If operator is last token, parent is prior operator
            if operators and len(operators) > 1 and i == len(tokens) - 1:
                child_map.setdefault(operators[-2][2], []).append(i)

    # Search through child map for sibling
    for parent, children in child_map.items():

        # Check whether final token shares list with another token, that
        # will be its sibling
        if len(tokens) - 1 in children:
            if len(children) == 2:
                index = children.index(len(tokens) - 1)
                return (copy.deepcopy(tokens[children[1]]) if index == 0 else
                        copy.deepcopy(tokens[children[0]]))

    return None
