# Calculate median value and index from a list of unsorted values.
# When there are an even number of elements in the list prefer_lower defines
# which element is returned.
# Therefore, this is not the true median value because it is forced to choose
# a specific element in the list.
# reverse_sort specifies whether to reverse the sorting.
def median(items, reverse_sort=False, prefer_lower=False):

    if not items:
        return None

    n = len(items)

    # Sort items
    items = sorted([(val, idx) for idx, val in enumerate(items)],
                   reverse=reverse_sort)

    # Odd number of values
    if n % 2 == 1:

        # True middle
        return items[n // 2]

    # Even number of values
    else:

        # Choose based on preference
        return items[n // 2 - 1] if prefer_lower else items[n // 2]
