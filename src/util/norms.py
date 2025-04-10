def normalise_value(value, v_min, v_max, norm_min=0.0, norm_max=1.0):
    """
    Normalizes a value to a range [norm_min, norm_max] given its original
    range [v_min, v_max].

    Args:
        value (float): The value to normalize.
        v_min (float): Minimum possible value of the input range.
        v_max (float): Maximum possible value of the input range.
        norm_min (float): Desired minimum of the normalized range.
        norm_max (float): Desired maximum of the normalized range.

    Returns:
        float: The normalized value.

    Raises:
        ValueError: If v_max <= v_min or norm_max <= norm_min.
    """
    # Check for valid ranges
    if v_max <= v_min:
        raise ValueError("v_max must be greater than v_min")
    if norm_max <= norm_min:
        raise ValueError("norm_max must be greater than norm_min")

    # Linear normalization formula
    normalized = norm_min + (value - v_min) * (norm_max - norm_min) / (v_max - v_min)
    return normalized
