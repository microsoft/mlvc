import torch


def extend_list(data_list, n, min_n):
    if min_n == 0:
        return []
    if not data_list:
        raise ValueError("data_list cannot be empty if min_n is nonzero")
    while len(data_list) < n:
        data_list.extend(data_list[:n - len(data_list)])
    return data_list


def find_prefix(input_ids, prefix):
    """
    input_ids: [B, N1], no start token
    prefix: [N2, ], no start token
    """
    len_prefix = prefix.shape[0]  # N2
    # Create all possible windows of len_prefix
    input_ids_unfold = input_ids.unfold(1, len_prefix, 1)
    # Check if all elements in the window match the sequence
    matches = (input_ids_unfold == prefix).all(dim=2)
    # Convert boolean matches to integers for argmax operation
    matches_int = matches.type(torch.int64)
    # Calculate indices for the first match, if any, otherwise set to -1
    indices = torch.where(
        matches.any(dim=1),
        matches_int.argmax(dim=1),
        torch.tensor(-1, dtype=torch.int64),
    )
    assert (indices >= 0).all(), "Some inputs do not contain prefix"
    return indices
