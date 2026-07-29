import torch

from .constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX


def tokenizer_image_token(
    prompt,
    tokenizer,
    image_token_index=IMAGE_TOKEN_INDEX,
    return_tensors=None,
):
    prompt_chunks = [
        tokenizer(chunk).input_ids if chunk else []
        for chunk in prompt.split(DEFAULT_IMAGE_TOKEN)
    ]

    def insert_separator(chunks, separator):
        return [
            element
            for chunk_and_separator in zip(
                chunks,
                [separator] * len(chunks),
            )
            for element in chunk_and_separator
        ][:-1]

    input_ids = []
    offset = 0
    if (
        prompt_chunks
        and prompt_chunks[0]
        and prompt_chunks[0][0] == tokenizer.bos_token_id
    ):
        offset = 1
        input_ids.append(prompt_chunks[0][0])

    for chunk in insert_separator(
        prompt_chunks,
        [image_token_index] * (offset + 1),
    ):
        input_ids.extend(chunk[offset:])

    if return_tensors is None:
        return input_ids
    if return_tensors == "pt":
        return torch.tensor(input_ids, dtype=torch.long)
    raise ValueError(f"Unsupported tensor type: {return_tensors}")
