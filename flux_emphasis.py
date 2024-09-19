from typing import TYPE_CHECKING, Optional
from pydash import flatten

import torch
from transformers.models.clip.tokenization_clip import CLIPTokenizer
from einops import repeat

if TYPE_CHECKING:
    from flux_pipeline import FluxPipeline


def parse_prompt_attention(text):
    """
    Parses a string with attention tokens and returns a list of pairs: text and its associated weight.
    Accepted tokens are:
      (abc) - increases attention to abc by a multiplier of 1.1
      (abc:3.12) - increases attention to abc by a multiplier of 3.12
      [abc] - decreases attention to abc by a multiplier of 1.1
      \\( - literal character '('
      \\[ - literal character '['
      \\) - literal character ')'
      \\] - literal character ']'
      \\ - literal character '\'
      anything else - just text

    >>> parse_prompt_attention('normal text')
    [['normal text', 1.0]]
    >>> parse_prompt_attention('an (important) word')
    [['an ', 1.0], ['important', 1.1], [' word', 1.0]]
    >>> parse_prompt_attention('(unbalanced')
    [['unbalanced', 1.1]]
    >>> parse_prompt_attention('\\(literal\\]')
    [['(literal]', 1.0]]
    >>> parse_prompt_attention('(unnecessary)(parens)')
    [['unnecessaryparens', 1.1]]
    >>> parse_prompt_attention('a (((house:1.3)) [on] a (hill:0.5), sun, (((sky))).')
    [['a ', 1.0],
     ['house', 1.5730000000000004],
     [' ', 1.1],
     ['on', 1.0],
     [' a ', 1.1],
     ['hill', 0.55],
     [', sun, ', 1.1],
     ['sky', 1.4641000000000006],
     ['.', 1.1]]
    """
    import re

    re_attention = re.compile(
        r"""
            \\\(|\\\)|\\\[|\\]|\\\\|\\|\(|\[|:([+-]?[.\d]+)\)|
            \)|]|[^\\()\[\]:]+|:
        """,
        re.X,
    )

    re_break = re.compile(r"\s*\bBREAK\b\s*", re.S)

    res = []
    round_brackets = []
    square_brackets = []

    round_bracket_multiplier = 1.1
    square_bracket_multiplier = 1 / 1.1

    def multiply_range(start_position, multiplier):
        for p in range(start_position, len(res)):
            res[p][1] *= multiplier

    for m in re_attention.finditer(text):
        text = m.group(0)
        weight = m.group(1)

        if text.startswith("\\"):
            res.append([text[1:], 1.0])
        elif text == "(":
            round_brackets.append(len(res))
        elif text == "[":
            square_brackets.append(len(res))
        elif weight is not None and len(round_brackets) > 0:
            multiply_range(round_brackets.pop(), float(weight))
        elif text == ")" and len(round_brackets) > 0:
            multiply_range(round_brackets.pop(), round_bracket_multiplier)
        elif text == "]" and len(square_brackets) > 0:
            multiply_range(square_brackets.pop(), square_bracket_multiplier)
        else:
            parts = re.split(re_break, text)
            for i, part in enumerate(parts):
                if i > 0:
                    res.append(["BREAK", -1])
                res.append([part, 1.0])

    for pos in round_brackets:
        multiply_range(pos, round_bracket_multiplier)

    for pos in square_brackets:
        multiply_range(pos, square_bracket_multiplier)

    if len(res) == 0:
        res = [["", 1.0]]

    # merge runs of identical weights
    i = 0
    while i + 1 < len(res):
        if res[i][1] == res[i + 1][1]:
            res[i][0] += res[i + 1][0]
            res.pop(i + 1)
        else:
            i += 1

    return res


def get_prompts_tokens_with_weights(
    clip_tokenizer: CLIPTokenizer, prompt: str, debug: bool = False
):
    """
    Get prompt token ids and weights, this function works for both prompt and negative prompt

    Args:
        pipe (CLIPTokenizer)
            A CLIPTokenizer
        prompt (str)
            A prompt string with weights

    Returns:
        text_tokens (list)
            A list contains token ids
        text_weight (list)
            A list contains the correspodent weight of token ids

    Example:
        import torch
        from transformers import CLIPTokenizer

        clip_tokenizer = CLIPTokenizer.from_pretrained(
            "stablediffusionapi/deliberate-v2"
            , subfolder = "tokenizer"
            , dtype = torch.float16
        )

        token_id_list, token_weight_list = get_prompts_tokens_with_weights(
            clip_tokenizer = clip_tokenizer
            ,prompt = "a (red:1.5) cat"*70
        )
    """
    texts_and_weights = parse_prompt_attention(prompt)
    text_tokens, text_weights = [], []
    maxlen = clip_tokenizer.model_max_length
    for word, weight in texts_and_weights:
        # tokenize and discard the starting and the ending token
        token = clip_tokenizer(
            word, truncation=False, padding=False, add_special_tokens=False
        ).input_ids
        # so that tokenize whatever length prompt
        # the returned token is a 1d list: [320, 1125, 539, 320]
        if debug:
            print(
                token,
                "|FOR MODEL LEN{}|".format(maxlen),
                clip_tokenizer.decode(
                    token, skip_special_tokens=True, clean_up_tokenization_spaces=True
                ),
            )
        # merge the new tokens to the all tokens holder: text_tokens
        text_tokens = [*text_tokens, *token]

        # each token chunk will come with one weight, like ['red cat', 2.0]
        # need to expand weight for each token.
        chunk_weights = [weight] * len(token)

        # append the weight back to the weight holder: text_weights
        text_weights = [*text_weights, *chunk_weights]
    return text_tokens, text_weights


def group_tokens_and_weights(
    token_ids: list,
    weights: list,
    pad_last_block=False,
    bos=49406,
    eos=49407,
    max_length=77,
    pad_tokens=True,
):
    """
    Produce tokens and weights in groups and pad the missing tokens

    Args:
        token_ids (list)
            The token ids from tokenizer
        weights (list)
            The weights list from function get_prompts_tokens_with_weights
        pad_last_block (bool)
            Control if fill the last token list to 75 tokens with eos
    Returns:
        new_token_ids (2d list)
        new_weights (2d list)

    Example:
        token_groups,weight_groups = group_tokens_and_weights(
            token_ids = token_id_list
            , weights = token_weight_list
        )
    """
    max_len = max_length - 2 if max_length < 77 else max_length
    # this will be a 2d list
    new_token_ids = []
    new_weights = []
    while len(token_ids) >= max_len:
        # get the first 75 tokens
        head_75_tokens = [token_ids.pop(0) for _ in range(max_len)]
        head_75_weights = [weights.pop(0) for _ in range(max_len)]

        # extract token ids and weights

        if pad_tokens:
            if bos is not None:
                temp_77_token_ids = [bos] + head_75_tokens + [eos]
                temp_77_weights = [1.0] + head_75_weights + [1.0]
            else:
                temp_77_token_ids = head_75_tokens + [eos]
                temp_77_weights = head_75_weights + [1.0]

        # add 77 token and weights chunk to the holder list
        new_token_ids.append(temp_77_token_ids)
        new_weights.append(temp_77_weights)

    # padding the left
    if len(token_ids) > 0:
        if pad_tokens:
            padding_len = max_len - len(token_ids) if pad_last_block else 0

            temp_77_token_ids = [bos] + token_ids + [eos] * padding_len + [eos]
            new_token_ids.append(temp_77_token_ids)

            temp_77_weights = [1.0] + weights + [1.0] * padding_len + [1.0]
            new_weights.append(temp_77_weights)
        else:
            new_token_ids.append(token_ids)
            new_weights.append(weights)
    return new_token_ids, new_weights


def standardize_tensor(
    input_tensor: torch.Tensor, target_mean: float, target_std: float
) -> torch.Tensor:
    """
    This function standardizes an input tensor so that it has a specific mean and standard deviation.

    Parameters:
    input_tensor (torch.Tensor): The tensor to standardize.
    target_mean (float): The target mean for the tensor.
    target_std (float): The target standard deviation for the tensor.

    Returns:
    torch.Tensor: The standardized tensor.
    """

    # First, compute the mean and std of the input tensor
    mean = input_tensor.mean()
    std = input_tensor.std()

    # Then, standardize the tensor to have a mean of 0 and std of 1
    standardized_tensor = (input_tensor - mean) / std

    # Finally, scale the tensor to the target mean and std
    output_tensor = standardized_tensor * target_std + target_mean

    return output_tensor


def apply_weights(
    prompt_tokens: torch.Tensor,
    weight_tensor: torch.Tensor,
    token_embedding: torch.Tensor,
    eos_token_id: int,
    pad_last_block: bool = True,
) -> torch.FloatTensor:
    mean = token_embedding.mean()
    std = token_embedding.std()
    if pad_last_block:
        pooled_tensor = token_embedding[
            torch.arange(token_embedding.shape[0], device=token_embedding.device),
            (
                prompt_tokens.to(dtype=torch.int, device=token_embedding.device)
                == eos_token_id
            )
            .int()
            .argmax(dim=-1),
        ]
    else:
        pooled_tensor = token_embedding[:, -1]

    for j in range(len(weight_tensor)):
        if weight_tensor[j] != 1.0:
            token_embedding[:, j] = (
                pooled_tensor
                + (token_embedding[:, j] - pooled_tensor) * weight_tensor[j]
            )
    return standardize_tensor(token_embedding, mean, std)


@torch.inference_mode()
def get_weighted_text_embeddings_flux(
    pipe: "FluxPipeline",
    prompt: str = "",
    num_images_per_prompt: int = 1,
    device: Optional[torch.device] = None,
    target_device: Optional[torch.device] = torch.device("cuda:0"),
    target_dtype: Optional[torch.dtype] = torch.bfloat16,
    debug: bool = False,
):
    """
    This function can process long prompt with weights, no length limitation
    for Stable Diffusion XL

    Args:
        pipe (StableDiffusionPipeline)
        prompt (str)
        prompt_2 (str)
        neg_prompt (str)
        neg_prompt_2 (str)
        num_images_per_prompt (int)
        device (torch.device)
    Returns:
        prompt_embeds (torch.Tensor)
        neg_prompt_embeds (torch.Tensor)
    """
    device = device or pipe._execution_device

    eos = pipe.clip.tokenizer.eos_token_id
    eos_2 = pipe.t5.tokenizer.eos_token_id
    bos = pipe.clip.tokenizer.bos_token_id
    bos_2 = pipe.t5.tokenizer.bos_token_id

    clip = pipe.clip.hf_module
    t5 = pipe.t5.hf_module

    tokenizer_clip = pipe.clip.tokenizer
    tokenizer_t5 = pipe.t5.tokenizer

    t5_length = 512 if pipe.name == "flux-dev" else 256
    clip_length = 77

    # tokenizer 1
    prompt_tokens_clip, prompt_weights_clip = get_prompts_tokens_with_weights(
        tokenizer_clip, prompt, debug=debug
    )

    # tokenizer 2
    prompt_tokens_t5, prompt_weights_t5 = get_prompts_tokens_with_weights(
        tokenizer_t5, prompt, debug=debug
    )

    prompt_tokens_clip_grouped, prompt_weights_clip_grouped = group_tokens_and_weights(
        prompt_tokens_clip,
        prompt_weights_clip,
        pad_last_block=True,
        bos=bos,
        eos=eos,
        max_length=clip_length,
    )
    prompt_tokens_t5_grouped, prompt_weights_t5_grouped = group_tokens_and_weights(
        prompt_tokens_t5,
        prompt_weights_t5,
        pad_last_block=True,
        bos=bos_2,
        eos=eos_2,
        max_length=t5_length,
        pad_tokens=False,
    )
    prompt_tokens_t5 = flatten(prompt_tokens_t5_grouped)
    prompt_weights_t5 = flatten(prompt_weights_t5_grouped)
    prompt_tokens_clip = flatten(prompt_tokens_clip_grouped)
    prompt_weights_clip = flatten(prompt_weights_clip_grouped)

    prompt_tokens_clip = tokenizer_clip.decode(
        prompt_tokens_clip, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )
    prompt_tokens_clip = tokenizer_clip(
        prompt_tokens_clip,
        add_special_tokens=True,
        padding="max_length",
        truncation=True,
        max_length=clip_length,
        return_tensors="pt",
    ).input_ids.to(device)
    prompt_tokens_t5 = tokenizer_t5.decode(
        prompt_tokens_t5, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )
    prompt_tokens_t5 = tokenizer_t5(
        prompt_tokens_t5,
        add_special_tokens=True,
        padding="max_length",
        truncation=True,
        max_length=t5_length,
        return_tensors="pt",
    ).input_ids.to(device)

    prompt_weights_t5 = torch.cat(
        [
            torch.tensor(prompt_weights_t5, dtype=torch.float32),
            torch.full(
                (t5_length - torch.tensor(prompt_weights_t5).numel(),),
                1.0,
                dtype=torch.float32,
            ),
        ],
        dim=0,
    ).to(device)

    clip_embeds = clip(
        prompt_tokens_clip, output_hidden_states=True, attention_mask=None
    )["pooler_output"]
    if clip_embeds.shape[0] == 1 and num_images_per_prompt > 1:
        clip_embeds = repeat(clip_embeds, "1 ... -> bs ...", bs=num_images_per_prompt)

    weight_tensor_t5 = torch.tensor(
        flatten(prompt_weights_t5), dtype=torch.float32, device=device
    )
    t5_embeds = t5(prompt_tokens_t5, output_hidden_states=True, attention_mask=None)[
        "last_hidden_state"
    ]
    t5_embeds = apply_weights(prompt_tokens_t5, weight_tensor_t5, t5_embeds, eos_2)
    if debug:
        print(t5_embeds.shape)
    if t5_embeds.shape[0] == 1 and num_images_per_prompt > 1:
        t5_embeds = repeat(t5_embeds, "1 ... -> bs ...", bs=num_images_per_prompt)
    txt_ids = torch.zeros(
        num_images_per_prompt,
        t5_embeds.shape[1],
        3,
        device=target_device,
        dtype=target_dtype,
    )
    t5_embeds = t5_embeds.to(target_device, dtype=target_dtype)
    clip_embeds = clip_embeds.to(target_device, dtype=target_dtype)

    return (
        clip_embeds,
        t5_embeds,
        txt_ids,
    )
