# %%
import re
import string
import warnings
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import torch
import torch.utils.data as data
from datasets import load_dataset
from tqdm import tqdm


def format_stories(stories: List[str]):
    return "\n\n".join([f"{'='*10}\n{story}\n{'='*10}" for story in stories])


def format_stories_with_steer_coeffs(steered_stories: Dict[float, List[str]]):
    combined_str = ""
    for coeff, steered_texts in steered_stories.items():
        combined_str += f"\n\n===coeff={coeff:.2f}===\n"
        combined_steered_text = "\n\n======\n\n".join(steered_texts)
        combined_str += combined_steered_text + "\n"
    return combined_str


def strip_all_padding(tokenizer, strings: List[str]) -> List[str]:
    # strip off padding tokens from the beginning of the generated text
    stripped_strs = []
    for s in strings:
        padding_token = tokenizer.pad_token
        while s.startswith(f"{padding_token}"):
            s = s[len(f"{padding_token}") :]  # Remove one EOT
        stripped_strs.append(s)
    return stripped_strs


def remove_punctuation_and_strip(s: str):
    # Remove outer whitespace and punctuation
    return s.translate(str.maketrans("", "", string.punctuation)).strip()


def get_words(story: str):
    story_words = [
        remove_punctuation_and_strip(story_word) for story_word in story.lower().split()
    ]
    return story_words


def count_target_words_in_story(story: str, target_keywords: List[str]) -> int:
    search_words = set(
        remove_punctuation_and_strip(word.lower()) for word in target_keywords
    )
    total_count = sum([word in search_words for word in get_words(story)])
    return total_count


def steered_stories_without_prompts(
    steered_stories: Dict[float, List[str]], prompts: List[str]
) -> Dict[float, List[str]]:
    return {c: stories_without_prompt(s, prompts) for c, s in steered_stories.items()}


def truncate_stories(stories, tokenizer, max_tokens: int) -> List[str]:
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        warnings.warn("Tokenizer has no pad token. Setting pad token to eos token.")

    tokens = tokenizer(
        stories,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_tokens,
    )
    decoded = tokenizer.batch_decode(tokens["input_ids"], skip_special_tokens=True)
    return decoded


def simple_linear_regression(x: List, y: List):
    x_mat = np.stack((np.ones_like(x), x)).T
    return np.linalg.solve(x_mat.T @ x_mat, x_mat.T @ y)


def estimate_truncation_length(
    sample_stories,
    tokenizer,
    max_tokens: int,
) -> int:
    lists_of_tokens = tokenizer(sample_stories)["input_ids"]  # type: ignore
    char_lens = [len(story) for story in sample_stories]
    tok_lens = [len(tokens) for tokens in lists_of_tokens]
    intercept, slope = simple_linear_regression(char_lens, tok_lens)

    num_characters = int((max_tokens - intercept) / slope)
    return num_characters


def truncate_stories_approximate(
    stories,
    tokenizer,
    max_tokens: int,
    num_sample_stories: int,
    num_characters_buffer: int = 0,
) -> List[str]:
    """A much faster implementation which may not truncate perfectly."""
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        warnings.warn("Tokenizer has no pad token. Setting pad token to eos token.")

    num_characters = estimate_truncation_length(
        stories[:num_sample_stories], tokenizer, max_tokens
    )

    truncate_stories = [
        story[: num_characters + num_characters_buffer] for story in stories
    ]
    return truncate_stories


def truncate_stories_by_chars(stories: list[str], max_character_len: int):
    return [story[:max_character_len] for story in stories]


def avg_target_word_count_in_steered_stories(
    steered_stories: Dict[float, List[str]], target_words: List[str]
) -> Dict[float, float]:
    """
    Returns:
        Dict[float, float]: A dictionary mapping steering coefficients to the average
        number of target words per steered story.
    """
    coeff_to_avg_count = {}
    for coeff, steered_texts in steered_stories.items():
        count = sum(
            [count_target_words_in_story(t, target_words) for t in steered_texts]
        )
        coeff_to_avg_count[coeff] = count / len(steered_texts)
    return coeff_to_avg_count


def split_stories_by_concept(
    stories: List[str],
    target_words,
    verbose: bool = False,
) -> Tuple[List[str], List[str]]:
    for w in target_words:
        if w.startswith(" ") or w.endswith(" "):
            warnings.warn("Stripping whitespace from target words for story splitting.")
    target_words = [w.strip() for w in target_words]
    pattern = re.compile(
        r"\b(" + "|".join(re.escape(word) for word in target_words) + r")\b",
        re.IGNORECASE,
    )

    concept_stories = []
    other_stories = []
    for story_idx, story in tqdm(enumerate(stories), total=len(stories)):
        assert isinstance(story, str), f"`story` should be a string, got {type(story)}"
        if pattern.search(story):
            if verbose:
                print(f"\nStory {story_idx}")
                for snippet in target_word_snippets(story, target_words):
                    print(f'"{snippet}"')
            concept_stories.append(story)
        else:
            other_stories.append(story)
    return concept_stories, other_stories


def split_and_label_stories_by_concept(
    stories: List[str],
    target_words,
    verbose: bool = False,
) -> Tuple[List[tuple], List[tuple]]:
    concept, other = split_stories_by_concept(stories, target_words, verbose=verbose)
    return [(concept, 0) for concept in concept], [(other, 1) for other in other]


def load_dataset_with_split(
    dataset_name: str,
    split: str,
    max_stories: Optional[int],
) -> list[str]:
    split = f"{split}[:{max_stories}]" if max_stories else split
    dataset = load_dataset(dataset_name, split=split)
    stories = dataset["text" if "text" in dataset.column_names else "story"]  # type: ignore
    return list(stories)  # type: ignore


def tokenize_batch(
    batch: list[str],
    tokenizer,
    prepend_bos: bool,
    truncate_at: int,
    padding_side: Literal["left", "right"],
    device: torch.device,
    apply_chat_template: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns `input_ids`, `attention_mask`
    """
    tokenizer.padding_side = padding_side
    if prepend_bos:
        assert not apply_chat_template, "Don't prepend BOS when applying chat template."
        batch = [tokenizer.bos_token + story for story in batch]
    if apply_chat_template:
        chats = []
        for prompt in batch:
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ]
            chats.append(messages)
        tokens = tokenizer.apply_chat_template(
            chats,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=truncate_at,
        ).to(device)
    else:
        tokens = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=truncate_at,
        ).to(device)
    return tokens["input_ids"], tokens["attention_mask"]


def concat_with_porportion(
    concept_data: List[str],
    other_data: List[str],
    fraction_of_concept: float,
    num_to_return: int,
) -> list[str]:
    concept_count = int(fraction_of_concept * num_to_return)
    other_count = num_to_return - concept_count
    selected_concept_data = (concept_data * (concept_count // len(concept_data) + 1))[
        :concept_count
    ]
    selected_other_data = (other_data * (other_count // len(other_data) + 1))[
        :other_count
    ]
    selected_data = selected_concept_data + selected_other_data
    print(f"Total number of data: {len(selected_data)}")
    print(f"Number of concept data: {len(selected_concept_data)}")
    print(f"Number of other data: {len(selected_other_data)}")
    return selected_data


def truncate_and_split_concept_data(
    data: list[str],
    tokenizer,
    concept_words: List[str],
    max_tokens: int,
    use_approximate_truncation: bool,
    verbose: bool = False,
) -> Tuple[List[str], List[str]]:
    warnings.warn(
        "This function is deprecated. Use `truncate_stories_by_chars` and `split_stories_by_concept` instead."
    )
    print("Truncating stories...", end="")
    if use_approximate_truncation and len(data) > 1000:
        data = truncate_stories_approximate(
            data,
            tokenizer,
            max_tokens=max_tokens,
            num_sample_stories=1000,
            num_characters_buffer=0,
        )
    else:
        data = truncate_stories(data, tokenizer, max_tokens=max_tokens)
    print("done.")
    print("Splitting stories by concept...", end="")
    concept_stories, normal_stories = split_stories_by_concept(
        stories=data,
        target_words=concept_words,
        verbose=verbose,
    )
    print("done.")
    return concept_stories, normal_stories


class ListDataset(data.Dataset):
    def __init__(self, data: list):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

