"""
Delethink tracing demo:
- Runs chunked (bounded-context) rollouts with an sglang Engine
- Decodes traces, parses math answers from model outputs, and verifies correctness
- Reports average score and average trace length

NOTE: Logic is intentionally unchanged from the original—only comments/docstrings added.
"""

import argparse
import asyncio
import functools
from pprint import pprint
from typing import Any

import numpy as np
import sglang as sgl
from datasets import load_dataset
from math_verify import LatexExtractionConfig, LatexNormalizationConfig, parse, verify
from tqdm.asyncio import tqdm as tqdm_asyncio
from transformers import AutoConfig

# High-level user prompt template. The solver is expected to emit a final boxed answer.
PROMPT_TEMPLATE = "{problem}\n\nPlease reason step by step, and put your final answer within \\boxed{{}}."

# Parser for gold answers (ground truth). Favors boxed expressions; no fallback heuristics.
parse_gold_answer_fn = functools.partial(
    parse,
    extraction_config=[
        LatexExtractionConfig(
            LatexNormalizationConfig(
                basic_latex=True,
                boxed="all",
                malformed_operators=False,
                units=True,
            ),
            boxed_match_priority=0,
        )
    ],
    fallback_mode="no_fallback",
    extraction_mode=["first_match"],
)

# Parser for predicted answers (model outputs). Same normalization, with a short timeout.
parse_predicted_answer_fn = functools.partial(
    parse,
    extraction_config=[
        LatexExtractionConfig(
            LatexNormalizationConfig(
                basic_latex=True,
                boxed="all",
                malformed_operators=False,
                units=True,
            ),
            boxed_match_priority=0,
        )
    ],
    parsing_timeout=1,
    fallback_mode="no_fallback",
    extraction_mode=["first_match"],
)


def compute_score(
    *,
    solution_str: str = None,
    ground_truth: str = None,
    enclosing_think_tag_str: str = "</think>",
    extract_from_answer_only: bool = True,
) -> float:
    """
    Extract a boxed mathematical expression from the model output and compare it
    to the ground-truth answer using a math-aware verifier.

    Args:
        solution_str: Full model output (may include chain-of-thought and final answer).
        ground_truth: Gold answer string (may or may not be in LaTeX math mode).
        enclosing_think_tag_str: Tag that marks the end of the "thinking" section.
            If extract_from_answer_only is True, everything after this tag is treated
            as the answer-only region to parse.
        extract_from_answer_only: If True, parse only the region after 'enclosing_think_tag_str'.
            If the tag is missing, score is 0.0.

    Returns:
        1.0 if verified correct, else 0.0. Any parsing/verifying exceptions yield 0.0.
    """
    if extract_from_answer_only:
        # Require the explicit think/answer boundary for robust extraction.
        if enclosing_think_tag_str not in solution_str:
            return 0.0
        parts = solution_str.split(enclosing_think_tag_str, 1)
        assert len(parts) == 2, f"Expected 2 parts, but got {len(parts)}: {parts}"
        solution_str = parts[1]

    # Ensure ground truth is in LaTeX math mode so the parser can handle it uniformly.
    enclosed_with_brackets = "\\[" in ground_truth and "\\]" in ground_truth
    enclosed_with_parentheses = "\\(" in ground_truth and "\\)" in ground_truth
    enclosed_with_dollars = ground_truth.startswith("$") and ground_truth.endswith("$")
    enclosed_with_math_mode = enclosed_with_brackets or enclosed_with_parentheses or enclosed_with_dollars
    if not enclosed_with_math_mode:
        ground_truth = f"\\[{ground_truth}\\]"
    ground_truth = parse_gold_answer_fn(ground_truth)

    try:
        predicted_answer = parse_predicted_answer_fn(solution_str)
    except Exception as e:
        print(f"Error parsing predicted answer: {e}")
        return 0.0

    try:
        # Math equivalence check with a short timeout to avoid long hangs.
        is_correct = verify(
            gold=ground_truth,
            target=predicted_answer,
            timeout_seconds=2,
        )
        return 1.0 if is_correct else 0.0
    except Exception as e:
        print(f"Error verifying answer: {e}")
        return 0.0


def _get_output_ids(response: dict[str, Any]) -> list[int]:
    """Extract generated token IDs from an sglang response."""
    if "output_ids" in response:
        return response["output_ids"]
    output_token_logprobs = response["meta_info"]["output_token_logprobs"]
    _, output_token_ids = zip(
        *[(log_prob, token_ids) for log_prob, token_ids, _ in output_token_logprobs],
        strict=True,
    )
    return list(output_token_ids)


async def delethink_tracing(
    llm: sgl.Engine,
    sample: dict[str, Any],
    context_size: int,
    markovian_size: int,
    max_iterations: int,
) -> tuple[dict[str, Any], list[int]]:
    """
    Run a single Delethink-style rollout for one sample.

    The rollout proceeds in fixed-size "chunks":
      - Iteration 0 uses up to 'context_size' new tokens.
      - Later iterations use at most (context_size - markovian_size) new tokens,
        then carry over only the last 'markovian_size' tokens as the next chunk's prompt tail.
      - The prompt is reset at each chunk boundary (KV cache is effectively cleared),
        so the carryover has to be re-encoded (see paper text).

    Returns:
        (original sample, concatenated list of all generated token IDs across chunks)
    """
    tokenizer = llm.tokenizer_manager.tokenizer
    sampling_params = {"temperature": 0.6}  # TODO: expose via args if desired

    # Build the user message and tokenize with chat template (adds generation prompt).
    query = PROMPT_TEMPLATE.format(problem=sample["problem"])
    query_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": query}], tokenize=True, add_generation_prompt=True
    )

    trace_response_ids = []  # list of lists, one per chunk

    iterations = 0
    prompt_ids = query_ids  # current chunk's prompt (reset/rebuilt each iteration)

    while iterations < max_iterations:
        # First iteration: full budget; subsequent: only non-Markovian part
        params = sampling_params.copy()
        params["max_new_tokens"] = (context_size - markovian_size) if iterations > 0 else context_size

        # Generate next chunk
        response = await llm.async_generate(input_ids=prompt_ids, sampling_params=params, return_logprob=True)
        response_ids = _get_output_ids(response)
        trace_response_ids.append(response_ids)

        # Heuristic: after the first chunk, persist the first 100 generated tokens
        # into the running "query_ids" prefix (kept as-is; matches original logic).
        if iterations == 0:
            query_ids = query_ids + response_ids[:100]

        # Stop if the model indicates EOS ('stop' finish reason in sglang semantics).
        finish_reason_is_eos = response["meta_info"]["finish_reason"]["type"] == "stop"
        if finish_reason_is_eos:
            break

        # Next chunk prompt = (running query_ids) + (last m tokens from current chunk).
        # This enforces the Markovian carryover and resets the rest of the context.
        prompt_ids = query_ids + response_ids[-markovian_size:]

        iterations += 1

    # Flatten list of lists into a single sequence of token IDs.
    trace_response_ids = sum(trace_response_ids, [])

    return sample, trace_response_ids


async def run_inference(llm, ds, args):
    """
    Launch Delethink rollouts over the dataset with 'rollout_n' per sample.
    Displays an asyncio-aware progress bar and computes scores/trace lengths.
    """
    tasks = []
    for sample in ds:
        for _ in range(args.rollout_n):
            tasks.append(
                asyncio.create_task(
                    delethink_tracing(
                        llm=llm,
                        sample=sample,
                        context_size=args.delethink_context_size,
                        markovian_size=args.delethink_markovian_size,
                        max_iterations=args.delethink_iteration_cap,
                    )
                )
            )

    # Progress bar updates as individual tasks finish.
    scores = []
    trace_lengths = []
    for task in tqdm_asyncio.as_completed(tasks, desc="Sampling Responses"):
        sample, response_ids = await task
        # Decode entire concatenated trace. Keep special tokens as in original.
        response = llm.tokenizer_manager.tokenizer.decode(response_ids, skip_special_tokens=False)
        score = compute_score(
            solution_str=response,
            ground_truth=sample["answer"],
        )
        scores.append(score)
        trace_lengths.append(len(response_ids))

    return scores, trace_lengths


def main():
    """Parse args, initialize engine/dataset, run inference, and report averages."""
    parser = argparse.ArgumentParser(description="Run sglang engine with specified model.")
    parser.add_argument(
        "--model",
        type=str,
        default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        help="Model path to use for sglang.Engine",
    )
    parser.add_argument("--tp", type=int, default=1, help="Tensor model parallel size")
    parser.add_argument(
        "--dp",
        type=int,
        default=1,
        help="Data parallel sizee. i.e. parallel inference engines.",
    )
    parser.add_argument("--samples", type=int, default=-1, help="Number of samples (default: all)")
    parser.add_argument("--rollout_n", type=int, default=1, help="Number of rollouts per sample")
    parser.add_argument(
        "--delethink_context_size",
        type=int,
        default=8192,
        help="Delethink context size (C)",
    )
    parser.add_argument(
        "--delethink_markovian_size",
        type=int,
        default=4096,
        help="Delethink markovian size (m)",
    )
    parser.add_argument(
        "--delethink_iteration_cap",
        type=int,
        default=5,
        help="Delethink iteration cap (I)",
    )
    parser.add_argument(
        "--debug",
        type=bool,
        default=False,
        help="Debug mode",
    )
    args = parser.parse_args()

    print("Arguments:")
    pprint(args.__dict__)

    # Effective token budget across I chunks:
    #   C + (I-1) * (C - m)
    # where C = delethink_context_size, m = markovian_size.
    effective_token_budget = args.delethink_context_size + (args.delethink_iteration_cap - 1) * (
        args.delethink_context_size - args.delethink_markovian_size
    )
    print(f"Effective Token Budget: {effective_token_budget}")

    # Load AIME 2024 train split as a list of dicts.
    ds = load_dataset("HuggingFaceH4/aime_2024", split="train").to_list()

    # Optional downsampling for quick runs.
    if args.samples != -1:
        ds = ds[: args.samples]

    print("Dataset length:", len(ds))

    # Model config used for dtype, etc. (tokenizer comes from sglang Engine).
    config = AutoConfig.from_pretrained(args.model)

    # Base prompt budget (outside Delethink chunking). Kept as a fixed cap here.
    max_prompt_length = 2048

    # Initialize sglang Engine.
    # context_length accounts for the base prompt plus the per-chunk Delethink budget.
    llm = sgl.Engine(
        model_path=args.model,
        dtype=config.torch_dtype,
        tp_size=args.tp,
        dp_size=args.dp,
        trust_remote_code=True,  # NOTE: Implicitly trusts model repo code.
        context_length=max_prompt_length + args.delethink_context_size,
        attention_backend="flashinfer",
        mem_fraction_static=0.8,
        log_level="INFO" if args.debug else "WARNING",
    )

    # Run all rollouts (async) and summarize.
    scores, trace_lengths = asyncio.run(run_inference(llm, ds, args))
    print(f"Average score: {np.mean(scores):.4f}")
    print(f"Average trace length: {np.mean(trace_lengths):.4f}")


if __name__ == "__main__":
    main()
