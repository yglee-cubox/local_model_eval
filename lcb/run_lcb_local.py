"""Run LiveCodeBench code generation on a locally trained (HF-format) checkpoint.

lcb_runner only knows models listed in lcb_runner/lm_styles.py. This wrapper registers the checkpoint
on the fly and then hands over to lcb_runner's own main(), so generation, evaluation and output files
are exactly the official ones.

Prompt styles (--prompt_style):
  chat_template  (default) system + user message rendered with the checkpoint's own tokenizer chat_template.
                 Use this for instruction-tuned / chat models.
  base           lcb_runner's GenericBase few-shot prompt for base (non-chat) models.
  <LMStyle>      any lcb_runner LMStyle name (e.g. CodeQwenInstruct, DeepSeekR1) to reuse an official template.

Example:
  python run_lcb_local.py --model_path /purestorage/ailab/<me>/ckpt/step-1000 --model_name my-sft-1000 \
      --output_dir /purestorage/ailab/<me>/lcb_results -- --release_version release_v6 --n 10 --evaluate

Everything after `--` is passed to lcb_runner unchanged (see lcb_runner/runner/parser.py).
"""
import argparse
import os
import sys
from datetime import datetime

START_DIR = os.getcwd()
LCB_REPO = os.environ.get("LCB_REPO", "/purestorage/ailab/yglee/workspace/benchmarks/livecodebench")
sys.path.insert(0, LCB_REPO)
os.chdir(LCB_REPO)  # lcb_runner opens few-shot example files relative to the repo root at import time

from lcb_runner import lm_styles  # noqa: E402
from lcb_runner.lm_styles import LanguageModel, LMStyle  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model_path", required=True, help="HF-format checkpoint dir (config.json, tokenizer, weights)")
    p.add_argument("--model_name", default=None, help="name used for the output folder (default: last 2 path parts)")
    p.add_argument("--prompt_style", default="chat_template")
    p.add_argument("--output_dir", required=True, help="results go to <output_dir>/output/<model_name>/")
    args, rest = p.parse_known_args()
    if rest and rest[0] == "--":
        rest = rest[1:]
    return args, rest


def install_chat_template_prompt(model_path: str):
    """Make codegeneration prompts use the checkpoint's own chat template."""
    from transformers import AutoTokenizer

    from lcb_runner.prompts.code_generation import PromptConstants, get_generic_question_template_answer
    from lcb_runner.runner import scenario_router

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if not tokenizer.chat_template:
        sys.exit(f"{model_path} has no chat_template in tokenizer_config.json; use --prompt_style base")

    def format_prompt_generation(question, model_style):
        messages = [
            {"role": "system", "content": PromptConstants.SYSTEM_MESSAGE_GENERIC},
            {"role": "user", "content": get_generic_question_template_answer(question)},
        ]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    scenario_router.format_prompt_generation = format_prompt_generation
    return tokenizer


def main():
    args, rest = parse_args()
    model_path = os.path.abspath(os.path.join(START_DIR, args.model_path))
    name = args.model_name or "-".join(model_path.rstrip("/").split("/")[-2:])

    if "--scenario" in rest and args.prompt_style == "chat_template":
        scenario = rest[rest.index("--scenario") + 1]
        if scenario != "codegeneration":
            sys.exit("--prompt_style chat_template supports only --scenario codegeneration; "
                     "pick an LMStyle (e.g. CodeQwenInstruct) for other scenarios")

    if args.prompt_style == "chat_template":
        # Placeholder style: routes to the vLLM runner and extracts the last ```python block.
        style = LMStyle.CodeQwenInstruct
        tokenizer = install_chat_template_prompt(model_path)
        # lcb_runner stops on "###" by default, which cuts off markdown headings in chat answers.
        if "--stop" not in rest:
            rest += ["--stop", tokenizer.eos_token or "</s>"]
    elif args.prompt_style == "base":
        style = LMStyle.GenericBase
    else:
        style = LMStyle[args.prompt_style]

    lm = LanguageModel(name, name, style, datetime(2023, 1, 1), link=model_path)
    lm_styles.LanguageModelList.append(lm)
    lm_styles.LanguageModelStore[name] = lm

    if os.environ.get("SAMPLING_KWARGS"):
        # Extra request params for API-style runners (OpenAIChat), e.g. '{"presence_penalty":1.5}'.
        # lcb_runner hard-codes presence_penalty=0 etc. in OpenAIRunner.client_kwargs.
        import json

        from lcb_runner.runner.oai_runner import OpenAIRunner

        extra = json.loads(os.environ["SAMPLING_KWARGS"])
        orig_init = OpenAIRunner.__init__

        def init(self, *a, **kw):
            orig_init(self, *a, **kw)
            self.client_kwargs.update(extra)

        OpenAIRunner.__init__ = init
        print(f"[run_lcb_local] extra request params: {extra}", flush=True)

    from lcb_runner.runner.main import main as lcb_main

    output_dir = os.path.abspath(os.path.join(START_DIR, args.output_dir))
    os.makedirs(output_dir, exist_ok=True)
    os.chdir(output_dir)  # lcb_runner writes to ./output/<model_name>/
    sys.argv = [sys.argv[0], "--model", name, "--local_model_path", model_path, *rest]
    print(f"[run_lcb_local] model={name} style={args.prompt_style} path={model_path}\n"
          f"[run_lcb_local] lcb_runner args: {' '.join(sys.argv[1:])}\n"
          f"[run_lcb_local] results: {output_dir}/output/{name}/", flush=True)
    lcb_main()


if __name__ == "__main__":
    main()
