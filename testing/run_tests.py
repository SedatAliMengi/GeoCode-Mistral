#!/usr/bin/env python3
"""
GeoCode-Mistral — self-contained evaluation harness.

Runs the 25 questions in test_questions.jsonl through the fine-tuned model,
saving each response immediately (resumable), then generates a manual-review
report at results/evaluation_report.md.

Usage (from anywhere):
    python testing/run_tests.py

- Loads ./models/final_model once (bfloat16, device_map="auto") — same loading
  call as chatbot.py, with bf16 as requested.
- Mistral [INST] prompt format, same as chatbot.py.
- Fully automatic, no input needed. Stop with Ctrl+C and re-run to resume.
"""

import json
import re
from collections import OrderedDict
from datetime import datetime
from pathlib import Path

import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# --------------------------------------------------------------------------- #
# Paths (resolved relative to this file, so it runs from any working directory)
# --------------------------------------------------------------------------- #
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
MODEL_PATH = PROJECT_ROOT / "models" / "final_model"   # same model as chatbot.py
QUESTIONS_FILE = SCRIPT_DIR / "test_questions.jsonl"
RESULTS_DIR = SCRIPT_DIR / "results"
RESPONSES_FILE = RESULTS_DIR / "responses.jsonl"
REPORT_FILE = RESULTS_DIR / "evaluation_report.md"

# --------------------------------------------------------------------------- #
# Report / run metadata
# --------------------------------------------------------------------------- #
PROJECT_NAME = "GeoCode-Mistral"
HARDWARE = "NVIDIA RTX 4070 Laptop GPU (8GB VRAM), Intel Core i7-14700HX, 32GB RAM"
EVAL_DATASET = "GeoCode-Eval (Hou et al., 2025)"

# Output budget per task type (MCQ answers are tiny; code generation needs room)
MAX_NEW_TOKENS = {
    "api_knowledge": 128,
    "platform_identify": 128,
    "code_to_summary": 192,
    "summary_to_code": 512,
}
DEFAULT_MAX_NEW_TOKENS = 256

SCORING_METHODOLOGY = (
    "Scoring was performed by human review against expected outputs from the "
    "official GeoCode-Eval benchmark (Hou et al., 2025). A response was marked "
    "as Pass if it correctly addressed the task without hallucinated API references. "
    "MCQ responses were marked Pass if the correct option was selected. "
    "Code generation responses were marked Pass if the output was syntactically "
    "valid and semantically correct for the described spatial task."
)


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def load_questions():
    questions = []
    with open(QUESTIONS_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(json.loads(line))
    return questions


def load_done_ids():
    """Return the set of question_ids already present in responses.jsonl."""
    done = set()
    if RESPONSES_FILE.exists():
        with open(RESPONSES_FILE, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    done.add(json.loads(line)["question_id"])
                except (json.JSONDecodeError, KeyError):
                    continue
    return done


# --------------------------------------------------------------------------- #
# Prompt building — same Mistral [INST] format as chatbot.py
# --------------------------------------------------------------------------- #
def build_prompt(instruction, input_text):
    user_msg = (instruction or "").strip()
    if input_text and str(input_text).strip():
        user_msg += "\n\n" + str(input_text).strip()
    return f"<s>[INST] {user_msg} [/INST]"


# --------------------------------------------------------------------------- #
# Automated scoring — ONLY for multiple-choice tasks (single-letter answers).
# Open-ended code/summary tasks are left for manual human review.
# --------------------------------------------------------------------------- #
MCQ_TASKS = {"api_knowledge", "platform_identify"}


def extract_mcq_letter(text):
    """
    Pull the chosen A/B/C/D option from a model response, conservatively.
    Returns the letter, or None if no clear choice can be identified (so a
    rambling answer that never commits to a letter is NOT guessed at).
    """
    if not text:
        return None
    up = text.strip().upper()
    # Only trust a letter the model commits to at the very START of its answer.
    # (a) The whole answer is just a letter: "C", "C.", "(B)", "D)"
    m = re.match(r"^\(?\s*([ABCD])\s*[\).:\-]?\s*$", up)
    if m:
        return m.group(1)
    # (b) Leading letter + a real choice delimiter: "C.", "C)", "(B)", "D: ..."
    #     A plain space is NOT accepted, so "A great tool" won't match.
    m = re.match(r"^\(?\s*([ABCD])\s*[\).:]", up)
    if m:
        return m.group(1)
    # Anything else (essay-style answers, echoed choice lists) -> undetermined.
    # We deliberately do NOT scan mid-text for letters: this model often repeats
    # the whole "A) ... B) ... C) ..." list, which would cause false matches.
    return None


def score_mcq(question, response_text):
    """
    Return (predicted_letter, expected_letter, status) for an MCQ item.
    status is "pass", "fail", or "manual" (no clear letter -> human review).
    """
    predicted = extract_mcq_letter(response_text)
    expected = (question.get("expected_output") or "").strip().upper()
    if predicted is None:
        status = "manual"
    elif predicted == expected:
        status = "pass"
    else:
        status = "fail"
    return predicted, expected, status


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
def load_model():
    print(f"Loading model from {MODEL_PATH} (bfloat16, device_map=auto)...")
    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_PATH),
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH))
    model.eval()
    print("Model loaded.\n")
    return model, tokenizer


def generate(model, tokenizer, question):
    prompt = build_prompt(question["instruction"], question["input"])
    input_device = "cuda" if torch.cuda.is_available() else model.device
    inputs = tokenizer(prompt, return_tensors="pt").to(input_device)
    input_len = inputs["input_ids"].shape[1]
    max_new = MAX_NEW_TOKENS.get(question["task_type"], DEFAULT_MAX_NEW_TOKENS)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new,
            do_sample=False,                 # greedy → reproducible benchmark runs
            repetition_penalty=1.2,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    new_tokens = out[0][input_len:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def append_response(record):
    """Append one response and flush immediately so Ctrl+C is always safe."""
    with open(RESPONSES_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()


# --------------------------------------------------------------------------- #
# Report generation
# --------------------------------------------------------------------------- #
def load_responses():
    responses = {}
    if RESPONSES_FILE.exists():
        with open(RESPONSES_FILE, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    responses[rec["question_id"]] = rec  # last write wins
                except (json.JSONDecodeError, KeyError):
                    continue
    return responses


def _fence(text):
    """Render text in a fenced block, picking a fence that won't collide."""
    text = "" if text is None else str(text)
    fence = "```"
    while fence in text:
        fence += "`"
    return f"{fence}\n{text}\n{fence}"


def generate_report(questions):
    responses = load_responses()
    today = datetime.now().strftime("%Y-%m-%d")

    # Auto-score MCQ items only: qid -> (predicted, expected, status)
    mcq = {}
    for q in questions:
        if q["task_type"] in MCQ_TASKS:
            resp = responses.get(q["question_id"], {}).get("response", "")
            mcq[q["question_id"]] = score_mcq(q, resp)
    mcq_total = len(mcq)
    mcq_pass = sum(1 for v in mcq.values() if v[2] == "pass")
    mcq_fail = sum(1 for v in mcq.values() if v[2] == "fail")
    mcq_manual = sum(1 for v in mcq.values() if v[2] == "manual")
    mcq_determined = mcq_pass + mcq_fail

    # Aggregate counts
    by_task = OrderedDict()
    by_level = OrderedDict()
    for q in questions:
        by_task.setdefault(q["task_type"], 0)
        by_task[q["task_type"]] += 1
        by_level.setdefault(q["difficulty"], 0)
        by_level[q["difficulty"]] += 1

    lines = []
    a = lines.append

    # ---- Header ----
    a(f"# {PROJECT_NAME} — Evaluation Report")
    a("")
    a(f"- **Project:** {PROJECT_NAME}")
    a(f"- **Date of evaluation:** {today}")
    a(f"- **Model path:** `{MODEL_PATH.as_posix()}`")
    a(f"- **Hardware:** {HARDWARE}")
    a(f"- **Evaluation dataset:** {EVAL_DATASET}")
    a(f"- **Questions evaluated:** {len(questions)}")
    a("")

    # ---- Automated scoring summary ----
    mcq_pct = f"{mcq_pass / mcq_determined * 100:.0f}%" if mcq_determined else "n/a"
    a("## Automated Scoring Summary")
    a("")
    a(f"- **Multiple-choice (auto-scored): {mcq_pass}/{mcq_determined} correct "
      f"({mcq_pct})** out of {mcq_total} MCQ.")
    if mcq_manual:
        a(f"- **{mcq_manual} MCQ could not be auto-scored** (model gave no clear "
          "A/B/C/D answer) — moved to manual review below.")
    a("- **Code generation + summarization (15 questions): pending manual review** "
      "— these cannot be reliably auto-graded and are scored by hand below.")
    a("- **Overall score:** to be finalized after manual review (run `--tally`).")
    a("")

    # ---- Results by task type ----
    a("## Results by Task Type")
    a("")
    a("| Task Type | Questions | Pass | Fail | Score |")
    a("|---|---|---|---|---|")
    for task, n in by_task.items():
        if task in MCQ_TASKS:
            ids = [q["question_id"] for q in questions if q["task_type"] == task]
            p = sum(1 for i in ids if mcq[i][2] == "pass")
            f_ = sum(1 for i in ids if mcq[i][2] == "fail")
            det = p + f_
            score = f"{p / det * 100:.0f}%" if det else "manual"
            a(f"| {task} | {n} | {p} | {f_} | {score} |")
        else:
            a(f"| {task} | {n} | | | |")
    a("")
    a("_MCQ rows auto-scored (any undetermined MCQ are excluded here and graded "
      "manually); code/summary rows left blank for manual review._")
    a("")

    # ---- Results by difficulty ----
    a("## Results by Difficulty Level")
    a("")
    a("| Level | Questions | Pass | Fail | Score |")
    a("|---|---|---|---|---|")
    for level in sorted(by_level):
        ids = [q["question_id"] for q in questions if q["difficulty"] == level]
        tasks_here = {q["task_type"] for q in questions if q["difficulty"] == level}
        if tasks_here <= MCQ_TASKS:  # MCQ-only level → auto-scored
            p = sum(1 for i in ids if mcq[i][2] == "pass")
            f_ = sum(1 for i in ids if mcq[i][2] == "fail")
            det = p + f_
            score = f"{p / det * 100:.0f}%" if det else "manual"
            a(f"| {level} | {len(ids)} | {p} | {f_} | {score} |")
        else:
            a(f"| {level} | {len(ids)} | | | |")
    a("")

    # ---- Comparison ----
    a("## Comparison with Original GeoCode-GPT")
    a("")
    a("| Metric | GeoCode-GPT (original) | GeoCode-Mistral (this project) |")
    a("|---|---|---|")
    a("| Base model | Code Llama | Mistral-7B |")
    a("| Hardware | A100 80GB | RTX 4070 8GB |")
    a("| Dataset | Full | ~40% subset |")
    a("| Reported score | 73% | [TO BE FILLED] |")
    a("")

    # ---- MCQ detail ----
    a("## Automated MCQ Scoring Detail")
    a("")
    a("| QID | Task Type | Model chose | Expected | Result |")
    a("|---|---|---|---|---|")
    for q in questions:
        qid = q["question_id"]
        if qid in mcq:
            pred, exp, status = mcq[qid]
            label = {"pass": "PASS", "fail": "FAIL", "manual": "MANUAL REVIEW"}[status]
            a(f"| {qid} | {q['task_type']} | {pred or 'N/A'} | {exp} | {label} |")
    a("")

    # ---- Individual results ----
    a("## Individual Question Results")
    a("")
    for q in questions:
        qid = q["question_id"]
        rec = responses.get(qid)
        response_text = rec.get("response", "") if rec else "(no response recorded)"

        a(f"### Q{qid} — {q['task_type']} (difficulty {q['difficulty']})")
        a("")
        a(f"_Source: `{q['source_file']}`_")
        a("")
        a("**Prompt (instruction + input):**")
        a(_fence(build_prompt(q["instruction"], q["input"])))
        a("")
        a("**Model response:**")
        a(_fence(response_text))
        a("")
        a("**Expected output:**")
        a(_fence(q["expected_output"]))
        a("")
        if qid in mcq:
            pred, exp, status = mcq[qid]
            if status == "manual":
                a(f"**Automated MCQ score:** UNDETERMINED — model gave no clear "
                  f"A/B/C/D answer (expected: {exp}). Please grade manually.")
                a("")
                a("- [ ] Pass")
                a("- [ ] Fail")
            else:
                passed = status == "pass"
                a(f"**Automated MCQ score:** {'PASS' if passed else 'FAIL'} "
                  f"(model chose: {pred}, expected: {exp})")
                a("")
                a(f"- [{'x' if passed else ' '}] Pass")
                a(f"- [{'x' if not passed else ' '}] Fail")
        else:
            a("**Manual grade — tick one:**")
            a("- [ ] Pass")
            a("- [ ] Fail")
        a("")
        a("**Notes:** ")
        a("")
        a("---")
        a("")

    # ---- Methodology ----
    a("## Scoring Methodology")
    a("")
    a(SCORING_METHODOLOGY)
    a("")
    a("_Automation note: the 10 multiple-choice questions were auto-scored by "
      "option-letter match where the model committed to a clear A/B/C/D; any MCQ "
      "with no parseable choice was moved to manual review rather than scored as "
      "wrong. The 15 code-generation and summarization questions were graded by "
      "manual human review._")
    a("")

    # ---- Summary & limitations ----
    a("## Summary and Limitations")
    a("")
    a("**What the numbers mean.** The Pass/Fail and Score columns above are left "
      "blank intentionally — they are filled in manually after human review of "
      "each response against the expected GeoCode-Eval output. Per-task and "
      "per-level scores are simple pass rates over the questions in each bucket.")
    a("")
    a("**Hardware constraint context.** This model was trained and is evaluated "
      "on a single consumer laptop GPU (RTX 4070, 8GB VRAM), versus the data-center "
      "A100 (80GB) used by the original GeoCode-GPT. Inference here relies on "
      "CPU offloading because the 7B model exceeds 8GB VRAM, so results reflect "
      "what is achievable under tight resource constraints rather than a "
      "best-case setup.")
    a("")
    a("**Subset acknowledgment.** This evaluation covers **25 questions** sampled "
      "from the GeoCode-Eval benchmark — a small subset chosen for tractable "
      "manual review, not the full benchmark (3,000+ MCQ plus 500 generation and "
      "500 summarization items). The fine-tuning data itself is also only ~40% of "
      "the original corpus. Scores should therefore be read as an indicative "
      "sanity check of capability, not a statistically robust benchmark result.")
    a("")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Report written to {REPORT_FILE}")


# --------------------------------------------------------------------------- #
# Tally — compute final accuracy from the graded report (Pass/Fail checkboxes)
# --------------------------------------------------------------------------- #
def tally_report():
    if not REPORT_FILE.exists():
        print(f"No report found at {REPORT_FILE}. Run the evaluation first.")
        return

    text = REPORT_FILE.read_text(encoding="utf-8")
    qmap = {q["question_id"]: q for q in load_questions()}

    # Split the markdown into per-question blocks: "### Q<id> — ..."
    parts = re.split(r"(?m)^### Q(\d+) ", text)
    graded = {}  # qid -> "pass" | "fail" | None
    it = iter(parts[1:])
    for qid_str, body in zip(it, it):
        qid = int(qid_str)
        passed = bool(re.search(r"-\s*\[x\]\s*Pass", body, re.I))
        failed = bool(re.search(r"-\s*\[x\]\s*Fail", body, re.I))
        if passed and not failed:
            graded[qid] = "pass"
        elif failed and not passed:
            graded[qid] = "fail"
        else:
            graded[qid] = None  # ungraded, or both ticked (treated as not graded)

    def bucket(filter_fn):
        ids = [qid for qid in qmap if filter_fn(qmap[qid])]
        p = sum(1 for i in ids if graded.get(i) == "pass")
        f = sum(1 for i in ids if graded.get(i) == "fail")
        u = len(ids) - p - f
        acc = f"{p / (p + f) * 100:.0f}%" if (p + f) else "n/a"
        return len(ids), p, f, u, acc

    total = len(qmap)
    n, p, f, u, _ = bucket(lambda q: True)
    ungraded_ids = sorted(i for i in qmap if graded.get(i) is None)
    today = datetime.now().strftime("%Y-%m-%d")

    out = []
    w = out.append
    w(f"# {PROJECT_NAME} — Final Scores")
    w("")
    w(f"_Computed {today} from `{REPORT_FILE.name}` (auto-scored MCQ + manual grades)._")
    w("")
    w("## Overall")
    w("")
    w(f"- Graded: **{p + f} / {total}**")
    w(f"- Pass: **{p}**")
    w(f"- Fail: **{f}**")
    w(f"- Ungraded: **{u}**" + (f" (Q{', Q'.join(map(str, ungraded_ids))})" if ungraded_ids else ""))
    w(f"- **Accuracy (pass / graded): {p / (p + f) * 100:.1f}%**" if (p + f) else "- Accuracy: n/a")
    w(f"- Accuracy (pass / all {total}): {p / total * 100:.1f}%")
    w("")
    w("## By Task Type")
    w("")
    w("| Task Type | Questions | Pass | Fail | Ungraded | Accuracy |")
    w("|---|---|---|---|---|---|")
    for task in dict.fromkeys(q["task_type"] for q in qmap.values()):
        n, p2, f2, u2, acc = bucket(lambda q: q["task_type"] == task)
        w(f"| {task} | {n} | {p2} | {f2} | {u2} | {acc} |")
    w("")
    w("## By Difficulty Level")
    w("")
    w("| Level | Questions | Pass | Fail | Ungraded | Accuracy |")
    w("|---|---|---|---|---|---|")
    for level in sorted({q["difficulty"] for q in qmap.values()}):
        n, p2, f2, u2, acc = bucket(lambda q: q["difficulty"] == level)
        w(f"| {level} | {n} | {p2} | {f2} | {u2} | {acc} |")
    w("")

    scores_file = RESULTS_DIR / "final_scores.md"
    scores_file.write_text("\n".join(out), encoding="utf-8")

    # Console summary
    print("=" * 50)
    print(f"Graded {p + f}/{total}  |  Pass {p}  Fail {f}  Ungraded {u}")
    if p + f:
        print(f"Overall accuracy (pass/graded): {p / (p + f) * 100:.1f}%")
    if ungraded_ids:
        print(f"Still ungraded: Q{', Q'.join(map(str, ungraded_ids))}")
    print(f"Saved -> {scores_file}")
    print("=" * 50)


def maybe_generate_report(questions):
    """Generate the report, but never silently overwrite manual grades."""
    if REPORT_FILE.exists():
        print(f"\nReport already exists at {REPORT_FILE}.")
        print("Not overwriting (it may contain your manual grades).")
        print("  - To compute final scores from your grades:  python testing/run_tests.py --tally")
        print("  - To rebuild the report from scratch:        python testing/run_tests.py --force-report")
        return
    generate_report(questions)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(force_report=False):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    questions = load_questions()
    total = len(questions)
    done = load_done_ids()

    pending = [q for q in questions if q["question_id"] not in done]
    if not pending:
        print(f"All {total} questions already answered.")
        (generate_report if force_report else maybe_generate_report)(questions)
        return

    if done:
        print(f"Resuming: {len(done)} already done, {len(pending)} remaining.\n")

    model, tokenizer = load_model()

    pbar = tqdm(total=total, initial=len(done), desc="Evaluating", unit="q", dynamic_ncols=True)
    try:
        for q in questions:
            qid = q["question_id"]
            if qid in done:
                continue

            pbar.set_postfix_str(f"Q{qid} {q['task_type']}")
            response = generate(model, tokenizer, q)

            append_response({
                "question_id": qid,
                "task_type": q["task_type"],
                "difficulty": q["difficulty"],
                "source_file": q["source_file"],
                "response": response,
                "timestamp": datetime.now().isoformat(),
            })
            preview = response.replace("\n", " ")[:80]
            pbar.write(f"  Q{qid}/{total} [{q['task_type']}] saved -> "
                       f"{preview}{'...' if len(response) > 80 else ''}")
            pbar.update(1)

    except KeyboardInterrupt:
        pbar.close()
        print("\n\nInterrupted. Progress is saved — re-run to resume where you left off.")
        return
    finally:
        pbar.close()

    # All questions answered → build the report
    done = load_done_ids()
    if all(q["question_id"] in done for q in questions):
        print("\nAll questions answered. Generating report...")
        (generate_report if force_report else maybe_generate_report)(questions)
    else:
        missing = [q["question_id"] for q in questions if q["question_id"] not in done]
        print(f"\nFinished loop but missing responses for: {missing}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="GeoCode-Mistral evaluation harness")
    parser.add_argument("--tally", action="store_true",
                        help="Compute final scores from the graded report (Pass/Fail checkboxes)")
    parser.add_argument("--force-report", action="store_true",
                        help="Rebuild evaluation_report.md even if it exists (overwrites manual grades)")
    args = parser.parse_args()

    if args.tally:
        tally_report()
    else:
        main(force_report=args.force_report)
