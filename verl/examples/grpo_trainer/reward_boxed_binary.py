import math
import re
from fractions import Fraction
from typing import Optional

try:
    from verl.utils.reward_score.prime_math import grade_answer as _grade_answer
except Exception:
    _grade_answer = None


_WHITESPACE_RE = re.compile(r"\s+")
_TEXT_WRAPPER_RE = re.compile(r"^\\text\{(.+)\}$", re.DOTALL)
_SIMPLE_FRAC_RE = re.compile(r"^\\frac\{([^{}]+)\}\{([^{}]+)\}$")
_PLAIN_FRAC_RE = re.compile(r"^[-+]?\d+\s*/\s*[-+]?\d+$")
_PLAIN_FLOAT_RE = re.compile(r"^[-+]?\d+(?:\.\d+)?$")


def extract_last_boxed(text: str) -> Optional[str]:
    if not text:
        return None

    start = max(text.rfind(r"\boxed{"), text.rfind(r"\fbox{"))
    if start < 0:
        return None

    left_brace = text.find("{", start)
    if left_brace < 0:
        return None

    depth = 0
    buf: list[str] = []
    i = left_brace
    while i < len(text):
        ch = text[i]
        if ch == "{":
            depth += 1
            if depth > 1:
                buf.append(ch)
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return "".join(buf).strip()
            if depth < 0:
                return None
            buf.append(ch)
        else:
            if depth >= 1:
                buf.append(ch)
        i += 1
    return None


def _strip_outer_math_delimiters(ans: str) -> str:
    s = ans.strip()
    while len(s) >= 2 and s.startswith("$") and s.endswith("$"):
        s = s[1:-1].strip()
    return s


def _strip_redundant_outer_parens(ans: str) -> str:
    s = ans.strip()
    while len(s) >= 2 and s[0] == "(" and s[-1] == ")":
        depth = 0
        valid = True
        for i, ch in enumerate(s):
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth < 0:
                    valid = False
                    break
                if depth == 0 and i != len(s) - 1:
                    valid = False
                    break
        if not valid or depth != 0:
            break
        s = s[1:-1].strip()
    return s


def normalize_math_answer(ans: str) -> str:
    s = (ans or "").strip()
    s = re.sub(r"\\{2,}", r"\\", s)
    s = _strip_outer_math_delimiters(s)
    s = s.replace(r"\left", "").replace(r"\right", "")
    s = s.replace(r"\dfrac", r"\frac").replace(r"\tfrac", r"\frac")

    text_match = _TEXT_WRAPPER_RE.match(s)
    if text_match:
        s = text_match.group(1).strip()

    s = _strip_redundant_outer_parens(s)
    s = _WHITESPACE_RE.sub("", s)
    return s


def _to_numeric_value(ans: str) -> Optional[float]:
    s = (ans or "").strip().replace(",", "")
    if not s:
        return None

    frac_match = _SIMPLE_FRAC_RE.match(s)
    if frac_match:
        num_s = frac_match.group(1).strip()
        den_s = frac_match.group(2).strip()
        try:
            num = Fraction(num_s)
            den = Fraction(den_s)
            if den == 0:
                return None
            return float(num / den)
        except Exception:
            return None

    if _PLAIN_FRAC_RE.match(s):
        try:
            num_s, den_s = [x.strip() for x in s.split("/", 1)]
            den = int(den_s)
            if den == 0:
                return None
            return float(Fraction(int(num_s), den))
        except Exception:
            return None

    if _PLAIN_FLOAT_RE.match(s):
        try:
            return float(s)
        except Exception:
            return None

    return None


def answers_equivalent(pred: str, gt: str) -> bool:
    if not pred or not gt:
        return False

    pred_norm = normalize_math_answer(pred)
    gt_norm = normalize_math_answer(gt)

    if pred_norm == gt_norm:
        return True

    if _grade_answer is not None:
        try:
            if _grade_answer(pred_norm, gt_norm):
                return True
        except Exception:
            pass

    pred_num = _to_numeric_value(pred_norm)
    gt_num = _to_numeric_value(gt_norm)
    if pred_num is not None and gt_num is not None:
        return math.isclose(pred_num, gt_num, rel_tol=1e-9, abs_tol=1e-9)

    return pred_norm == gt_norm


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    _ = (data_source, extra_info, kwargs)
    gt_norm = normalize_math_answer(str(ground_truth))

    boxed = extract_last_boxed(solution_str or "")
    pred_norm = normalize_math_answer(boxed or "")
    is_correct = boxed is not None and answers_equivalent(pred_norm, gt_norm)
    score = 1.0 if is_correct else 0.0

    return {
        "score": score,
        "acc": score,
        "pred": pred_norm,
        "gt_norm": gt_norm,
    }
