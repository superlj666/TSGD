import re
import regex
import multiprocessing
from math import isclose
from typing import Union, Optional
from collections import defaultdict
from sympy import simplify, N
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application, convert_xor
try:
    from sympy.parsing.latex import parse_latex
except ImportError:
    parse_latex = None

try:
    from latex2sympy2 import latex2sympy
except ImportError:
    latex2sympy = None

def choice_answer_clean(pred: str):
    pred = pred.strip("\n").rstrip(".").rstrip("/").strip(" ").lstrip(":")
    # Clean the answer based on the dataset
    tmp = re.findall(r"\b(A|B|C|D|E)\b", pred.upper())
    if tmp:
        pred = tmp
    else:
        pred = [pred.strip().strip(".")]
    pred = pred[-1]
    # Remove the period at the end, again!
    pred = pred.rstrip(".").rstrip("/")
    return pred

def parse_digits(num):
    num = str(num).replace(",", "")
    if "/" in num and "(" not in num and "Matrix" not in num:
        try:
            parts = num.split("/")
            if len(parts) == 2:
                return float(parts[0]) / float(parts[1])
        except:
            pass
    try:
        return float(num)
    except:
        if str(num).endswith("%"):
            num = str(num)[:-1]
            if num.endswith("\\"):
                num = num[:-1]
            try:
                return float(num) / 100
            except:
                pass
    return None

def is_digit(num):
    return parse_digits(num) is not None

def str_to_pmatrix(input_str):
    input_str = input_str.strip()
    matrix_str = re.findall(r"\{.*,.*\}", input_str)
    pmatrix_list = []

    for m in matrix_str:
        m = m.strip("{}")
        pmatrix = r"\begin{pmatrix}" + m.replace(",", "\\") + r"\end{pmatrix}"
        pmatrix_list.append(pmatrix)

    return ", ".join(pmatrix_list)

def split_pm(s: str) -> list[str]:
    r"""
    Expands a string containing \pm or \mp into multiple strings.
    """
    if r"\pm" not in s and r"\mp" not in s:
        return [s]
    
    parts = [s]
    # Handle \pm
    while any(r"\pm" in p for p in parts):
        new_parts = []
        for p in parts:
            if r"\pm" in p:
                new_parts.append(p.replace(r"\pm", "+", 1))
                new_parts.append(p.replace(r"\pm", "-", 1))
            else:
                new_parts.append(p)
        parts = new_parts
        
    # Handle \mp
    while any(r"\mp" in p for p in parts):
        new_parts = []
        for p in parts:
            if r"\mp" in p:
                new_parts.append(p.replace(r"\mp", "-", 1))
                new_parts.append(p.replace(r"\mp", "+", 1))
            else:
                new_parts.append(p)
        parts = new_parts
        
    return list(set(parts))

def split_list(s: str) -> list[str]:
    """
    Splits a string by commas, but only if the comma is not inside braces or parentheses.
    """
    if "," not in s:
        return [s]
    
    parts = []
    current = []
    depth = 0
    for char in s:
        if char in "{([":
            depth += 1
            current.append(char)
        elif char in "})]":
            depth -= 1
            current.append(char)
        elif char == "," and depth == 0:
            parts.append("".join(current).strip())
            current = []
        else:
            current.append(char)
    parts.append("".join(current).strip())
    return [p for p in parts if p]

def numeric_equal(prediction: float, reference: float):
    return isclose(reference, prediction, rel_tol=1e-4)

def symbolic_equal(a: str, b: str) -> bool:
    """
    Checks if two mathematical expressions are symbolically equivalent using SymPy.
    """
    def _parse(s: str):
        if not s:
            return None
            
        from sympy import Matrix, Symbol, Integer, Float
        from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application, convert_xor
        local_dict = {"Matrix": Matrix, "Symbol": Symbol, "Integer": Integer, "Float": Float}
        transformations = standard_transformations + (implicit_multiplication_application, convert_xor)

        # 1. Pre-clean for sympy parse_expr
        # Replace LaTeX-specific symbols
        s_clean = s.replace(r"\cdot", "*").replace(r"\times", "*")
        s_clean = s_clean.replace("^", "**")
        
        # Handle \frac
        s_clean = re.sub(r"\\(?:d|t)?frac\{([^{}]*)\}\{([^{}]*)\}", r"((\1)/(\2))", s_clean)
        
        # Handle \sqrt
        s_clean = re.sub(r"\\sqrt\{([^{}]*)\}", r"sqrt(\1)", s_clean)
        s_clean = re.sub(r"\\sqrt\[([^{}]*)\]\{([^{}]*)\}", r"(\2)**(1/(\1))", s_clean)
        
        # Remove remaining LaTeX commands and braces
        s_clean = re.sub(r"\\[a-zA-Z]+", " ", s_clean) # remove \left, \right, etc.
        s_clean = s_clean.replace("{", "(").replace("}", ")")
        
        try:
            res = parse_expr(s_clean, local_dict=local_dict, transformations=transformations, evaluate=True)
            if res is not None: return res
        except:
            pass

        # 2. Try LaTeX parsers as fallback
        if latex2sympy:
            try:
                res = latex2sympy(s)
                if res is not None: return res
            except: pass
            
        if parse_latex:
            try:
                res = parse_latex(s)
                if res is not None: return res
            except: pass
            
        return None

    a_parsed = _parse(a)
    b_parsed = _parse(b)
    
    if a_parsed is None or b_parsed is None:
        return False

    # 1. Direct equality
    try:
        if a_parsed == b_parsed or str(a_parsed) == str(b_parsed):
            return True
    except:
        pass

    # 2. Algebraic simplification
    try:
        diff = simplify(a_parsed - b_parsed)
        if diff == 0 or diff.is_zero:
            return True
    except:
        pass
        
    # 3. Floating point evaluation
    try:
        val_a = N(a_parsed)
        val_b = N(b_parsed)
        if numeric_equal(float(val_a), float(val_b)):
            return True
    except:
        pass

    # 4. Matrix equality
    try:
        if hasattr(a_parsed, 'shape') and a_parsed.shape == b_parsed.shape:
            if simplify(a_parsed - b_parsed).is_zero_matrix:
                return True
    except:
        pass

    return False

def symbolic_equal_process(a, b, output_queue):
    result = symbolic_equal(a, b)
    output_queue.put(result)

def call_with_timeout(func, *args, timeout=1, **kwargs):
    output_queue = multiprocessing.Queue()
    process_args = args + (output_queue,)
    process = multiprocessing.Process(target=func, args=process_args, kwargs=kwargs)
    process.start()
    process.join(timeout)

    if process.is_alive():
        process.terminate()
        process.join()
        return False

    try:
        return output_queue.get(block=False)
    except:
        return False

def normalize_str(s: str) -> str:
    """
    Robust normalization for math strings to handle LaTeX formatting variants.
    """
    if not s:
        return ""
    
    s = str(s).strip()
    
    # Remove leading/trailing quotes that might be added by the model
    if len(s) >= 2 and (s[0] == s[-1]) and s[0] in ["'", '"']:
        s = s[1:-1].strip()
    
    # 1. Remove LaTeX formatting commands and basic cleanup
    s = s.replace(r"\left", "").replace(r"\right", "")
    s = s.replace(r"\{", "{").replace(r"\}", "}")
    s = s.replace(r"\(", "(").replace(r"\)", ")")
    s = s.replace(r"\[", "[").replace(r"\]", "]")
    s = s.replace(r"\$", "").replace("$", "")
    
    # 2. Handle text-wrapping commands like \text{Evelyn} -> Evelyn
    # For numeric comparison, we often want to strip these if they are just units
    def clean_text_cmd(match):
        text_content = match.group(1).strip()
        # If it's purely alphabetical (like a unit), we might want to strip it later
        # For now, we keep the content but we'll improve unit stripping
        return text_content

    s = re.sub(r"\\(?:text|mathrm|textbf|textit|mathit|mathbf|num|mathsf|mathtt|mathcal|mbox)\{([^{}]*)\}", clean_text_cmd, s)
    
    # 3. Normalize fractions
    # Handle \frac{a}b or \frac ab or \frac a{b} -> \frac{a}{b}
    s = s.replace(r"\dfrac", r"\frac").replace(r"\tfrac", r"\frac")
    # Case 1: \frac{a}b -> \frac{a}{b}
    s = re.sub(r"\\frac\{([^{}]*)\}([^\{])", r"\\frac{\1}{\2}", s)
    # Case 2: \frac ab -> \frac{a}{b}
    s = re.sub(r"\\frac ([^\{])([^\{])", r"\\frac{\1}{\2}", s)
    # Case 3: \frac a{b} -> \frac{a}{b}
    s = re.sub(r"\\frac ([^\{])\{([^{}]*)\}", r"\\frac{\1}{\2}", s)
    
    # 4. Remove LaTeX spaces and common formatting artifacts
    s = s.replace(r"\quad", "").replace(r"\qquad", "")
    s = s.replace(r"\!", "").replace(r"\,", "").replace(r"\;", "").replace(r"\:", "")
    s = s.replace(r"~", "") # Non-breaking space in LaTeX
    
    # 5. Normalize operators
    s = s.replace(r"\bigcup", r"\cup")
    s = s.replace(r"\bigcap", r"\cap")
    s = s.replace(r"\times", "*").replace(r"\cdot", "*")
    
    # 6. Handle thousands separator in numbers (e.g., 32,348 -> 32348)
    s = re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", s)
    
    # 7. Remove common units and suffixes
    units = [
        "cm^2", "cm^3", "in^2", "in^3", "m^2", "m^3", "units^2", "unit^2", "units^3", "unit^3",
        "cm", "m", "km", "kg", "g", "lb", "oz", "inches", "inch", "ft", "feet", 
        "yards", "miles", "units", "unit", "degrees", "degree", "°", "sq", "cubic",
        "cents", "cent", "grade", "minutes", "minute", "hours", "hour", "seconds", "second",
        "years", "year", "months", "month", "weeks", "week", "days", "day",
        "meters", "meter", "km/h", "mph", "usd", "eur", "gbp", "points", "point",
        "th", "st", "nd", "rd",
        "^{th}", "^{st}", "^{nd}", "^{rd}",
        r"^\circ", r"\circ", "^2", "^3", "^{2}", "^{3}"
    ]
    
    # Clean up spaces before checking units
    s = re.sub(r"\s+", "", s)
    
    changed = True
    while changed:
        changed = False
        for unit in units:
            if s.lower().endswith(unit.lower()):
                s = s[:-len(unit)]
                changed = True
                break
            
    return s.lower().strip()

def normalize_matrix(s: str) -> str:
    r"""
    Converts LaTeX pmatrix/matrix environments to a uniform SymPy-friendly Matrix format.
    Example: \begin{pmatrix} a \\ b \end{pmatrix} -> Matrix([[a], [b]])
    """
    if "matrix" not in s:
        return s
    
    # Pre-process fractions inside the string to simplify matrix element parsing
    s = s.replace(r"\dfrac", r"\frac").replace(r"\tfrac", r"\frac")
    s = re.sub(r"\\frac\{([^{}]*)\}\{([^{}]*)\}", r"(\1)/(\2)", s)
    
    # Extract content between \begin{...matrix} and \end{...matrix}
    match = re.search(r"\\begin\{[p|b]?matrix\}(.*?)\\end\{[p|b]?matrix\}", s, re.DOTALL)
    if not match:
        return s
        
    content = match.group(1).strip()
    # Split by rows (\\)
    rows = [r.strip() for r in content.split(r"\\") if r.strip()]
    matrix_data = []
    for row in rows:
        # Split by columns (&)
        cols = [c.strip() for c in row.split("&") if c.strip()]
        # Clean each element: remove LaTeX artifacts but keep it as a parseable string
        cols = [c.replace(r"\right", "").replace(r"\left", "").strip() for c in cols]
        matrix_data.append(cols)
    
    return f"Matrix({matrix_data})"

def math_equal(
    prediction: Union[bool, float, str],
    reference: Union[float, str],
    include_percentage: bool = True,
    is_close: bool = True,
    timeout: bool = False,
) -> bool:
    """
    Official Omni-MATH Rule-based evaluation logic with enhanced normalization.
    """
    if prediction is None or reference is None:
        return False
    
    pred_str = str(prediction).strip()
    ref_str = str(reference).strip()
    
    # 0. Handle Equations and Set Membership (e.g., "x=5" vs "5", "x \in [1,2]" vs "[1,2]")
    # If one is a relation and the other is a simple value, compare the RHS
    def _strip_variable(s):
        # Handle both '=' and '\in' (including LaTeX \in)
        for rel in ["=", "\\in", "\u2208"]: # \u2208 is the unicode ∈ symbol
            if rel in s:
                parts = s.split(rel)
                # Take the last part (usually the value/expression)
                return parts[-1].strip()
        return s

    # Only strip if the other side doesn't have the same relation
    def _has_relation(s):
        return any(rel in s for rel in ["=", "\\in", "\u2208"])

    if _has_relation(pred_str) != _has_relation(ref_str):
        if math_equal(_strip_variable(pred_str), _strip_variable(ref_str), include_percentage, is_close, timeout):
            return True

    # 0.1 Handle \pm and lists (Set-based matching)
    # Expand \pm in both and split by commas
    ref_parts = []
    for p in split_pm(ref_str):
        ref_parts.extend(split_list(p))
    
    pred_parts = []
    for p in split_pm(pred_str):
        pred_parts.extend(split_list(p))
    
    if len(ref_parts) > 1 or len(pred_parts) > 1:
        # Normalize and remove duplicates
        ref_parts = sorted(list(set([p.strip() for p in ref_parts if p.strip()])))
        pred_parts = sorted(list(set([p.strip() for p in pred_parts if p.strip()])))
        
        if len(ref_parts) == len(pred_parts):
            # Try to match each part
            matched_indices = set()
            for p_part in pred_parts:
                found = False
                for i, r_part in enumerate(ref_parts):
                    if i not in matched_indices:
                        # Recursive call for each part, but disable set-based matching to avoid infinite recursion
                        # We use a simplified check or pass a flag. Here we just call math_equal.
                        if math_equal(p_part, r_part, include_percentage, is_close, timeout):
                            matched_indices.add(i)
                            found = True
                            break
                if not found:
                    break
            
            if len(matched_indices) == len(ref_parts):
                return True

    # Quick normalization check
    if normalize_str(pred_str) == normalize_str(ref_str):
        return True
        
    # Choice matching: (C) vs C vs \text{(C)}
    cleaned_pred = choice_answer_clean(pred_str)
    cleaned_ref = choice_answer_clean(ref_str)
    choices = ["A", "B", "C", "D", "E"]
    if cleaned_pred in choices and cleaned_ref in choices and cleaned_pred == cleaned_ref:
        return True
        
    if ref_str in choices and cleaned_pred == ref_str:
        return True

    try:  # 1. numerical equal
        # Try direct numeric conversion first
        p_val = None
        r_val = None
        
        if is_digit(pred_str) and is_digit(ref_str):
            p_val = parse_digits(pred_str)
            r_val = parse_digits(ref_str)
        else:
            # Try to extract a single number from the normalized strings
            norm_p = normalize_str(pred_str)
            norm_r = normalize_str(ref_str)
            if is_digit(norm_p) and is_digit(norm_r):
                p_val = parse_digits(norm_p)
                r_val = parse_digits(norm_r)
        
        if p_val is not None and r_val is not None:
            if include_percentage:
                gt_results = [r_val / 100, r_val, r_val * 100]
            else:
                gt_results = [r_val]
            for item in gt_results:
                if is_close:
                    if numeric_equal(p_val, item):
                        return True
                else:
                    if item == p_val:
                        return True
    except:
        pass

    if not pred_str and prediction not in [0, False]:
        return False

    # 2. symbolic equal
    if "pmatrix" in pred_str and "pmatrix" not in ref_str:
        ref_str = str_to_pmatrix(ref_str)

    # 1.8 Normalize matrices for symbolic check
    if "matrix" in pred_str or "matrix" in ref_str:
        pred_str = normalize_matrix(pred_str)
        ref_str = normalize_matrix(ref_str)

    # Symbolic check with SymPy (Attempt before destructive normalization)
    if timeout:
        if call_with_timeout(symbolic_equal_process, pred_str, ref_str):
            return True
    else:
        if symbolic_equal(pred_str, ref_str):
            return True

    # basic normalization for brackets
    p_norm, r_norm = pred_str, ref_str
    if (p_norm.startswith("[") and p_norm.endswith("]") and not r_norm.startswith("(")) or \
       (p_norm.startswith("(") and p_norm.endswith(")") and not r_norm.startswith("[")):
        p_norm = p_norm.strip("[]()")
        r_norm = r_norm.strip("[]()")
    
    for s in ["{", "}", "(", ")", "\\"]:
        r_norm = r_norm.replace(s, "")
        p_norm = p_norm.replace(s, "")
    
    # Also strip spaces in this fallback check
    r_norm = re.sub(r"\s+", "", r_norm)
    p_norm = re.sub(r"\s+", "", p_norm)

    if p_norm.lower() == r_norm.lower():
        return True

    return False

def extract_answer(text: str) -> str:
    """
    Extracts content from \boxed{} or <answer> tags.
    Handles nested structures and multiple occurrences.
    """
    # 1. Try <answer> tags first (take the LAST one)
    answer_contents = re.findall(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if answer_contents:
        content = answer_contents[-1].strip()
        # If the content contains \boxed{}, extract from it
        if "\\boxed{" in content:
            # Recursive call or nested extraction
            boxed_match = extract_answer(content)
            if boxed_match:
                return boxed_match
        return content

    # 2. Try \boxed{} tags
    # Handle nested braces for \boxed{}
    def find_balanced_braces(s, start_index):
        count = 0
        for i in range(start_index, len(s)):
            if s[i] == '{':
                count += 1
            elif s[i] == '}':
                count -= 1
                if count == 0:
                    return i
        return -1

    boxed_starts = [m.start() for m in re.finditer(r"\\boxed\s*\{", text)]
    if boxed_starts:
        # Use the last boxed content
        last_start = boxed_starts[-1]
        # Find the opening brace position
        open_brace_pos = text.find("{", last_start)
        content_start = open_brace_pos + 1
        end_index = find_balanced_braces(text, open_brace_pos)
        if end_index != -1:
            return text[content_start:end_index].strip()

    return ""
