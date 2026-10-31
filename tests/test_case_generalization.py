import pytest
from src.language_model import LanguageModel

@pytest.fixture
def lm():
    return LanguageModel()

def test_all_uppercase_document(lm):
    text = "THE QUICK BROWN FOX JUMPS"
    result = lm.correct_sentence(text)
    assert result == "THE QUICK BROWN FOX JUMPS"

def test_standard_sentence_case(lm):
    text = "The quick brown fox jumps over the lazy dog."
    result = lm.correct_sentence(text)
    # Ensure it is not converted to all-caps
    assert result.startswith("The ")
    assert any(c.islower() for c in result)

def test_lowercase_document(lm):
    text = "learning never exhausts the mind."
    result = lm.correct_sentence(text)
    assert result == "learning never exhausts the mind."

def test_acronym_inside_sentence(lm):
    text = "We visited NASA yesterday."
    result = lm.correct_sentence(text)
    # 'We' and 'NASA' must preserve their casing and not force all other words to uppercase
    assert "NASA" in result

def test_case_sensitive_vs_insensitive_evaluation():
    import numpy as np
    ref = "SUCCESS IS NOT FINAL"
    hyp_upper = "SUCRES IS NOT FINAL"
    hyp_lower = "sucres is not final"
    
    def cer(r, h):
        r_list, h_list = list(r), list(h)
        d = np.zeros((len(r_list)+1, len(h_list)+1), dtype=np.uint16)
        for i in range(len(r_list)+1): d[i, 0] = i
        for j in range(len(h_list)+1): d[0, j] = j
        for i in range(1, len(r_list)+1):
            for j in range(1, len(h_list)+1):
                if r_list[i-1] == h_list[j-1]: d[i, j] = d[i-1, j-1]
                else: d[i, j] = 1 + min(d[i-1, j], d[i, j-1], d[i-1, j-1])
        return d[len(r_list), len(h_list)] / max(1, len(r_list))

    # Case-insensitive CER should match regardless of casing
    assert cer(ref.lower(), hyp_upper.lower()) == cer(ref.lower(), hyp_lower.lower())
    # Case-sensitive CER shows divergence
    assert cer(ref, hyp_lower) > cer(ref, hyp_upper)
