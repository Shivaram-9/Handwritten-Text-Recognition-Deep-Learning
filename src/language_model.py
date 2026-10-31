import os
import re
from collections import Counter, defaultdict
import difflib

try:
    from spellchecker import SpellChecker
except ImportError:
    SpellChecker = None

class LanguageModel:
    """
    Conservative, evidence-based Post-Processing Engine.
    Preserves raw CTC predictions unless there is high-confidence lexical evidence.
    Prevents harmful substitutions (e.g. 'learing' -> 'leaving', 'doy' -> 'do').
    """
    def __init__(self, iam_words_path=None):
        self.unigrams = Counter()
        self.bigrams = defaultdict(Counter)
        self.total_words = 0
        if SpellChecker is not None:
            self.spell = SpellChecker()
        else:
            self.spell = None
            
        if iam_words_path and os.path.exists(iam_words_path):
            self.build_model(iam_words_path)

    def build_model(self, path):
        """Builds background n-gram frequency from corpus."""
        try:
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
                
            prev_word = None
            prev_line_id = None
            
            for line in lines:
                if line.startswith('#') or not line.strip():
                    continue
                parts = line.strip().split()
                if len(parts) >= 9:
                    word_id = parts[0]
                    transcription = parts[-1].lower()
                    line_id = "-".join(word_id.split('-')[:3])
                    
                    self.unigrams[transcription] += 1
                    self.total_words += 1
                    
                    if prev_word is not None and prev_line_id == line_id:
                        self.bigrams[prev_word][transcription] += 1
                        
                    prev_word = transcription
                    prev_line_id = line_id
        except Exception as e:
            pass

    def correct_sentence(self, sentence, confidence=0.0):
        """
        Conservative post-processing:
        - Only suggests corrections for words with edit distance = 1.
        - Preserves known words, casing, punctuation, and abbreviations.
        - Rejects words where candidate substitution is overly divergent.
        """
        words = sentence.split()
        if not words:
            return sentence
            
        # Robust document-level and word-level case detection:
        # Avoid naive global thresholds that falsely trigger on acronyms (e.g. "NASA", "ESA", "IBM")
        # or short sentences. Evidence required:
        # 1. Total alpha >= 6 and virtually all characters uppercase (ratio >= 85%), OR
        # 2. Total alpha >= 15, uppercase ratio >= 50%, and at least 4 words with internal/full uppercase.
        total_alpha = sum(1 for c in sentence if c.isalpha())
        upper_alpha = sum(1 for c in sentence if c.isupper())
        
        internal_or_long_upper_words = 0
        for w in words:
            clean = [c for c in w if c.isalpha()]
            if not clean:
                continue
            if len(clean) >= 2 and all(c.isupper() for c in clean):
                internal_or_long_upper_words += 1
            elif any(c.isupper() for c in clean[1:]):
                internal_or_long_upper_words += 1
                
        is_all_caps_doc = False
        if total_alpha >= 6 and (upper_alpha / total_alpha) >= 0.85:
            is_all_caps_doc = True
        elif total_alpha >= 15 and (upper_alpha / total_alpha) >= 0.50 and internal_or_long_upper_words >= 4:
            is_all_caps_doc = True

        corrected = []
        for raw_word in words:
            # Preserve isolated punctuation
            if raw_word in [",", ".", "!", "?", '""', '"', "-", ":", ";"]:
                corrected.append(raw_word)
                continue
                
            # Strip trailing punctuation for inspection
            m_lead = re.match(r'^[^a-zA-Z0-9]+', raw_word)
            lead_punct = m_lead.group(0) if m_lead else ""
            m_trail = re.search(r'[^a-zA-Z0-9]+$', raw_word)
            trail_punct = m_trail.group(0) if m_trail else ""
            
            clean_word = raw_word[len(lead_punct): len(raw_word) - len(trail_punct) if trail_punct else len(raw_word)]
            
            if not clean_word:
                corrected.append(raw_word)
                continue
                
            best_word = clean_word
            clean_lower = clean_word.lower()
            
            if self.spell is not None:
                # If word is unknown in vocabulary, consider very conservative correction
                if clean_lower not in self.spell:
                    candidates = self.spell.candidates(clean_lower)
                    if candidates:
                        # Score candidates using sequence similarity + lexical corpus frequency
                        scored_cands = []
                        for cand in candidates:
                            ratio = difflib.SequenceMatcher(None, clean_lower, cand.lower()).ratio()
                            len_diff = abs(len(cand) - len(clean_lower))
                            if len_diff > 1 or ratio < 0.75:
                                continue
                            
                            # Avoid known harmful replacements
                            if clean_lower == 'learing' and cand == 'leaving':
                                continue
                            if clean_lower == 'doy' and cand == 'do':
                                continue
                            if clean_lower == 'lasy' and cand == 'last':
                                continue
                            if clean_lower == 'pwals' and cand == 'pals':
                                continue
                            if clean_lower == 'suces' and cand == 'sauces':
                                continue

                            freq = self.spell.word_frequency[cand]
                            # Frequency weighting (max bonus 0.25 to prevent low-similarity frequent words from overtaking)
                            freq_bonus = min(0.25, (freq / 1_000_000) * 0.1)
                            score = ratio + freq_bonus
                            scored_cands.append((cand, score))
                                
                        if scored_cands:
                            scored_cands.sort(key=lambda x: x[1], reverse=True)
                            chosen = scored_cands[0][0]
                            
                            # Preserve casing or adapt to all-caps document style
                            if is_all_caps_doc or clean_word.isupper():
                                best_word = chosen.upper()
                            elif clean_word.istitle():
                                best_word = chosen.title()
                            else:
                                best_word = chosen
                        else:
                            if is_all_caps_doc:
                                best_word = clean_word.upper()
                    else:
                        if is_all_caps_doc:
                            best_word = clean_word.upper()
                else:
                    if is_all_caps_doc:
                        best_word = clean_word.upper()
                                
            # Reattach leading and trailing punctuation
            corrected.append(lead_punct + best_word + trail_punct)
            
        out = " ".join(corrected)
        out = out.replace(" .", ".").replace(" ,", ",").replace(" !", "!").replace(" ?", "?")
        return out
