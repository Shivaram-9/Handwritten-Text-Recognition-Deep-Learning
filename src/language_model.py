import os
import re
from collections import Counter, defaultdict
import difflib
import math

try:
    from spellchecker import SpellChecker
except ImportError:
    SpellChecker = None

class LanguageModel:
    def __init__(self, iam_words_path):
        self.unigrams = Counter()
        self.bigrams = defaultdict(Counter)
        self.total_words = 0
        if SpellChecker is not None:
            self.spell = SpellChecker()
        else:
            self.spell = None
        self.build_model(iam_words_path)
        
    def build_model(self, path):
        # 1. Parse IAM Words for domain probabilities
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                
            prev_word = None
            prev_line_id = None
            
            for line in lines:
                if line.startswith('#') or not line.strip():
                    continue
                parts = line.strip().split()
                if len(parts) >= 9:
                    word_id = parts[0]
                    transcription = parts[-1]
                    line_id = "-".join(word_id.split('-')[:3])
                    
                    self.unigrams[transcription] += 1
                    self.total_words += 1
                    
                    if prev_word is not None and prev_line_id == line_id:
                        self.bigrams[prev_word][transcription] += 1
                        
                    prev_word = transcription
                    prev_line_id = line_id

    def correct_sentence(self, sentence, confidence=0.0):
        # We process the sentence, fixing simple spelling errors if we have spellchecker
        words = sentence.split()
        if not words:
            return sentence
            
        corrected = []
        for raw_word in words:
            # Keep punctuation isolated if the model predicted it
            if raw_word in [",", ".", "!", "?", '""', '"']:
                corrected.append(raw_word)
                continue
                
            # Clean punctuation from ends for spell checking
            clean_word = raw_word.strip('.,!?"')
            if not clean_word:
                corrected.append(raw_word)
                continue
                
            best_word = raw_word
            
            if self.spell is not None:
                # If word is unknown, get best correction
                if clean_word.lower() not in self.spell:
                    cand = self.spell.correction(clean_word.lower())
                    if cand is not None:
                        # Match capitalization
                        if clean_word.istitle():
                            cand = cand.title()
                        elif clean_word.isupper():
                            cand = cand.upper()
                            
                        # Only accept if it's reasonably close (don't hallucinate completely different words)
                        seq = difflib.SequenceMatcher(None, clean_word.lower(), cand.lower())
                        if seq.ratio() > 0.6:
                            best_word = cand
                            
            # Reattach punctuation if any was stripped
            if raw_word.startswith('"'): best_word = '"' + best_word
            if raw_word.endswith('.'): best_word += '.'
            elif raw_word.endswith(','): best_word += ','
            elif raw_word.endswith('!'): best_word += '!'
            elif raw_word.endswith('?'): best_word += '?'
            elif raw_word.endswith('"'): best_word += '"'
            
            corrected.append(best_word)
            
        # Clean up space before punctuation
        out = " ".join(corrected)
        out = out.replace(" .", ".").replace(" ,", ",").replace(" !", "!").replace(" ?", "?")
        return out

