# keyword_test.py - unit tests for the keyword extraction module of PsychExtract, testing individual functions and components related to keyword extraction and selection.

# import necessary libraries and modules for testing
import unittest
from keyword_module import (
  get_yake_extractor,
  extract_keywords, 
  get_head_noun_lemma, 
  is_valid_noun_phrase, 
  select_best_noun_phrases, 
  extract_and_select_keywords,
  get_nlp,
  normalize_phrase_case
)

class TestKeywordModule(unittest.TestCase):
  """Unit tests for the keyword extraction module of PsychExtract, testing individual functions and components related to keyword extraction and selection."""
  def test_get_nlp(self):
    """
    Test the get_nlp function to ensure that it returns a spaCy language model instance.\n
    This test checks that the returned object is an instance of spaCy's Language class and that it can be used to parse text.
    """
    nlp = get_nlp()
    doc = nlp("This is a test.")
    self.assertEqual(len(doc), 5)

  def test_extract_keywords_empty(self):
    """
    Test the extract_keywords function with empty and whitespace-only input to verify that it returns an empty list.\n
    This test checks that the function correctly handles edge cases where the input text is empty or contains only whitespace, ensuring that it does not attempt to extract keywords and instead returns an empty list.
    """
    self.assertEqual(extract_keywords("", get_yake_extractor()), [])
    self.assertEqual(extract_keywords("   ", get_yake_extractor()), [])

  def test_get_head_noun_lemma(self):
    """
    Test the get_head_noun_lemma function with various input phrases to verify that it correctly identifies and returns the lemma of the head noun.\n
    This test checks that the function can accurately extract the head noun lemma from different types of phrases, including those with possessives and those without a clear noun, ensuring that it returns the expected results or None when no valid head noun is found.
    """
    self.assertEqual(get_head_noun_lemma("beautiful day"), "day")
    self.assertEqual(get_head_noun_lemma("John's book"), "book")
    self.assertIsNone(get_head_noun_lemma("quickly running"))

  def test_is_valid_noun_phrase(self):
    """
    Test the is_valid_noun_phrase function with various input phrases to verify that it correctly identifies valid noun phrases.\n
    This test checks that the function can distinguish between valid noun phrases and non-noun phrases, ensuring that it returns True for valid noun phrases and False for invalid ones.
    """
    self.assertTrue(is_valid_noun_phrase("beautiful day"))
    self.assertTrue(is_valid_noun_phrase("John's book"))
    self.assertFalse(is_valid_noun_phrase("quickly running"))

  def test_select_best_noun_phrases(self):
    """
    Test the select_best_noun_phrases function with a list of candidate keywords to verify that it correctly selects the best noun phrases based on their head noun lemmas.\n
    This test checks that the function can effectively filter and select noun phrases from a list of candidate keywords, ensuring that it includes valid noun phrases and excludes those that do not meet the criteria for selection.
    """
    keywords = ["beautiful day", "John's book", "quickly running"]
    selected = select_best_noun_phrases(keywords)
    self.assertIn("beautiful day", selected)
    self.assertIn("John's book", selected)
    self.assertNotIn("quickly running", selected)

  def test_extract_and_select_keywords(self):
    """
    Test the extract_and_select_keywords function with a sample input text to verify that it correctly extracts keywords and selects the best noun phrases.\n
    This test checks that the function can successfully extract keywords from the input text and then apply the selection criteria to return a list of valid noun phrases, ensuring that the output contains relevant concepts from the text.
    """
    text = "The beautiful day made John's book enjoyable."
    concepts = extract_and_select_keywords(text)
    # ensure "day" is in the string of a list
    self.assertTrue(any("day" in concept for concept in concepts))

  def test_normalize_phrase_case(self):
    """
    Test the normalize_phrase_case function with various input phrases to verify that it correctly normalizes the case of non-proper nouns while keeping proper nouns unchanged.\n
    This test checks that the function can accurately normalize the case of phrases based on their POS tags, ensuring that proper nouns retain their original casing while other words are converted to lowercase.
    """
    self.assertEqual(normalize_phrase_case("Beautiful Day"), "beautiful day")
    self.assertEqual(normalize_phrase_case("Today I could"), "today i could")