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
  def test_get_nlp(self):
    nlp = get_nlp()
    doc = nlp("This is a test.")
    self.assertEqual(len(doc), 5)

  def test_extract_keywords_empty(self):
    self.assertEqual(extract_keywords("", get_yake_extractor()), [])
    self.assertEqual(extract_keywords("   ", get_yake_extractor()), [])

  def test_get_head_noun_lemma(self):
    self.assertEqual(get_head_noun_lemma("beautiful day"), "day")
    self.assertEqual(get_head_noun_lemma("John's book"), "book")
    self.assertIsNone(get_head_noun_lemma("quickly running"))

  def test_is_valid_noun_phrase(self):
    self.assertTrue(is_valid_noun_phrase("beautiful day"))
    self.assertTrue(is_valid_noun_phrase("John's book"))
    self.assertFalse(is_valid_noun_phrase("quickly running"))

  def test_select_best_noun_phrases(self):
    keywords = ["beautiful day", "John's book", "quickly running"]
    selected = select_best_noun_phrases(keywords)
    self.assertIn("beautiful day", selected)
    self.assertIn("John's book", selected)
    self.assertNotIn("quickly running", selected)

  def test_extract_and_select_keywords(self):
    text = "The beautiful day made John's book enjoyable."
    concepts = extract_and_select_keywords(text)
    # ensure "day" is in the string of a list
    self.assertTrue(any("day" in concept for concept in concepts))

  def test_normalize_phrase_case(self):
    self.assertEqual(normalize_phrase_case("Beautiful Day"), "beautiful day")
    self.assertEqual(normalize_phrase_case("Today I could"), "today i could")