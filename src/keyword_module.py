# ! pip install yake spacy
# ! python -m spacy download en_core_web_sm
import yake 
import spacy

def get_nlp():
  return spacy.load("en_core_web_sm")

def get_yake_extractor() -> yake.KeywordExtractor:
  """
  Initializes and returns a YAKE keyword extractor with specific parameters.
  
  :return:  A configured YAKE keyword extractor instance.
  :rtype: yake.KeywordExtractor
  """
  return yake.KeywordExtractor(
    lan="en",
    n=2,
    dedupLim=0.9,
    top=10
  )

def extract_keywords(text: str, keyword_extractor: yake.KeywordExtractor) -> list:
  """
  Extracts keywords from the given text using the provided YAKE extractor.
  
  :param text: The input text from which to extract keywords.
  :type text: str
  :param yake_extractor: An instance of a YAKE keyword extractor.
  :type yake_extractor: yake.KeywordExtractor
  :return: A list of extracted keywords sorted by relevance.
  :rtype: list
  """
  if not isinstance(text, str) or not text.strip():
    return []
  keywords = keyword_extractor.extract_keywords(text)
  keywords = sorted(keywords, key=lambda x: x[1])
  return [kw for kw, score in keywords]

def get_head_noun_lemma(phrase: str) -> str:
  """
  Extracts the lemma of the head noun from a given phrase using spaCy's dependency parsing.
  
  :param phrase: The input phrase from which to extract the head noun lemma.
  :type phrase: str
  :return: The lemma of the head noun if found, otherwise None.
  :rtype: str
  """
  nlp = get_nlp()
  doc = nlp(phrase)
  for token in doc:
    if token.dep_ == "ROOT" and token.pos_ in ("NOUN", "PROPN"):
      return token.lemma_
  return None

def is_valid_noun_phrase(phrase: str) -> bool:
  """
  Validates whether a given phrase is a valid noun phrase based on its POS tags.
  
  :param phrase: The input phrase to validate as a noun phrase.
  :type phrase: str
  :return: True if the phrase is a valid noun phrase, False otherwise.
  :rtype: bool
  """
  nlp = get_nlp()
  doc = nlp(phrase)
  tokens = [t for t in doc]
  # Rule: must end in NOUN or PROPN
  if not tokens:
    return False
  return tokens[-1].pos_ in {"NOUN", "PROPN"}

def select_best_noun_phrases(keywords: list) -> list:
  """
  Selects the best noun phrases from a list of keywords based on their head noun lemmas.
  
  :param keywords: A list of keyword phrases to evaluate.
  :type keywords: list
  :return: A list of selected noun phrases that represent unique concepts.
  :rtype: list
  """
  concepts = {}
  for phrase in keywords:
    if not is_valid_noun_phrase(phrase):
      continue
    head = get_head_noun_lemma(phrase)
    if not head:
      continue
    if head not in concepts:
      concepts[head] = phrase

  return [v for v in concepts.values()]

def extract_and_select_keywords(text: str) -> list:
  """
  Extracts keywords from the input text and selects the best noun phrases representing unique concepts.

  :param text: The input text from which to extract and select keywords.
  :type text: str
  :return: A list of selected noun phrases representing unique concepts.
  :rtype: list
  """
  yake_extractor = get_yake_extractor()
  keywords = extract_keywords(text, yake_extractor)
  selected_concepts = select_best_noun_phrases(keywords)
  return selected_concepts