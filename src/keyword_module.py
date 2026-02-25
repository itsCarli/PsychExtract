# keyword_module.py - module for keyword extraction and noun phrase selection using YAKE and spaCy

# import necessary libraries
import yake 
import spacy

def get_nlp() -> spacy.language.Language:
  """
  Loads and returns the spaCy English language model for natural language processing tasks.
  Returns:
    spacy.language.Language: The loaded spaCy English language model.
  """
  return spacy.load("en_core_web_sm")

def get_yake_extractor() -> yake.KeywordExtractor:
  """
  Initializes and returns a YAKE keyword extractor with specific parameters.
  Returns:
    yake.KeywordExtractor: A configured YAKE keyword extractor instance.
  """
  # language is english, 
  # n=2 (up to bigrams), 
  # deduplication threshold=0.9, 
  # top 10 keywords
  return yake.KeywordExtractor(
    lan="en",
    n=2,
    dedupLim=0.9,
    top=10
  )

def extract_keywords(text: str, keyword_extractor: yake.KeywordExtractor) -> list[str]:
  """
  Extracts keywords from the given text using the provided YAKE extractor.
  Args:
    text (str): The input text from which to extract keywords.
    keyword_extractor (yake.KeywordExtractor): An instance of a YAKE keyword extractor to use for extraction.
  Returns:
    list[str]: A list of extracted keywords sorted by relevance.
  """
  if not isinstance(text, str) or not text.strip():
    # if text is empty or not a string, return empty list
    return []
  keywords = keyword_extractor.extract_keywords(text)
  # sort keywords by score (second element of tuple), lowest score is best
  keywords = sorted(keywords, key=lambda x: x[1])
  return [kw for kw, _ in keywords]

def get_head_noun_lemma(phrase: str) -> str:
  """
  Extracts the lemma of the head noun from a given phrase using spaCy's dependency parsing.
  Args:
    phrase (str): The input phrase from which to extract the head noun lemma.
  Returns:
    str: The lemma of the head noun if found, otherwise None.
  """
  # load spaCy model
  nlp = get_nlp() 
  # parse the phrase
  doc = nlp(phrase)
  # look for the ROOT token that is a NOUN or PROPN, and return its lemma
  for token in doc:
    if token.dep_ == "ROOT" and token.pos_ in ("NOUN", "PROPN"):
      return token.lemma_
  return None

def is_valid_noun_phrase(phrase: str) -> bool:
  """
  Validates whether a given phrase is a valid noun phrase based on its POS tags.
  Args:
    phrase (str): The input phrase to validate as a noun phrase.
  Returns:
    bool: True if the phrase is a valid noun phrase, False otherwise.
  """
  # load spaCy model
  nlp = get_nlp()
  # parse the phrase
  doc = nlp(phrase)
  # convert to list of tokens for easier processing
  tokens = [t for t in doc]
  # a valid noun phrase should have at least one token and end with a NOUN or PROPN
  if not tokens:
    return False
  return tokens[-1].pos_ in {"NOUN", "PROPN"}

def normalize_phrase_case(phrase: str) -> str:
  """
  Normalizes the case of a phrase by converting non-proper nouns to lowercase while keeping proper nouns unchanged.
  Args:
    phrase (str): The input phrase to normalize.
  Returns:
    str: The normalized phrase with proper nouns unchanged and others in lowercase.
  """
  # load spaCy model
  nlp = get_nlp()
  # parse the phrase
  doc = nlp(phrase)
  # iterate through tokens and normalize case based on POS tags
  normalized = ""
  for token in doc:
    if token.pos_ == "PROPN":
      # keep original casing
      normalized += token.text_with_ws
    else:
      # lowercase everything else, preserve original spacing
      normalized += token.text.lower() + token.whitespace_
  return normalized.strip()

def select_best_noun_phrases(keywords: list[str]) -> list[str]:
  """
  Selects the best noun phrases from a list of keywords based on their head noun lemmas.
  Args:
    keywords (list[str]): A list of keyword phrases to evaluate.
  Returns:
    list[str]: A list of selected noun phrases that represent unique concepts.
  """
  concepts = {}
  for phrase in keywords:
    if not is_valid_noun_phrase(phrase):
      # skip phrases that are not valid noun phrases
      continue
    head = get_head_noun_lemma(phrase)
    if not head:
      # if head noun lemma not found, skip this phrase
      continue
    if head not in concepts:
      # if this head noun lemma is not yet in concepts, add it
      concepts[head] = normalize_phrase_case(phrase)
  # return the list of unique noun phrases representing concepts
  return [v for v in concepts.values()]

def extract_and_select_keywords(text: str) -> list[str]:
  """
  Extracts keywords from the input text and selects the best noun phrases representing unique concepts.
  Args:
    text (str): The input text from which to extract and select keywords.
  Returns:
    list[str]: A list of selected noun phrases representing unique concepts.
  """
  yake_extractor = get_yake_extractor()
  keywords = extract_keywords(text, yake_extractor)
  return select_best_noun_phrases(keywords)