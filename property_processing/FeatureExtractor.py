import json
import spacy
from itertools import chain
from collections import defaultdict
from spacy.matcher import PhraseMatcher


class FeatureExtractor:
    nlp = spacy.load("en_core_web_sm")

    def __init__(self, feature_dict=None, feature_dict_path=None):
        """
        :param feature_dict: dictionary that contains mapping between nouns of
                             interest and their possible expressions in text.
        :param feature_dict_path: the path to json file that contains
                                  `feature_dict`.
        """
        if feature_dict is not None and feature_dict_path is not None:
            raise ValueError("`feature_dict` and `feature_dict_path` arguments "
                             "are mutually exclusive and can't have non-`None` "
                             "value simultaneously!")
        if feature_dict is None and feature_dict_path is None:
            raise ValueError("At least one argument should not be None!")

        if feature_dict_path is not None:
            with open(feature_dict_path) as json_file:
                feature_dict = json.load(json_file)
        self.feature_dict = feature_dict
        self.phrase_matcher = FeatureExtractor.get_phrase_matcher(feature_dict)

    def __call__(self, doc: spacy.tokens.Doc):
        """
        This method iterates through the document `doc` and extracts
        features/properties for matched nouns of interest.
        """
        name_to_features = defaultdict(list)
        for match_id, start, end in self.phrase_matcher(doc):
            noun_match = {"name": FeatureExtractor.nlp.vocab.strings[match_id],
                          "start": start,
                          "end": end}
            for feature in list(FeatureExtractor.get_features(doc[end - 1],
                                                              noun_match)):
                name_to_features[noun_match["name"]].append(feature)
        return name_to_features

    @staticmethod
    def get_features(token, noun_match, include_token=False):
        """
        Returns features/properties for a given token, which has been previously
        matched by the `self.phrase_matcher`
        :param token: spacy.tokens.Token that corresponds to the left-most token
                      from the matched phrase.
        :param noun_match: contains information about the corresponding noun
                           of interest, such as name, start and end of the
                           match.
        :param include_token: indicates whether the current token should be
                              included as a feature.
        """
        modifiers = list(FeatureExtractor.get_modifiers(token))
        compound_chain = list(FeatureExtractor.get_compound_chain(token, noun_match))
        modifiers = sorted(chain(modifiers, compound_chain,
                                 [token] if include_token else []),
                           key=lambda c: c.i)
        if modifiers:
            yield [e.text.lower() for e in modifiers]

        for child in token.children:
            if child.dep_ == "nmod":
                yield from FeatureExtractor.get_features(child, noun_match,
                                                         include_token=True)
        if token.dep_ == "nmod" and not modifiers:
            modifiers = FeatureExtractor.get_modifiers(token.head)
            yield [e.text.lower() for e in modifiers]

    @staticmethod
    def get_modifiers(token):
        """
        This method returns modifiers for a given `token` such as adjectives,
        numerals, past participles and other combination of tags and
        dependencies that can be treated as a modifier for a noun of interest.
        """
        modifier_tags = {"JJ", "CD", "VBN", "NNP"}
        modifier_deps = {"nummod", "amod", "advmod", "npadvmod"}
        for child in token.children:
            if child.dep_ in modifier_deps and \
                    (child.tag_ in modifier_tags or
                     (child.tag_ in {"RB", "NN"} and token.tag_ in modifier_tags)):
                yield child
                yield from FeatureExtractor.get_modifiers(child)

    @staticmethod
    def get_compound_chain(token, noun_match):
        """
        This method has similar purpose to `FeatureExtractor.get_modifiers` but
        also extracts chain of compound nouns that modify a given `token`.
        """
        for child in token.children:
            if child.dep_ == "compound":
                if child.i < noun_match["start"] or \
                        child.i >= noun_match["end"]:
                    yield child
                    yield from FeatureExtractor.get_modifiers(child)
                yield from FeatureExtractor.get_compound_chain(child, noun_match)

    @staticmethod
    def get_phrase_matcher(feature_dict):
        """
        This method creates a spaCy PhraseMatcher for each noun of interest,
        which will later be used for extracting properties of such nouns.
        :param feature_dict: dictionary that contains mapping between nouns of
                             interest and their possible expressions in text.
        """
        phrase_matcher = PhraseMatcher(FeatureExtractor.nlp.vocab, attr="LEMMA")
        for feature_name, noun_phrases in feature_dict.items():
            noun_phrases = [FeatureExtractor.nlp(e) for e in noun_phrases]
            phrase_matcher.add(feature_name, noun_phrases)
        return phrase_matcher
