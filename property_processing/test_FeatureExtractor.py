import spacy
from unittest import TestCase
from property_processing import FeatureExtractor


class TestFeatureExtractor(TestCase):
    def setUp(self):
        self.feature_extractor = FeatureExtractor(feature_dict_path=
                                                  "data/features_dict.json")
        self.nlp = spacy.load("en_core_web_sm")
        self.text = "A spacious and rather elegant raised ground floor two " \
                    "bedroom apartment with two bathrooms (one en-suite) on " \
                    "this historic garden square, set within this wonderful " \
                    "stucco fronted property."

    def test_raised_exception(self):
        with self.assertRaises(ValueError):
            FeatureExtractor()

    def test_positive_call(self):
        features = self.feature_extractor(self.nlp(self.text))
        result = features["property"]
        gs_result = [["two", "bedroom"], ["raised", "ground", "floor"]]
        self.assertTrue(len(result) == len(gs_result))
        for e_1, e_2 in zip(result, gs_result):
            self.assertTrue(" ".join(e_1) == " ".join(e_2))

    def test_negative_call(self):
        features = self.feature_extractor(self.nlp(self.text))
        for e in features["property"]:
            if "elegant" in " ".join(e):
                self.fail()
        else:
            self.assertTrue(True)
