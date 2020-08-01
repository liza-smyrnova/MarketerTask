import re
import spacy
from word2number import w2n
from property_processing import FeatureExtractor


class PropertyDescription:
    """
    This class represents a real estate/property description.
    """
    nlp = spacy.load("en_core_web_sm")

    def __init__(self, feature_extractor: FeatureExtractor,
                 file_path=None, text=None):
        """
        :param feature_extractor: this function is used to extract
                                  features/properties for nouns of interest from
                                  a given text.
        :param file_path: path to the file that contains realty description.
        :param text: realty description
        """
        if file_path is not None and text is not None:
            raise ValueError("`file_path` and `text` arguments are mutually "
                             "exclusive and can't have non-`None` value "
                             "simultaneously!")
        if file_path is None and text is None:
            raise ValueError("At least one argument should not be None!")

        if file_path is not None:
            with open(file_path) as f:
                text = f.read()
        text = re.sub(" +", " ", text)
        self.doc = PropertyDescription.nlp(text)
        self.features = feature_extractor(self.doc)

    def get_similarity(self, other: 'PropertyDescription'):
        sim = 2
        sim -= float(("garden" in self.features) ^ ("garden" in other.features))
        for feature_name in ["bathroom", "bedroom"]:
            num = {"self": -1, "other": -1}
            obj = {"self": self, "other": other}
            for e in ["self", "other"]:
                for feature in obj[e].features[feature_name]:
                    for word in feature:
                        try:
                            candidate = w2n.word_to_num(word)
                            num[e] = candidate if candidate > num[e] else num[e]
                        except ValueError:
                            continue
            if num["self"] != -1 and num["other"] != -1:
                sim -= 0.5 * abs(num["self"] - num["other"])
        return sim
