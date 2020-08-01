import os
import glob
import json
from property_processing import FeatureExtractor
from property_processing import PropertyDescription


if __name__ == '__main__':
    feature_extractor = FeatureExtractor(feature_dict_path=
                                         "data/features_dict.json")

    for file_path in glob.glob("data/properties/*.txt"):
        p_description = PropertyDescription(feature_extractor,
                                            file_path=file_path)
        base_name = os.path.basename(file_path)
        out_file_path = f"data/features/{os.path.splitext(base_name)[0]}.json"
        with open(out_file_path, "w") as f:
            json.dump(p_description.features, f, sort_keys=True, indent=2)
