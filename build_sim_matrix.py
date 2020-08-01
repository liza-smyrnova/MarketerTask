import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from property_processing import FeatureExtractor
from property_processing import PropertyDescription


def save_sim_matrix(sim_matrix, image_file_path, text_file_path):
    np.savetxt(text_file_path, sim_matrix, fmt="%1.3f")
    fig, ax = plt.subplots()
    cax = ax.matshow(sim_matrix, cmap="RdYlGn")
    plt.title("Similarity matrix")
    plt.xticks(range(5), labels)
    plt.yticks(range(5), labels)
    fig.colorbar(cax, ticks=[-2, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0])
    plt.savefig(image_file_path, dpi=400)


if __name__ == '__main__':
    feature_extractor = FeatureExtractor(feature_dict_path=
                                         "data/features_dict.json")
    descriptions = dict()
    for file_path in glob.glob("data/properties/*.txt"):
        p_description = PropertyDescription(feature_extractor,
                                            file_path=file_path)
        base_name = os.path.basename(file_path)
        base_name = os.path.splitext(base_name)[0]
        descriptions[base_name] = p_description

    sim_matrix = np.empty((len(descriptions), len(descriptions)))
    labels = list(descriptions.keys())
    for i, (description_name_i, description_i) in enumerate(descriptions.items()):
        for j, (description_name_j, description_j) in enumerate(descriptions.items()):
            sim_matrix[i, j] = description_i.get_similarity(description_j)
    save_sim_matrix(sim_matrix, image_file_path="data/sim_matrix.png",
                    text_file_path="data/sim_matrix.txt")
