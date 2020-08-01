# Realty Feature Extraction and Comparison

The goal of this assignment is to extract adjectives and numbers that apply to
nouns of interest. However, it seems that the spirit of the task is not exactly
about just extracting relevant adjectives and numbers but it is more general.
Working on this assignment my assumption was that the task is concerned with 
identifying features/properties of a particular real estate given its 
description along different dimensions which are defined by the nouns of 
interest. (e.g. I extracted ``raised ground floor`` as a feature of `property` 
location instead of just adjective `spacious`.)

## SubTask 1:
For this sub-task, I've mainly used a [spaCy](https://spacy.io/) library in 
order to use the dependency parser.

Running `python process.py` will read property description from 
`data/properties` folder and extract adjectives/numbers and other relevant 
features to the `data/features/` folder.
The provided list of nouns of interest is stored at `data/features_dict.json`.
I made just two modifications to this list, namely added `address` as a possible
expression of the noun `location` and `shower room` for the noun `bathroom`.
Naturally, further modifications are possible.

### Example
For example the following description:
```text
A spacious and rather elegant raised ground floor two bedroom apartment with two bathrooms (one en-suite) on this historic garden square.
```
will be converted to:
```json
{
  "bathroom": [
    ["two"]
  ],
  "property": [
    ["spacious"],
    ["two", "bedroom"],
    ["raised", "ground", "floor"]
  ]
}
```

## SubTask 2:
To build a similarity matrix run `python build_sim_matrix.py`. The visualization 
and a textual file will be saved to `data/sim_matrix.png` and 
`data/sim_matrix.txt` correspondingly. From the `png`, file one can see that 
there are two clusters of properties: properties #`1,2,3` and #`4,5`.
The numbers of `bathrooms`, `bedrooms` and the presence of the `garden` were 
used to compute the similarity score.
