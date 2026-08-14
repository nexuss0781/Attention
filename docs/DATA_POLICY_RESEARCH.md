# Stage 4 Data-Policy Research Findings

The official PG-19 repository describes PG-19 as a language-modeling benchmark built from Project Gutenberg books published before 1919, with train, validation, and test partitions and metadata including book IDs, titles, and publication dates. It states that only license boilerplate was removed and that some offensive discriminatory words were mapped to placeholders. The repository also cautions that PG-19 is not recommended as the sole source for a general-purpose production dialogue model because of dated language and historical biases. Source: https://github.com/google-deepmind/pg19

TensorFlow Datasets documents PG-19 version 0.1.1, its 28,602/50/100 train-validation-test split, and fields including book ID, link, text, title, and publication date. Source: https://www.tensorflow.org/datasets/catalog/pg19

Creative Commons explains that CC BY-SA 4.0 permits sharing and adaptation, including commercially, provided appropriate attribution, a license link, change indication, and ShareAlike compliance are preserved. The deed is only a summary; the full legal code must be reviewed for a release decision. Source: https://creativecommons.org/licenses/by-sa/4.0/deed.en

Project Gutenberg states that users outside the United States must check their local copyright law and that some hosted material may be restricted outside the United States. It also directs automated bulk users to mirrors and catalog feeds rather than repeatedly downloading from the main site. Source: https://www.gutenberg.org/policy/terms_of_use.html

Policy implication: the repository may record PG-19 as an evaluation/reference source with explicit provenance and license review, but it must not silently treat every source text as globally public-domain training data. The first ingestion implementation should accept only manifest entries with documented license status, source identifier, retrieval timestamp, checksum, permitted use, and exclusion/review status. Sensitive, disallowed, corrupted, or license-uncertain material is rejected or held out pending review.
