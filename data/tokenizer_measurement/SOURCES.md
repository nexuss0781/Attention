# Tokenizer Measurement Sources

The English sample is taken from the United Nations Universal Declaration of Human Rights page: https://www.un.org/en/about-us/universal-declaration-of-human-rights

The Amharic sample is the official OHCHR/National Commission for UNESCO, Ethiopia translation PDF: https://www.ohchr.org/sites/default/files/UDHR/Documents/UDHR_Translations/amh.pdf

The measurement harness records source URLs, byte counts, UTF-8 validity, encoded token counts, bytes-per-token, and unknown-token counts. The byte-level tokenizer has no unknown-token path for valid raw bytes, so an unknown rate of zero is an implementation property rather than a quality claim.
