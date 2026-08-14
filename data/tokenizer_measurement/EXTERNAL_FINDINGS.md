# External Findings for Tokenizer Measurements

The United Nations Universal Declaration of Human Rights page provides the English source text used for measurement: https://www.un.org/en/about-us/universal-declaration-of-human-rights. The page contains the preamble and Articles 1–30.

The OHCHR Amharic translation page identifies the Amharic source as the National Commission for UNESCO, Ethiopia and links the official PDF at https://www.ohchr.org/sites/default/files/UDHR/Documents/UDHR_Translations/amh.pdf. The PDF is six pages and uses a custom font encoding that `pdftotext` cannot recover as Unicode Amharic; its extracted text contains zero Ethiopic Unicode characters. It remains recorded as authoritative provenance but is not used as a directly extracted measurement text until a reliable OCR/text extraction path is available.

For a directly extractable secondary Amharic sample, the Amharic Wikipedia Ethiopia article was retrieved from https://am.wikipedia.org/wiki/ኢትዮጵያ. The extracted content contains substantial Ethiopic-script prose, including sections on Ethiopia, geography, history, language, and society. It is suitable for a transparent exploratory measurement sample, but it is not a representative corpus and must not be treated as a quality benchmark.

The tokenizer measurements will report bytes, UTF-8 validity, byte-token count, bytes per token, and unknown-token count. Because the current tokenizer maps every valid input byte directly, unknown rate is expected to be zero by construction and is not evidence of semantic quality.
