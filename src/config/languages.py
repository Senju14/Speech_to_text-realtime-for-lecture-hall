"""
Language code mappings for multi-language support

Maps between:
- ISO 639-1 codes (used by Whisper/WhisperX): "vi", "en", "ja", ...
- NLLB-200 BCP-47 codes: "vie_Latn", "eng_Latn", "jpn_Jpan", ...
- Display names for frontend
"""

# ISO → NLLB language code mapping
# Full list: https://github.com/facebookresearch/flores/blob/main/flores200/README.md
LANG_TO_NLLB = {
    "vi": "vie_Latn",
    "en": "eng_Latn",
    "zh": "zho_Hans",
    "ja": "jpn_Jpan",
    "ko": "kor_Hang",
    "fr": "fra_Latn",
    "de": "deu_Latn",
    "es": "spa_Latn",
    "pt": "por_Latn",
    "ru": "rus_Cyrl",
    "th": "tha_Thai",
    "id": "ind_Latn",
    "ms": "zsm_Latn",
    "ar": "arb_Arab",
    "hi": "hin_Deva",
    "it": "ita_Latn",
    "nl": "nld_Latn",
    "pl": "pol_Latn",
    "tr": "tur_Latn",
    "uk": "ukr_Cyrl",
    "cs": "ces_Latn",
    "sv": "swe_Latn",
    "da": "dan_Latn",
    "fi": "fin_Latn",
    "el": "ell_Grek",
    "he": "heb_Hebr",
    "ro": "ron_Latn",
    "hu": "hun_Latn",
    "no": "nob_Latn",
    "bg": "bul_Cyrl",
    "ca": "cat_Latn",
    "hr": "hrv_Latn",
    "sk": "slk_Latn",
    "tl": "tgl_Latn",
    "my": "mya_Mymr",
    "km": "khm_Khmr",
    "lo": "lao_Laoo",
}

# Display names (for API responses; frontend has its own labels)
LANG_NAMES = {
    "vi": "Vietnamese",
    "en": "English",
    "zh": "Chinese",
    "ja": "Japanese",
    "ko": "Korean",
    "fr": "French",
    "de": "German",
    "es": "Spanish",
    "pt": "Portuguese",
    "ru": "Russian",
    "th": "Thai",
    "id": "Indonesian",
    "ms": "Malay",
    "ar": "Arabic",
    "hi": "Hindi",
    "it": "Italian",
    "nl": "Dutch",
    "pl": "Polish",
    "tr": "Turkish",
    "uk": "Ukrainian",
    "cs": "Czech",
    "sv": "Swedish",
    "da": "Danish",
    "fi": "Finnish",
    "el": "Greek",
    "he": "Hebrew",
    "ro": "Romanian",
    "hu": "Hungarian",
    "no": "Norwegian",
    "bg": "Bulgarian",
    "ca": "Catalan",
    "hr": "Croatian",
    "sk": "Slovak",
    "tl": "Filipino",
    "my": "Burmese",
    "km": "Khmer",
    "lo": "Lao",
}

# Languages that WhisperX alignment supports (wav2vec2-based)
# If a language is not here, we skip word-level alignment
WHISPERX_ALIGN_LANGUAGES = {
    "vi", "en", "zh", "ja", "ko", "fr", "de", "es", "pt", "ru",
    "th", "id", "ar", "hi", "it", "nl", "pl", "tr", "uk", "cs",
    "da", "fi", "el", "he", "ro", "hu", "no", "ca",
}


def iso_to_nllb(lang_code: str) -> str:
    """Convert ISO 639-1 code to NLLB code. Returns eng_Latn if unknown."""
    if not lang_code:
        return "eng_Latn"
    return LANG_TO_NLLB.get(lang_code, "eng_Latn")


def is_supported_language(lang_code: str) -> bool:
    """Check if an ISO language code is supported."""
    return lang_code in LANG_TO_NLLB


def supports_alignment(lang_code: str) -> bool:
    """Check if WhisperX alignment is available for this language."""
    return lang_code in WHISPERX_ALIGN_LANGUAGES
