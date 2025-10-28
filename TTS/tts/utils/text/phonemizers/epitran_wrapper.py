"""Wrapper to call epitran for IPA phonemization."""

import logging
from typing import Dict

from TTS.tts.utils.text.phonemizers.base import BasePhonemizer
from TTS.tts.utils.text.punctuation import Punctuation

logger = logging.getLogger(__name__)


def _is_epitran_available() -> bool:
    """Check if epitran is installed."""
    try:
        import epitran
        return True
    except ImportError:
        return False


class Epitran(BasePhonemizer):
    """Wrapper for Epitran G2P (Grapheme-to-Phoneme) conversion.
    
    Uses the epitran library to convert text to IPA phonemes for various languages.
    Particularly useful for African languages like Swahili and Yoruba.
    
    Args:
        language (str):
            Language code in format 'xxx-Latn' (e.g., 'swa-Latn' for Swahili, 
            'yor-Latn' for Yoruba). See epitran documentation for supported codes.
        
        punctuations (str):
            Characters to be treated as punctuation. Defaults to Punctuation.default_puncs().
        
        keep_puncs (bool):
            If True, keep the punctuations after phonemization. Defaults to True.
    
    Example:
        >>> from TTS.tts.utils.text.phonemizers import Epitran
        >>> phonemizer = Epitran("swa-Latn")
        >>> phonemizer.phonemize("Habari yako", separator="|")
        'h|a|b|a|r|i| |j|a|k|o'
        
        >>> phonemizer = Epitran("yor-Latn")
        >>> phonemizer.phonemize("yorùbá", separator="|")
        'j|o|r|ù|b|á'
    """

    def __init__(
        self,
        language: str,
        punctuations: str = Punctuation.default_puncs(),
        keep_puncs: bool = True,
    ):
        if not _is_epitran_available():
            raise ImportError(
                "[!] epitran not found. Install it with: pip install epitran"
            )

        # language mapping example: 'swa-Latn' for Swahili, 'yor-Latn' for Yoruba
        language_mapping = {
            'swa-Latn': 'swahili',
            'yor-Latn': 'yoruba',
            'hau-Latn': 'hausa',
            'kin-Latn': 'kinyarwanda',
            'zul-Latn': 'zulu'
        }

        # Validate language code format
        if '-' not in language:
            # first check if language is in mapping
            if language not in language_mapping:
                # map to epitran code if possible
                mapped_language = {v: k for k, v in language_mapping.items()}.get(language)
                if mapped_language:
                    language = mapped_language
                else:
                    raise ValueError(
                        f"[!] Language code must be in format 'xxx-Latn' (e.g., 'swa-Latn'), got: {language}"
                    )

        super().__init__(language, punctuations=punctuations, keep_puncs=keep_puncs)
        
        # Initialize epitran for the specified language
        try:
            import epitran
            self.epi = epitran.Epitran(language)
            logger.info(f"Initialized Epitran for language: {language}")
        except Exception as e:
            raise ValueError(
                f"[!] Failed to initialize Epitran for language '{language}'. "
                f"Make sure the language code is supported. Error: {e}"
            )

    @staticmethod
    def name() -> str:
        """Return the name of the phonemizer."""
        return "epitran"

    def _phonemize(self, text: str, separator: str = "") -> str:
        """Convert input text to IPA phonemes using epitran.
        
        Args:
            text (str): Text to be converted to phonemes.
            separator (str): Separator to use between phonemes. Defaults to "".
        
        Returns:
            str: Phonemized text in IPA.
        """
        # Use epitran's transliterate method
        phonemes = self.epi.transliterate(text)
        
        # If separator is specified and not empty, insert it between characters
        if separator and separator != "":
            # Split into individual phoneme characters and join with separator
            phonemes = separator.join(phonemes)
        
        return phonemes

    @staticmethod
    def supported_languages() -> Dict[str, str]:
        """Get a dictionary of supported languages.
        
        Returns:
            Dict: Dictionary mapping language codes to language names.
        """
        # Common African languages supported by epitran
        # This is not exhaustive - epitran supports many more
        return {
            'swa-Latn': 'Swahili (Latin script)',
            'yor-Latn': 'Yoruba (Latin script)',
            'hau-Latn': 'Hausa (Latin script)',
            'kin-Latn': 'Kinyarwanda (Latin script)',
            'zul-Latn': 'Zulu (Latin script)'
            # Add more as needed
        }

    def version(self) -> str:
        """Get the version of epitran.
        
        Returns:
            str: Version of the epitran library.
        """
        try:
            import epitran
            return epitran.__version__
        except AttributeError:
            return "unknown"

    @classmethod
    def is_available(cls) -> bool:
        """Check if Epitran is available.
        
        Returns:
            bool: True if epitran is installed, False otherwise.
        """
        return _is_epitran_available()


if __name__ == "__main__":
    # Test the phonemizer
    print("Testing Epitran phonemizer...")
    print(f"Is available: {Epitran.is_available()}")
    print(f"Supported languages: {Epitran.supported_languages()}")
    
    if Epitran.is_available():
        # Test Swahili
        print("\n--- Swahili Test ---")
        epi_swa = Epitran(language="swa-Latn", keep_puncs=True)
        text_swa = "Habari yako, wewe ni mzuri."
        phonemes_swa = epi_swa.phonemize(text_swa, separator="|")
        print(f"Text: {text_swa}")
        print(f"Phonemes: {phonemes_swa}")
        
        # Test Yoruba
        print("\n--- Yoruba Test ---")
        epi_yor = Epitran(language="yor-Latn", keep_puncs=True)
        text_yor = "Báwo ni, yorùbá"
        phonemes_yor = epi_yor.phonemize(text_yor, separator="|")
        print(f"Text: {text_yor}")
        print(f"Phonemes: {phonemes_yor}")