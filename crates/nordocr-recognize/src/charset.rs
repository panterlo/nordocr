/// Nordic-aware character set for PARSeq text recognition.
///
/// The charset covers standard ASCII printable characters plus the
/// Nordic diacritical characters that are distinct letters (not accented
/// variants) in Swedish, Norwegian, Danish, Finnish, and Icelandic.

/// Full character set used by the recognition model.
/// Order must match the model's output vocabulary exactly.
pub static CHARSET: &[char] = &[
    // Special tokens (indices 0-2).
    '\0', // [PAD]
    '\u{FFFD}', // [UNK]
    '\n', // [EOS]
    // ASCII digits (indices 3-12).
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    // ASCII uppercase (indices 13-38).
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
    // ASCII lowercase (indices 39-64).
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
    'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
    // Nordic uppercase (indices 65-72).
    'Å', 'Ä', 'Ö', // Swedish, Finnish
    'Ø', 'Æ',       // Norwegian, Danish
    'Ð', 'Þ',       // Icelandic
    'Ü',             // Finnish (loan words), German influence
    // Nordic lowercase (indices 73-80).
    'å', 'ä', 'ö', // Swedish, Finnish
    'ø', 'æ',       // Norwegian, Danish
    'ð', 'þ',       // Icelandic
    'ü',             // Finnish (loan words)
    // Common punctuation (indices 81+).
    ' ', '!', '"', '#', '$', '%', '&', '\'', '(', ')', '*', '+', ',',
    '-', '.', '/', ':', ';', '<', '=', '>', '?', '@', '[', '\\', ']',
    '^', '_', '`', '{', '|', '}', '~',
    // Additional Nordic/European punctuation.
    '§', '°', '€', '£', '«', '»', '–', '—', '\u{2018}', '\u{2019}', '\u{201C}', '\u{201D}',
];

pub const PAD_TOKEN: usize = 0;
pub const UNK_TOKEN: usize = 1;
pub const EOS_TOKEN: usize = 2;
pub const CHARSET_SIZE: usize = 126; // update if CHARSET changes

// --- CTC charset for SVTRv2 ---
// CTC uses blank at index 0, then 125 characters from nordic_dict.txt at indices 1-125.
// Total classes: 126 (blank + 125 chars).
pub const CTC_BLANK: usize = 0;
pub const CTC_NUM_CLASSES: usize = 126;

/// CTC character set (indices 1-125). Index 0 is the CTC blank token.
/// Order matches `training/recognize/nordic_dict.txt`.
pub static CTC_CHARSET: &[char] = &[
    // Digits (CTC indices 1-10)
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    // Uppercase (CTC indices 11-36)
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
    // Lowercase (CTC indices 37-62)
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
    'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
    // Nordic uppercase (CTC indices 63-70)
    'Å', 'Ä', 'Ö', 'Ø', 'Æ', 'Ð', 'Þ', 'Ü',
    // Nordic lowercase (CTC indices 71-78)
    'å', 'ä', 'ö', 'ø', 'æ', 'ð', 'þ', 'ü',
    // Punctuation (CTC indices 79-111)
    ' ', '!', '"', '#', '$', '%', '&', '\'', '(', ')', '*', '+', ',',
    '-', '.', '/', ':', ';', '<', '=', '>', '?', '@', '[', '\\', ']',
    '^', '_', '`', '{', '|', '}', '~',
    // Additional symbols (CTC indices 112-125)
    '§', '°', '€', '£', '«', '»', '–', '—',
    '\u{2018}', '\u{2019}', '\u{201C}', '\u{201D}',
    '±', '×',
];

/// Map a CTC token index to a character. Returns None for blank (0) or out-of-range.
pub fn ctc_token_to_char(token: usize) -> Option<char> {
    if token == CTC_BLANK || token > CTC_CHARSET.len() {
        None
    } else {
        Some(CTC_CHARSET[token - 1])
    }
}

/// Look up the character for a given token index.
pub fn token_to_char(token: usize) -> Option<char> {
    CHARSET.get(token).copied()
}

/// Look up the token index for a given character.
pub fn char_to_token(c: char) -> Option<usize> {
    CHARSET.iter().position(|&ch| ch == c)
}

/// Check if a character is a Nordic diacritical letter.
pub fn is_nordic_diacritical(c: char) -> bool {
    matches!(
        c,
        'å' | 'ä' | 'ö' | 'ø' | 'æ' | 'ð' | 'þ'
            | 'Å' | 'Ä' | 'Ö' | 'Ø' | 'Æ' | 'Ð' | 'Þ'
    )
}

/// Nordic diacritical confusion pairs for evaluation.
/// These are commonly confused by OCR systems but are distinct letters.
pub static DIACRITICAL_PAIRS: &[(char, char)] = &[
    ('a', 'å'),
    ('a', 'ä'),
    ('å', 'ä'),
    ('o', 'ö'),
    ('o', 'ø'),
    ('ö', 'ø'),
    ('A', 'Å'),
    ('A', 'Ä'),
    ('Å', 'Ä'),
    ('O', 'Ö'),
    ('O', 'Ø'),
    ('Ö', 'Ø'),
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_round_trip() {
        for (i, &c) in CHARSET.iter().enumerate() {
            assert_eq!(token_to_char(i), Some(c));
            if i >= 3 {
                // Skip special tokens — they might not round-trip uniquely.
                assert_eq!(char_to_token(c), Some(i));
            }
        }
    }

    #[test]
    fn test_nordic_diacriticals_present() {
        for c in ['å', 'ä', 'ö', 'ø', 'æ', 'Å', 'Ä', 'Ö', 'Ø', 'Æ'] {
            assert!(
                char_to_token(c).is_some(),
                "Nordic character '{c}' missing from charset"
            );
        }
    }

    #[test]
    fn test_is_nordic() {
        assert!(is_nordic_diacritical('å'));
        assert!(is_nordic_diacritical('Ö'));
        assert!(!is_nordic_diacritical('a'));
        assert!(!is_nordic_diacritical('Z'));
    }
}
