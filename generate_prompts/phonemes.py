"""English phoneme coverage data for TTS training prompt generation.

Defines phoneme inventory, common diphones, and challenge words to ensure
generated training data covers the full range of English sounds.
"""

# Core English phonemes (IPA-ish notation for reference, actual usage is in
# word/sentence templates that naturally produce these sounds)

CONSONANTS = [
    "p", "b", "t", "d", "k", "g",       # stops
    "f", "v", "θ", "ð", "s", "z",       # fricatives
    "ʃ", "ʒ", "h",                       # fricatives cont.
    "tʃ", "dʒ",                          # affricates
    "m", "n", "ŋ",                       # nasals
    "l", "r", "w", "j",                  # approximants
]

VOWELS = [
    "iː", "ɪ", "e", "æ", "ɑː", "ɒ",   # monophthongs
    "ɔː", "ʊ", "uː", "ʌ", "ɜː", "ə",  # monophthongs cont.
    "eɪ", "aɪ", "ɔɪ",                   # diphthongs
    "aʊ", "əʊ", "ɪə", "eə", "ʊə",     # diphthongs cont.
]

# Words that contain challenging or rare sound combinations
CHALLENGE_WORDS = [
    # Consonant clusters
    "strengths", "sixths", "twelfths", "glimpsed", "prompts",
    "sculpts", "crisps", "warmths", "lengths", "depths",
    "scripts", "disrupts", "erupts", "attempts", "exempts",
    # Unusual phoneme sequences
    "clothes", "rhythm", "algorithm", "isthmus", "asthma",
    "chrysanthemum", "entrepreneurial", "anemone", "onomatopoeia",
    "surveillance", "questionnaire", "bourgeois", "lingerie",
    # British-specific pronunciations
    "lieutenant", "aluminium", "controversy", "schedule", "garage",
    "privacy", "vitamin", "tomato", "advertisement", "leisure",
    "neither", "harassment", "laboratory", "oregano", "yoghurt",
    # Numbers and dates (important for TTS)
    "first", "second", "third", "twelfth", "twentieth",
    "hundred", "thousand", "million", "billion",
    # Common function words (prosody markers)
    "however", "therefore", "nevertheless", "furthermore", "consequently",
    "although", "meanwhile", "nonetheless", "accordingly", "subsequently",
]

# Sentence patterns that produce different prosodic contours
PROSODY_PATTERNS = [
    "declarative",       # Falling intonation
    "yes_no_question",   # Rising intonation
    "wh_question",       # Falling intonation on content words
    "exclamation",       # Emphatic stress
    "imperative",        # Command intonation
    "list",              # Series intonation
    "conditional",       # If-then pattern
    "contrast",          # But/however pattern
    "parenthetical",     # Aside/embedded clause
    "tag_question",      # Statement + rising tag
]

# Categories for topical diversity
CATEGORIES = [
    "daily_life",
    "weather_nature",
    "food_cooking",
    "travel_directions",
    "technology",
    "health_wellness",
    "sports_hobbies",
    "education_learning",
    "business_finance",
    "arts_entertainment",
    "science",
    "history_culture",
    "emotions_social",
    "home_garden",
    "news_current_affairs",
    "british_culture",
    "numbers_dates_times",
    "phone_addresses",
]
