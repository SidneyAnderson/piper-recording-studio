"""Catalog of available Piper pre-trained checkpoints from HuggingFace."""

# Base URL for checkpoint downloads
HF_BASE = "https://huggingface.co/datasets/rhasspy/piper-checkpoints/resolve/main"

# Curated catalog of available checkpoints.
# Structure: language -> locale -> voice -> quality -> {file, size_mb}
CHECKPOINTS = {
    "Arabic": {
        "ar_JO": {
            "kareem": {
                "medium": {"file": "epoch=2164-step=1355540.ckpt", "size_mb": 400},
            },
        },
    },
    "Catalan": {
        "ca_ES": {
            "upc_ona": {
                "medium": {"file": "epoch=3199-step=810400.ckpt", "size_mb": 400},
            },
        },
    },
    "Czech": {
        "cs_CZ": {
            "jirka": {
                "medium": {"file": "epoch=3199-step=608000.ckpt", "size_mb": 400},
            },
        },
    },
    "Danish": {
        "da_DK": {
            "talesyntese": {
                "medium": {"file": "epoch=3199-step=621600.ckpt", "size_mb": 400},
            },
        },
    },
    "German": {
        "de_DE": {
            "thorsten": {
                "medium": {"file": "epoch=3199-step=1377600.ckpt", "size_mb": 400},
                "high": {"file": "epoch=2218-step=838782.ckpt", "size_mb": 998},
            },
        },
    },
    "English": {
        "en_GB": {
            "cori": {
                "medium": {"file": "epoch=3399-step=204600.ckpt", "size_mb": 400},
                "high": {"file": "cori-high-500.ckpt", "size_mb": 998},
            },
            "alan": {
                "medium": {"file": "epoch=4999-step=44000.ckpt", "size_mb": 400},
            },
            "alba": {
                "medium": {"file": "epoch=4999-step=45000.ckpt", "size_mb": 400},
            },
            "jenny_dioco": {
                "medium": {"file": "epoch=2159-step=1243520.ckpt", "size_mb": 400},
            },
            "northern_english_male": {
                "medium": {"file": "epoch=1889-step=242550.ckpt", "size_mb": 400},
            },
        },
        "en_US": {
            "lessac": {
                "low": {"file": "epoch=2164-step=1355540.ckpt", "size_mb": 200},
                "medium": {"file": "epoch=2164-step=1355540.ckpt", "size_mb": 400},
                "high": {"file": "epoch=2218-step=838782.ckpt", "size_mb": 998},
            },
            "libritts_r": {
                "medium": {"file": "epoch=4149-step=2662150.ckpt", "size_mb": 400},
                "high": {"file": "epoch=2218-step=838782.ckpt", "size_mb": 998},
            },
            "arctic": {
                "medium": {"file": "epoch=4999-step=789500.ckpt", "size_mb": 400},
            },
        },
    },
    "Spanish": {
        "es_ES": {
            "carlfm": {
                "medium": {"file": "epoch=3199-step=441600.ckpt", "size_mb": 400},
            },
        },
        "es_MX": {
            "ald": {
                "medium": {"file": "epoch=3199-step=422400.ckpt", "size_mb": 400},
            },
        },
    },
    "Finnish": {
        "fi_FI": {
            "harri": {
                "medium": {"file": "epoch=3199-step=843200.ckpt", "size_mb": 400},
            },
        },
    },
    "French": {
        "fr_FR": {
            "siwis": {
                "medium": {"file": "epoch=3199-step=561600.ckpt", "size_mb": 400},
                "high": {"file": "epoch=2218-step=838782.ckpt", "size_mb": 998},
            },
        },
    },
    "Greek": {
        "el_GR": {
            "rapunzelina": {
                "medium": {"file": "epoch=3199-step=432000.ckpt", "size_mb": 400},
            },
        },
    },
    "Hungarian": {
        "hu_HU": {
            "anna": {
                "medium": {"file": "epoch=3199-step=820800.ckpt", "size_mb": 400},
            },
        },
    },
    "Italian": {
        "it_IT": {
            "riccardo": {
                "medium": {"file": "epoch=3199-step=1612800.ckpt", "size_mb": 400},
            },
        },
    },
    "Kazakh": {
        "kk_KZ": {
            "issai": {
                "medium": {"file": "epoch=3199-step=561600.ckpt", "size_mb": 400},
            },
        },
    },
    "Nepali": {
        "ne_NP": {
            "google": {
                "medium": {"file": "epoch=3199-step=518400.ckpt", "size_mb": 400},
            },
        },
    },
    "Dutch": {
        "nl_NL": {
            "mls": {
                "medium": {"file": "epoch=3199-step=505600.ckpt", "size_mb": 400},
            },
        },
    },
    "Norwegian": {
        "no_NO": {
            "talesyntese": {
                "medium": {"file": "epoch=3199-step=604800.ckpt", "size_mb": 400},
            },
        },
    },
    "Polish": {
        "pl_PL": {
            "gosia": {
                "medium": {"file": "epoch=3199-step=432000.ckpt", "size_mb": 400},
            },
        },
    },
    "Portuguese": {
        "pt_BR": {
            "faber": {
                "medium": {"file": "epoch=3199-step=1459200.ckpt", "size_mb": 400},
            },
        },
    },
    "Romanian": {
        "ro_RO": {
            "mihai": {
                "medium": {"file": "epoch=3199-step=309600.ckpt", "size_mb": 400},
            },
        },
    },
    "Russian": {
        "ru_RU": {
            "irina": {
                "medium": {"file": "epoch=3199-step=1622400.ckpt", "size_mb": 400},
            },
        },
    },
    "Slovak": {
        "sk_SK": {
            "lili": {
                "medium": {"file": "epoch=3199-step=399600.ckpt", "size_mb": 400},
            },
        },
    },
    "Serbian": {
        "sr_RS": {
            "serbski_institut": {
                "medium": {"file": "epoch=3199-step=505600.ckpt", "size_mb": 400},
            },
        },
    },
    "Swedish": {
        "sv_SE": {
            "nst": {
                "medium": {"file": "epoch=3199-step=798400.ckpt", "size_mb": 400},
            },
        },
    },
    "Swahili": {
        "sw_CD": {
            "lanfrica": {
                "medium": {"file": "epoch=3199-step=432000.ckpt", "size_mb": 400},
            },
        },
    },
    "Turkish": {
        "tr_TR": {
            "dfki": {
                "medium": {"file": "epoch=3199-step=432000.ckpt", "size_mb": 400},
            },
        },
    },
    "Ukrainian": {
        "uk_UA": {
            "ukrainian_tts": {
                "medium": {"file": "epoch=3199-step=1459200.ckpt", "size_mb": 400},
            },
        },
    },
    "Vietnamese": {
        "vi_VN": {
            "vais1000": {
                "medium": {"file": "epoch=3199-step=686400.ckpt", "size_mb": 400},
            },
        },
    },
    "Chinese": {
        "zh_CN": {
            "huayan": {
                "medium": {"file": "epoch=3199-step=1118400.ckpt", "size_mb": 400},
            },
        },
    },
}

QUALITY_INFO = {
    "low": {"sample_rate": 16000, "description": "Fastest inference, smaller model"},
    "medium": {"sample_rate": 22050, "description": "Good balance of speed and quality"},
    "high": {"sample_rate": 22050, "description": "Best quality, larger model"},
}


def get_checkpoint_url(locale: str, voice: str, quality: str) -> str | None:
    """Build the download URL for a checkpoint."""
    for lang_name, locales in CHECKPOINTS.items():
        if locale in locales:
            voices = locales[locale]
            if voice in voices and quality in voices[voice]:
                info = voices[voice][quality]
                return f"{HF_BASE}/{locale[:2]}/{locale}/{voice}/{quality}/{info['file']}"
    return None


def get_checkpoint_filename(locale: str, voice: str, quality: str) -> str:
    """Return a descriptive filename for the checkpoint."""
    return f"{locale}-{voice}-{quality}.ckpt"


def list_all() -> list[dict]:
    """Return a flat list of all checkpoints for the UI."""
    result = []
    for lang_name, locales in CHECKPOINTS.items():
        for locale, voices in locales.items():
            for voice, qualities in voices.items():
                for quality, info in qualities.items():
                    result.append({
                        "language": lang_name,
                        "locale": locale,
                        "voice": voice,
                        "quality": quality,
                        "size_mb": info["size_mb"],
                        "sample_rate": QUALITY_INFO[quality]["sample_rate"],
                        "description": QUALITY_INFO[quality]["description"],
                    })
    return result
