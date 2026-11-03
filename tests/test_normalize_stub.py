"""``normalize_image`` invariants and StubDetector parity with Contract 2."""
from __future__ import annotations

import numpy as np
from PIL import Image

from conftest import EXAMPLES
from test_contract import check


def _photo(w=640, h=480) -> Image.Image:
    return Image.open(EXAMPLES / "ai_veo_3_stone_villa.jpg").convert("RGB").resize((w, h))


def test_normalize_image_is_deterministic_224():
    from truthlens.features import normalize_image

    im = _photo()
    a, box_a, _ = normalize_image(im)
    b, box_b, _ = normalize_image(im)
    assert a.size == (224, 224) and a.mode == "RGB"
    assert box_a == box_b == (80, 0, 560, 480)
    assert np.array_equal(np.asarray(a), np.asarray(b))


def test_normalize_matches_app_box():
    from truthlens.features import normalize_image
    from truthlens.ui import analysed_box

    for w, h in [(640, 480), (300, 900), (224, 224), (1001, 333)]:
        assert normalize_image(_photo(w, h))[1] == analysed_box((w, h))


def test_stub_matches_contract_and_is_deterministic():
    from truthlens.stub import StubDetector

    det = StubDetector()
    im = _photo(500, 300)
    a, b = det.predict(im), det.predict(im)
    check(a, im.size)
    assert a["prob_ai"] == b["prob_ai"] and a["verdict"] == b["verdict"]
