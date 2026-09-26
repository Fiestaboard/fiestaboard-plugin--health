"""Board-geometry conformance for the health plugin.

health has a fixed set of ~12 stats (allergy risk, AQI, pollen breakdown,
flu/cough/respiratory risk, PM2.5/PM10, a risk-scale legend) rather than an
unbounded feed, but that is still well more than a Flagship's 6 rows or a
Note's 3 -- a taller board must show strictly more of them, and a narrower
one must abbreviate each stat's own row rather than truncate it mid-word.
That makes ``strict_growth=True`` appropriate: this is a list of items
(one stat per row) growing with the board, not a clock with two lines to
give.
"""

import json
from pathlib import Path
from unittest.mock import Mock

from plugins.health import HealthPlugin, DEFAULT_LAT, DEFAULT_LON
from src.plugins.geometry_conformance import assert_board_conformance

MANIFEST = json.loads((Path(__file__).parent.parent / "manifest.json").read_text())


def _fake_api_response():
    return {
        "current": {
            "grass_pollen": 42,
            "birch_pollen": 12,
            "alder_pollen": 6,
            "ragweed_pollen": 7,
            "us_aqi": 145,
            "european_aqi": 78,
            "pm2_5": 32.4,
            "pm10": 55.1,
        }
    }


def _make_plugin_factory(monkeypatch):
    """Build a factory returning a fresh, network-stubbed plugin.

    The suite renders the returned plugin many times and never touches the
    network itself, so the stub is installed once here, up front. Values are
    chosen mid-scale (not all-zero) so every stat renders a non-trivial,
    non-trivially-sized value -- including a HIGH-tier allergy_risk label,
    the longest one the manifest declares.
    """
    response_payload = _fake_api_response()

    def _get(url, params=None, timeout=None):
        resp = Mock()
        resp.status_code = 200
        resp.json.return_value = response_payload
        resp.raise_for_status = Mock()
        return resp

    monkeypatch.setattr("plugins.health.requests.get", _get)

    def make_plugin() -> HealthPlugin:
        plugin = HealthPlugin(MANIFEST)
        plugin.config = {"latitude": DEFAULT_LAT, "longitude": DEFAULT_LON}
        return plugin

    return make_plugin


def test_renders_on_every_board_shape(monkeypatch):
    make_plugin = _make_plugin_factory(monkeypatch)
    assert_board_conformance(
        make_plugin,
        manifest=MANIFEST,
        strict_growth=True,
        require_note_array_preview=True,
    )
