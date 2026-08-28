"""Import được toàn bộ package và các script không nổ ở mức module."""

from __future__ import annotations

import importlib

import pytest

MODULES = [
    "luna_zero",
    "luna_zero.config",
    "luna_zero.data",
    "luna_zero.tokenizer",
    "luna_zero.model",
    "luna_zero.train",
]


@pytest.mark.parametrize("name", MODULES)
def test_import(name: str) -> None:
    assert importlib.import_module(name) is not None


def test_special_token_ids_khop_voi_thu_tu_khai_bao() -> None:
    from luna_zero import config

    assert config.SPECIAL_TOKENS.index(config.PAD_TOKEN) == config.PAD_ID
    assert config.SPECIAL_TOKENS.index(config.BOS_TOKEN) == config.BOS_ID
    assert config.SPECIAL_TOKENS.index(config.EOS_TOKEN) == config.EOS_ID
