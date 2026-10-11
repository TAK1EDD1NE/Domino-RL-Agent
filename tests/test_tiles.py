import pytest
from domino.tiles import (NUM_TILES, TILES, tile_id, tile_values, tile_pips, tile_name,
                          tile_from_name, ALL_TILE_IDS, DOUBLE_IDS, SIX_SIX_ID, TOTAL_PIPS)

def test_ids_cover_0_to_27_exactly_once():
    ids = [tile_id(a, b) for a in range(7) for b in range(a, 7)]
    assert sorted(ids) == list(range(NUM_TILES))
    assert len(set(ids)) == NUM_TILES

def test_id_is_order_insensitive():
    for a in range(7):
        for b in range(a, 7):
            assert tile_id(a, b) == tile_id(b, a)

def test_known_ids():
    assert tile_id(0, 0) == 0
    assert tile_id(0, 6) == 6
    assert tile_id(1, 1) == 7
    assert tile_id(6, 6) == 27 == SIX_SIX_ID

def test_values_and_pips_roundtrip():
    total = 0
    for tid in ALL_TILE_IDS:
        a, b = tile_values(tid)
        assert a <= b
        assert tile_id(a, b) == tid
        assert tile_pips(tid) == a + b
        total += a + b
    assert total == TOTAL_PIPS == 168

def test_doubles_and_names():
    assert len(DOUBLE_IDS) == 7
    assert all(tile_values(t)[0] == tile_values(t)[1] for t in DOUBLE_IDS)
    assert tile_name(27) == "[6|6]"
    assert tile_from_name("[6|6]") == 27
    for tid in ALL_TILE_IDS:
        assert tile_from_name(tile_name(tid)) == tid

def test_invalid_inputs():
    with pytest.raises(ValueError):
        tile_id(7, 8)
    with pytest.raises(ValueError):
        tile_from_name("nonsense")
