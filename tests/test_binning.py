from nli_groundedness.binning import add_bins


def test_add_bins_list_input():
    data = [{"value": float(i)} for i in range(10)]
    enriched, edges = add_bins(data, "value", num_bins=5)

    assert len(enriched) == 10
    assert len(edges) == 6
    bins = [row["bin"] for row in enriched]
    assert all(0 <= b <= 4 for b in bins)

    for row in enriched:
        assert row["bin_left"] <= row["value"] <= row["bin_right"]
