from argus_live.simulation.regression_library import build_regression_library, get_named_scenario


def test_regression_library_contains_named_bad_days():
    library = build_regression_library()
    names = {s.name for s in library}
    assert "clean_baseline" in names
    assert "correlated_bad_day" in names
    assert get_named_scenario("thin_book").plan.name == "thin_book"
