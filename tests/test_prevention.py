from datetime import date
from pathlib import Path

from streamlit.testing.v1 import AppTest

from heart_disease.prevention import printable_summary


def test_prevention_visit_summary(monkeypatch, tmp_path):
    monkeypatch.setenv("HEART_RUN_DIR", str(tmp_path))
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py", default_timeout=20).run()
    app.radio[0].set_value("Prevention Visit").run()
    app.button[0].click().run()
    assert not app.exception
    assert len(app.dataframe) == 3
    assert list(app.dataframe[1].value["Recorded value"]) == ["Not recorded"] * 8
    app.selectbox[0].select("Fictional example").run()
    app.selectbox[1].select("United States").run()
    assert app.get("link_button")[0].proto.url.endswith("/prevent-calculator")
    app.radio[1].set_value("mmol/L").run()
    score = next(widget for widget in app.number_input if widget.label == "Estimated risk (%)")
    score.set_value(0.0)
    app.button[0].click().run()
    assert len(app.error) == 1
    assert len(app.get("download_button")) == 0

    for field, value in zip(app.text_input, ["PREVENT", "United States", "Total CVD"]):
        field.set_value(value)
    next(widget for widget in app.number_input if widget.label == "Time horizon (years)").set_value(10)
    next(widget for widget in app.date_input if widget.label == "Assessment date").set_value(date(2026, 1, 1))
    app.button[0].click().run()
    assert not app.exception and not app.error
    assert "5.12 mmol/L" in list(app.dataframe[1].value["Recorded value"])
    assert "0%" in list(app.dataframe[2].value["Recorded value"])
    assert len(app.get("download_button")) == 1


def test_printable_summary_escapes_input():
    result = printable_summary([("Questions", [("Note", "<script>alert(1)</script>")])], True)
    assert "Fictional example" in result
    assert "<script>" not in result
    assert "&lt;script&gt;" in result
