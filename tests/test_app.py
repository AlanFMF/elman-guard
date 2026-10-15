"""The app is what visitors see first, so CI opens it the way Streamlit does.

``AppTest`` runs ``app.py`` headless -- loading the shipped model, scoring the
bundled sample and rendering every section -- and fails on any exception. If a
change would break the deployed demo, it breaks here first.
"""

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

pytestmark = pytest.mark.skipif(
    not (ROOT / "models" / "elman.keras").exists(),
    reason="shipped model not present",
)


@pytest.mark.parametrize("language", ["Português (BR)", "English"])
def test_app_renders_without_errors(language):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=180)
    at.run()
    at.sidebar.selectbox[0].select(language).run()

    assert not at.exception, [e.value for e in at.exception]
    assert [t.value for t in at.title] == ["ElmanGuard"]
    labels = [m.label for m in at.metric]
    assert len(labels) == 4, labels
    expected = "Janelas" if language.startswith("Português") else "Windows"
    assert labels[0] == expected
