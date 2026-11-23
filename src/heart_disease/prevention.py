"""International prevention visit notes and a standalone printable summary."""

from html import escape

import streamlit as st


GUIDANCE = {
    "Europe": (
        "ESC SCORE2 and SCORE2-OP",
        "https://www.escardio.org/guidelines/practice-tools/cvd-prevention-toolbox/score2-and-score2-op-calculators/",
    ),
    "United States": (
        "AHA PREVENT",
        "https://professional.heart.org/en/guidelines-and-statements/prevent-calculator",
    ),
    "WHO regional guidance": (
        "WHO HEARTS cardiovascular risk charts",
        "https://www.who.int/publications/i/item/9789240001367",
    ),
}
ROW_TEMPLATE = "<tr><th scope='row'>{}</th><td>{}</td></tr>"


def printable_summary(sections, fictional):
    body = []
    for heading, rows in sections:
        cells = "".join(
            ROW_TEMPLATE.format(escape(label), escape(value))
            for label, value in rows
        )
        body.append(f"<section><h2>{escape(heading)}</h2><table>{cells}</table></section>")
    badge = "<p><strong>Fictional example</strong></p>" if fictional else ""
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Cardiovascular prevention visit summary</title>
<style>
body {{ font: 16px system-ui, sans-serif; color: #122c33; max-width: 48rem;
       margin: 2rem auto; padding: 0 1rem; }}
h2 {{ font-size: 1.15rem; margin-top: 1.5rem; }}
table {{ width: 100%; border-collapse: collapse; }}
th, td {{ text-align: left; vertical-align: top; padding: .5rem;
          border-bottom: 1px solid #bcc9cc; overflow-wrap: anywhere; white-space: pre-wrap; }}
th {{ width: 40%; }}
@page {{ margin: 14mm; }}
@media print {{
    body {{ font-size: 10pt; line-height: 1.25; max-width: none; margin: 0; padding: 0; }}
    h1 {{ font-size: 19pt; line-height: 1.2; margin: 0 0 6pt; }}
    h2 {{ font-size: 12pt; margin: 0 0 4pt; break-after: avoid; }}
    p {{ margin: 0 0 6pt; }}
    section {{ margin-top: 9pt; break-inside: avoid; }}
    table {{ table-layout: fixed; }}
    th, td {{ padding: 3pt 5pt; }}
    tr {{ break-inside: avoid; }}
}}
</style></head><body>
<main><h1>Cardiovascular prevention visit summary</h1>{badge}
<p>For discussion with your clinician. This summary records entered information.
Any risk estimate shown was entered from a separate assessment.</p>
{''.join(body)}</main></body></html>"""


def measurement(value, unit):
    return "Not recorded" if value is None else f"{value:g} {unit}"


def prevention_visit():
    st.header("Cardiovascular prevention visit")
    st.write("Organize measurements and questions to discuss with your clinician.")
    st.caption("Avoid entering names or identifiers. The app keeps entries in this session and does not save them to the experiment database. Keep downloaded summaries private.")
    profile = st.selectbox("Start with", ["Blank summary", "Fictional example"])
    fictional = profile == "Fictional example"
    if fictional:
        st.info("Fictional example for testing the visit workflow.")
    region = st.selectbox(
        "Guidance to explore", ["No pathway selected", *GUIDANCE],
        help="Use the pathway appropriate for your care setting and confirm eligibility with your clinician.",
    )
    if region in GUIDANCE:
        name, url = GUIDANCE[region]
        st.link_button(name, url)
        st.caption("This opens official guidance without sending your form entries. Your clinician can confirm whether a tool applies.")
    else:
        st.caption("You can prepare a summary without selecting a risk pathway.")

    unit = st.radio("Cholesterol units", ["mg/dL", "mmol/L"], horizontal=True)
    with st.form(f"prevention-{profile}"):
        st.subheader("Measurements")
        c1, c2 = st.columns(2)
        with c1:
            age = st.number_input("Age (years)", min_value=1, value=54 if fictional else None)
            systolic = st.number_input("Systolic blood pressure (mm Hg)", min_value=1, value=132 if fictional else None)
            diastolic = st.number_input("Diastolic blood pressure (mm Hg)", min_value=1, value=82 if fictional else None)
            bp_date = st.date_input("Blood pressure measurement date", value=None)
        with c2:
            example_total, example_hdl = (198.0, 56.0) if unit == "mg/dL" else (5.12, 1.45)
            total = st.number_input(
                f"Total cholesterol ({unit})", min_value=0.01, value=example_total if fictional else None,
            )
            hdl = st.number_input(
                f"HDL cholesterol ({unit})", min_value=0.01, value=example_hdl if fictional else None,
            )
            lipid_date = st.date_input("Cholesterol measurement date", value=None)
        smoking = st.selectbox("Current smoking", ["Not recorded", "Yes", "No"])
        questions = st.text_area(
            "Questions for the visit", value="Which measurements should I track before my next visit?" if fictional else "",
            max_chars=800,
        )

        with st.expander("Clinician assessment · optional"):
            st.caption("Record a result from an external tool. This app does not calculate cardiovascular risk.")
            score = st.number_input("Estimated risk (%)", min_value=0.0, max_value=100.0, value=None)
            tool = st.text_input("Tool name", max_chars=80)
            calibration = st.text_input("Region or calibration used", max_chars=80)
            horizon = st.number_input("Time horizon (years)", min_value=1, value=None)
            outcome = st.text_input("Outcome estimated", max_chars=120)
            assessed_on = st.date_input("Assessment date", value=None)
        submitted = st.form_submit_button("Generate visit summary", type="primary")

    if not submitted:
        return

    details = (tool.strip(), calibration.strip(), horizon, outcome.strip(), assessed_on)
    if (score is not None or any(details)) and (score is None or not all(details)):
        st.error("Complete all clinician assessment fields to include a risk estimate, or clear them to continue without one.")
        return

    measurements = [
        ("Age", measurement(age, "years")),
        ("Systolic blood pressure", measurement(systolic, "mm Hg")),
        ("Diastolic blood pressure", measurement(diastolic, "mm Hg")),
        ("Blood pressure measurement date", str(bp_date or "Not recorded")),
        ("Total cholesterol", measurement(total, unit)),
        ("HDL cholesterol", measurement(hdl, unit)),
        ("Cholesterol measurement date", str(lipid_date or "Not recorded")),
        ("Current smoking", smoking),
    ]
    sections = [
        ("Visit preparation", [
            ("Guidance selected", region),
            ("Questions for the visit", questions.strip() or "Not recorded"),
        ]),
        ("Measurements", measurements),
    ]
    assessment = [("Risk estimate", "Not recorded")]
    if score is not None:
        assessment = [
            ("Risk estimate (entered externally)", f"{score:g}%"),
            ("Tool", details[0]), ("Region or calibration", details[1]),
            ("Time horizon", measurement(horizon, "years")), ("Outcome", details[3]),
            ("Assessment date", str(assessed_on)),
        ]
    sections.append(("Clinician assessment", assessment))

    st.subheader("Visit summary")
    for heading, rows in sections:
        st.markdown(f"**{heading}**")
        st.dataframe(
            {"Item": [label for label, _ in rows], "Recorded value": [value for _, value in rows]},
            hide_index=True,
        )
    st.download_button(
        "Download printable summary", printable_summary(sections, fictional),
        file_name="cardiovascular-visit-summary.html", mime="text/html", on_click="ignore",
    )
    st.caption("Open the downloaded file and print or save as PDF from your browser. Turn off Headers and footers to omit the local file path.")
