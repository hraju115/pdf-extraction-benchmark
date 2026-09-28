# Question spot-check (20 of the hand-authored questions)

**Result (2026-09-28): all 20 sampled questions are correct and answerable from the cited page of their PDF**
(`data/benchmark_pdfs/<doc>.pdf`), and so are the 10 questions that `corpus/verify.py` defers to a human because their
evidence lives in figure pixels (second table). Method: the cited page's text layer was searched for the expected
answer (`pdftotext -layout -f N -l N`); the two scanned NASA pages and every figure question were also rendered
(`pdftoppm`) and read by eye. The last column records where the answer was found.

Sample drawn with seed 20260926; unanswerable questions excluded.

| # | document | page | type | question | expected answer | checked |
|---|---|---|---|---|---|---|
| 1 | `erp2024_full` | 70 | table_multi_row | In Table 2-1, how did Q4/Q4 growth compare between federal defense and nondefense spending? | Defense 3.3 percent versus nondefense 4.7 percent | ✓ Table 2-1 rows "Defense 3.3" and "Nondefense 4.7" |
| 2 | `acsbr_023` | 3 | table_cell | What was the 2023 ACS median household income for Massachusetts? | 99,858 dollars | ✓ Table 1, Massachusetts row, 2023 estimate 99,858 |
| 3 | `acsbr_023` | 2 | lookup | What was the U.S. median household income according to the ACS? | $77,719 | ✓ "was $77,719, according to the ACS (Table 1)" |
| 4 | `rfc9158` | 2 | cross_page | Which RFC assigned the omitted identifiers that this document registers, and what are their names? | RFC 4212 assigned id-acTemplate and id-openPGPCertTemplateExt | ✓ page 2 names RFC 4212; the registry table on page 3 lists both identifiers |
| 5 | `nasa_tm100396` | 9 | lookup | How many MSFC/ABMA-related launches does the earlier report summarize? | 155 launches, through SA-208 (Skylab 4) | ✓ scan, read by eye: "the past 155 MSFC/ABMA-related vehicle launches through SA-208 (Skylab 4)" |
| 6 | `rfc9112` | 7 | lookup | May a recipient treat a single LF as a line terminator? | Yes, a recipient MAY recognize a single LF as a line terminator and ignore any preceding CR | ✓ verbatim on page 7 |
| 7 | `nasa_se_handbook` | 39 | table_multi_row | Which reviews are listed for Phase C? | CDR, PRR, SIR and Safety review | ✓ Phase C "Reviews" list: CDR, PRR, SIR, Safety review |
| 8 | `rfc9457` | 8 | lookup | What must clients do with extension members they do not recognize? | They MUST ignore them | ✓ "Clients consuming problem details MUST ignore any such extensions that they don't recognize" |
| 9 | `acsbr_023` | 3 | table_cell | What was the 2022 ACS Gini index for the United States? | 0.486 | ✓ Table 1, United States row, 2022 ACS Gini index 0.486 |
| 10 | `erp2024_table1` | 1 | table_cell | What was the percent change in real GDP in 1985? | 4.2 | ✓ 1985 row, first column 4.2 |
| 11 | `nasa_webb_factsheet` | 2 | lookup | Which space agencies partner with NASA on Webb? | ESA (European Space Agency) and the Canadian Space Agency | ✓ "Partners" paragraph |
| 12 | `rfc9112` | 1 | lookup | Which RFC does RFC 9112 partly obsolete, and what is its STD number? | It obsoletes portions of RFC 7230 and is STD 99 | ✓ header "STD: 99", "Obsoletes: 7230"; abstract "obsoletes portions of RFC 7230" |
| 13 | `nasa_tm100396` | 9 | cross_page | At what time was STS-34 launched, and at what time was the GOES-7 visible picture taken? | Launch at 1654 u.t.; the GOES-7 picture at 1701 u.t., 7 minutes after liftoff | ✓ launch "at 1654 u.t." on page 9; GOES-7 picture "at 1701 u.t. (7 min after liftoff)" on page 8 and in the Figure 3/4 captions |
| 14 | `nasa_tm100396` | 9 | lookup | Why were SRB descent/impact atmosphere data not taken for STS-34? | Because a ship was unavailable for STS-34 duty | ✓ scan, read by eye: "Since a ship was unavailable for STS-34 duty, the SRB descent/impact atmosphere data were not taken" |
| 15 | `erp2024_table3` | 1 | lookup | What is the title of the table and the period it covers? | Table B-3. Gross domestic product, 2008-2023 | ✓ title line "Table B–3. Gross domestic product, 2008–2023" |
| 16 | `p60_282` | 8 | table_cell | In Figure 1, what was the 2023 median income of Asian households? | $112,800 | ✓ Figure 1, Asian row, 2023 median income $112,800 |
| 17 | `erp2024_full` | 51 | table_cell | In Table 1-1, what was the percent change in the 90th/10th percentile wage ratio from 2020:Q2 to 2023:Q4? | -8 percent | ✓ Table 1-1, "90th percentile / 10th percentile", 2020:Q2–2023:Q4 column –8 |
| 18 | `nasa_tm_x9471` | 4 | lookup | What was the maximum chamber pressure available at the test stand? | 7.9 MPa | ✓ "chamber pressure was constrained to a maximum of 7.9 MPa" |
| 19 | `rfc9112` | 17 | lookup | What does the example header 'Transfer-Encoding: gzip, chunked' indicate? | The content was compressed with gzip and then chunked | ✓ "compressed using the gzip coding and then chunked using the chunked coding" |
| 20 | `rfc9457` | 5 | lookup | In the validation-error example, which status is returned? | 422 Unprocessable Content | ✓ example starts "HTTP/1.1 422 Unprocessable Content" |

## Questions the verifier defers to a human (evidence only in figure pixels)

`python -m benchmarks.pdf_extraction.corpus.verify` reports `166 verified, 10 need a human check, 0 problems` over the
176 hand-authored and 14 generated questions (the 14 unanswerable ones are skipped): of the 162 answerable hand-authored
questions 155 are machine-verified and 7 are below; of the 14 generated questions 11 are machine-verified and 3 are below.
All 10 were checked by eye on the rendered page.

| id | page | question | expected answer | checked |
|---|---|---|---|---|
| `nasa_tm_x9471#10` | 6 | In Figure 1, what nozzle exit pressure ratio does the thruster design point have? | 1.238 | ✓ upper curve, labelled "Thruster Design Point", 1.238 |
| `nasa_tm_x9471#11` | 6 | In Figure 1, what nozzle exit Mach number ratio does the thruster design point have? | 0.893 | ✓ lower curve 0.893 (0.522 is the chamber-pressure ratio on the x-axis) |
| `nasa_tm_x9471#12` | 6 | Which two nozzles are compared in Figure 2 and what unit is on its axis? | Thruster and RS-25, axis in centimeters | ✓ curves labelled "Thruster" and "RS-25", axis "[cm]" |
| `nasa_webb_factsheet#9` | 2 | What does the label ISIM stand for? | Integrated Science Instrument Module | ✓ diagram label |
| `nasa_webb_factsheet#10` | 2 | What does the label OTE stand for? | Optical Telescope Element | ✓ diagram label |
| `nasa_webb_factsheet#11` | 2 | Which two mirrors are labelled in the diagram? | Primary Mirror and Secondary Mirror | ✓ diagram labels |
| `nasa_webb_factsheet#12` | 2 | Which components are labelled Solar Array and Earth-Pointing Antenna? | Solar Array and Earth-Pointing Antenna on the spacecraft-bus side | ✓ diagram labels next to "Spacecraft Bus" |
| `gen_t4_bar#1` | 1 | In the chart 'Throughput by team', what is the value for Ember? | 86 | ✓ bar labelled 86 |
| `gen_t4_flow#1` | 1 | Which stage comes directly after Segment? | Vectorize | ✓ Collect → Normalize → Segment → Vectorize → Store |
| `gen_t4_line#1` | 1 | What is the final 2024 value of the Beta series? | 68 | ✓ Beta line ends at 68 |
