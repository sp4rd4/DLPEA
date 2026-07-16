# Results

This directory contains the exact experiment outputs cited in Chapter 4 of the dissertation.

| File | Produced by | Contents |
|---|---|---|
| `e1_real.json` | `src/run_e1_real.py` | E1: classification on ai4privacy PII and 20 Newsgroups |
| `e1_patch.json` | `src/run_e1_patch.py` | E1: expert patching scenario (new IBAN identifier) |
| `e2_cert.json` | `src/run_e2_cert.py` | E2: insider identification on CERT r4.2 |
| `e3_synth.json` | `src/run_e3_synth.py` | E3: drift detection on RandomRBF streams |
| `e3_real.json` | `src/run_e3_real.py` | E3: drift detection on INSECTS streams |
| `e4_real.json` | `src/run_e4_real.py` | E4: policy adaptation on Elec2 and INSECTS-Abr |
| `e4_policy.json` | `src/run_e4_policy.py` | E4: policy adaptation on policy-structured streams |
| `e4_prequential.png` | `src/run_e4_policy.py` | prequential accuracy curves (Figure 4.1) |
| `DIGEST_ch4.txt` | `digest_ch4.py` | human-readable digest of all results |

Re-running the experiment scripts overwrites these files.
