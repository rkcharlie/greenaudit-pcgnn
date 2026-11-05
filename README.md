# PC-GNN benchmark and experiments (code for the paper)

Paper: Peer-Conditioned Cross-Modal Consistency Scoring for ESG Disclosure Verification (Krishna and Ahamad).
Appendix F of the paper lists every script and what it writes. Everything is seeded.

## Environment
Python 3.11, numpy, scipy, pandas, scikit-learn, matplotlib, openpyxl (see requirements.txt; results were produced with numpy 2.4, scipy 1.17, scikit-learn 1.8).

## Order
1. `python theory.py`               -> theory_check.json (Monte Carlo check of Propositions 1 and 2)
2. `python run_main.py`            -> res_main_clean.json
3. `python run_sweeps.py`          -> res_all.json
4. `python run_final.py`           -> res_final.json
5. `python analyze_final.py` and `python final_analysis.py` -> res_final_summary.json, res_robust_rows.json, figs/
6. `python quad_check.py`          -> res_quad.json
7. `python explain.py`             -> res_explain.json
8. `python real_pilot.py`, `python heur_check.py` -> res_real.json, res_heur.json (read modeling_dataset.csv and company_weak_labels.csv, which are included)
9. `cd real_ext && python extended_panel.py` -> extended_panel.csv (reads new_companies.csv and the pilot files)
10. `python make_supp.py`          -> Supplementary Table S1 (spreadsheet; Indian Oil FY2023-24 corrected to the all-gas basis)
11. `python run_transfer.py 30 9000` then `python analyze_transfer.py` and `python make_fig_transfer.py` -> res_transfer.json, res_transfer_summary.json, figs/fig_transfer.png (Section 7.9; about 15 minutes)
12. `cd real_ext && python dcb_eval.py` -> res_dcb.json (Section 8.6; reads dcb_labels.csv and dcb_eval_set.csv)
13. `cd real_ext && python restatement_eval.py` -> restatement_obs.csv, res_restatement.json (Section 8.5; reads data/filings_v4a.py, data/filings_v4b.py and data/filings_v5.py, the hand-read Scope 1 and 2 values with file and page)
14. `cd real_ext && python restatement_sens.py` -> res_restatement_sens.json (other thresholds, Scope 1 only, Scope 2 only, total)
15. `cd real_ext && python restatement_power.py` -> res_restatement_power.json (what the label set could detect)
16. `python make_supp3.py`          -> Supplementary Table S2 (restatement pairs, results, sensitivity, power, labels, audit, scores)

`real_ext/restatement_tools.py` has two modes: `notes <folder> out.csv` lists candidate comparability notes in a folder of filings (a keyword scan that only lists candidates), and `compare earlier.pdf later.pdf` prints the Scope 1 and 2 values of the earlier year against the prior-year column of the later filing. Labels in `dcb_labels.csv` are the authors' reading of the notes and must be checked against the filing page cited.

The full runs take a long time on one machine; run_final.py uses multiprocessing.

## Data included
- `modeling_dataset.csv`, `company_weak_labels.csv`: the 38-company pilot panel and its heuristic labels (from the earlier GreenAudit-GNN work).
- `real_ext/new_companies.csv`: the 17 added companies with Scope 1 and 2 values, source file and page.
- `Supplementary_Table_S1_BRSR_Panel.xlsx`: all 55 companies with source file or URL and location in the filing.
- `real_ext/extract.py` and `real_ext/hits.json`: the text-extraction step for the 17 added companies. `extract.py` needs the filing PDFs on your own computer and a local path edit; `hits.json` holds its output.

## What this package does not contain
- The BRSR filing PDFs (they belong to the filers; see Supplementary Table S1 for source file or URL).
- A real anomaly benchmark. The real labels are 63 restatement pairs from 37 companies (9 positives, `real_ext/restatement_obs.csv`) and 20 comparability-break labels (4 scored positives, `real_ext/dcb_labels.csv`). Both are the authors' reading of public filings and must be checked against the cited pages; the companies were chosen by the authors, not sampled. The paper makes no claim about accuracy on real filings.
- Older pilot scripts (weak_label_rule.py, train_and_evaluate.py, greenaudit_pipeline.py) are in the earlier GreenAudit-GNN package.

Repository: https://github.com/rkcharlie/greenaudit-pcgnn. DOI and licence: to be added by the authors.

- `real_ext/data/filings_v4a.py`, `filings_v4b.py`: for each company-year, the file name part, PDF page, Scope 1 and 2 as printed in that filing and in the prior-year column, the unit multiplier and a reading note. The annual-report PDFs themselves are public (exchange portal) and are not included.
- `real_ext/verify_values.py`: checks that each hand-read value in `data/filings_*.py` appears on the cited page of the text extracted from the filing (needs the filings converted with `pdftotext -layout` into a `txt/` folder). It is a typing check, not a substitute for reading the page: small whole numbers (for example Tata Steel in whole Mt) can match by chance.
