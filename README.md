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
10. `python make_supp.py`          -> Supplementary Table S1 (spreadsheet)

The full runs take a long time on one machine; run_final.py uses multiprocessing.

## Data included
- `modeling_dataset.csv`, `company_weak_labels.csv`: the 38-company pilot panel and its heuristic labels (from the earlier GreenAudit-GNN work).
- `real_ext/new_companies.csv`: the 17 added companies with Scope 1 and 2 values, source file and page.
- `Supplementary_Table_S1_BRSR_Panel.xlsx`: all 55 companies with source file or URL and location in the filing.
- `real_ext/extract.py` and `real_ext/hits.json`: the text-extraction step for the 17 added companies. `extract.py` needs the filing PDFs on your own computer and a local path edit; `hits.json` holds its output.

## What this package does not contain
- The BRSR filing PDFs (they belong to the filers; see Supplementary Table S1 for source file or URL).
- Independent real anomaly labels (none exist; the paper makes no claim about accuracy on real filings).
- Older pilot scripts (weak_label_rule.py, train_and_evaluate.py, greenaudit_pipeline.py) are in the earlier GreenAudit-GNN package.

Repository link, DOI and licence: to be added by the authors.
