"""Builds Supplementary Table S2 (labels, restatement audit, scores, results) from the CSV and JSON files in real_ext/."""
import json, pandas as pd, numpy as np
from openpyxl import Workbook
from openpyxl.styles import Font, Alignment, PatternFill
from openpyxl.utils import get_column_letter

lab = pd.read_csv('real_ext/dcb_labels.csv'); aud = pd.read_csv('real_ext/restatement_audit.csv')
ev = pd.read_csv('real_ext/dcb_eval_set.csv'); res = json.load(open('real_ext/res_dcb.json'))
wb = Workbook(); ws = wb.active; ws.title = 'README'
readme = [
 'Supplementary Table S2 for: GreenAudit-GNN: Peer-Conditioned Cross-Modal Consistency Scoring for Explainable ESG Disclosure Anomaly Detection.',
 'Sheet "Labels": 20 companies whose filings were read for notes on restatements, boundary or method changes (Section 8.6). Label 1 = the company states in its own filing that the year-on-year comparison is not like for like. Label 0 = no such note found on the GHG table, its footnotes or by keyword scan. A 0 is not proof that no break exists.',
 'Sheet "Restatement audit": 18 comparisons of a filing\'s printed values with the prior-year column of the next filing (Coal India, Indian Oil, HDFC Bank; Section 8.5).',
 'Sheet "Scores": the 19 companies with a year-on-year change, with the numeric scores used in Section 8.6. Restaurant Brands Asia has no previous-year Scope 2 and is not scored.',
 'Sheet "Results": ROC-AUC of each score for separating the 4 labelled positives from the 15 negatives, with 95% bootstrap interval and exact one-sided permutation p-value (real_ext/dcb_eval.py).',
 'Page numbers are PDF page indexes from text extraction (pdftotext order) and may differ from printed page numbers. Nothing was estimated. Every label should be checked against the filing by a person before it is relied on.']
for t in readme: ws.append([t])
ws.column_dimensions['A'].width = 140
for r in ws.iter_rows():
    for c in r: c.alignment = Alignment(wrap_text=True, vertical='top'); c.font = Font(name='Arial', size=10)
hf = PatternFill('solid', fgColor='DDDDDD')
def sheet(name, df, widths, fmt=None):
    s = wb.create_sheet(name); s.append(list(df.columns))
    for row in df.itertuples(index=False): s.append([None if (isinstance(v, float) and np.isnan(v)) else v for v in row])
    for c in range(1, len(df.columns) + 1):
        cell = s.cell(1, c); cell.font = Font(name='Arial', bold=True, size=10); cell.fill = hf; cell.alignment = Alignment(wrap_text=True, vertical='top')
    for row in s.iter_rows(min_row=2):
        for cell in row:
            cell.font = Font(name='Arial', size=10); cell.alignment = Alignment(wrap_text=True, vertical='top')
            if fmt and isinstance(cell.value, float): cell.number_format = fmt
    for i, w in enumerate(widths, 1): s.column_dimensions[get_column_letter(i)].width = w
    s.freeze_panes = 'A2'
sheet('Labels', lab, [34, 7, 60, 16, 90, 44])
sheet('Restatement audit', aud, [28, 30, 11, 46, 16, 46, 18, 18, 34])
sc = ev[['company', 'dcb', 'y', 'yr', 'lr', 'lvl', 'peer_z']].copy()
sc.columns = ['Company', 'Disclosed break (1/0)', 'Year-on-year log change', 'Peer-centred YoY change', 'Peer-centred level (log10)', 'Level (log10)', 'Peer z-score']
sheet('Scores', sc, [40, 16, 18, 18, 18, 14, 14], '0.000')
rs = pd.DataFrame([dict(Score=k, **{'ROC-AUC': v['auc'], '95% bootstrap low': v['lo'], '95% bootstrap high': v['hi'],
      'Permutation p (one-sided, exact)': v['p'], 'Companies': v['n'], 'Positives': v['positives']}) for k, v in res.items()])
sheet('Results', rs, [36, 12, 16, 16, 28, 12, 10], '0.000')
wb.save('Supplementary_Table_S2_Label_Audit.xlsx'); print('saved')
