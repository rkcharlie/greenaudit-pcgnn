"""Builds Supplementary Table S2 (labels, restatement pairs, audit, scores, results) from the files in real_ext/."""
import json, pandas as pd, numpy as np
from openpyxl import Workbook
from openpyxl.styles import Font, Alignment, PatternFill
from openpyxl.utils import get_column_letter

R = 'real_ext/'
lab = pd.read_csv(R + 'dcb_labels.csv'); aud = pd.read_csv(R + 'restatement_audit.csv')
ev = pd.read_csv(R + 'dcb_eval_set.csv'); res = json.load(open(R + 'res_dcb.json'))
obs = pd.read_csv(R + 'restatement_obs.csv'); rres = json.load(open(R + 'res_restatement.json'))
sens = json.load(open(R + 'res_restatement_sens.json')); pw = json.load(open(R + 'res_restatement_power.json'))
NOTE = {('Adani Enterprises', 2025): 'Footnote in the next filing: fugitive methane is excluded from Scope 1 in both years.',
        ('Vedanta', 2024): 'Footnote in the next filing: Scope 1 data for FY2023-24 has been restated. No reason is given on the page.',
        ('BSE Ltd', 2024): 'Footnote in the next filing: the calculations use the updated CEA emission factors (version 20). The note does not link this to the change.',
        ('Tata Steel (standalone)', 2024): 'No note about this change found on the page. Values are printed in whole million tonnes.',
        ('ITC', 2024): 'No note found on the page. Values are printed in thousand tonnes.',
        ('NTPC', 2024): 'No note found on the page. The first filing prints Scope 1 in million tonnes with two decimals; the intensity per kWh also changed (834.81 to 867.87 g/kWh).',
        ('Hindustan Zinc', 2025): 'No note found on the page.',
        ('Gujarat Fluorochemicals', 2025): 'Footnote in the next filing: previous-year Scope 1 and Scope 2 restated after a review of activity data and calculation methodologies.'}
wb = Workbook(); ws = wb.active; ws.title = 'README'
readme = [
 'Supplementary Table S2 for: GreenAudit-GNN: Peer-Conditioned Cross-Modal Consistency Scoring for Explainable ESG Disclosure Anomaly Detection.',
 'Sheet "Restatement pairs": 63 company-years from 37 companies (Section 8.5). For each, the Scope 1 and Scope 2 values printed in the BRSR table of the first filing, the prior-year column of the next filing, the file and PDF page of both, and the label (1 = Scope 1 or Scope 2 restated). Rule, fixed before scoring: a value is restated if the next filing prints a value that differs by more than 1% of the first value and by more than 1.5 units of the precision at which the first filing prints it. The column "Checked by author" is empty on purpose: every row must be checked against the two pages before the label is relied on.',
 'Sheet "Restatement results": ROC-AUC, cluster-bootstrap interval and permutation p-value for five scores (real_ext/restatement_eval.py). Sheets "Rule sensitivity" and "Power" give the results of real_ext/restatement_sens.py and real_ext/restatement_power.py.',
 'Sheet "Labels": 20 companies whose filings were read for notes on restatements, boundary or method changes (Section 8.6). Label 1 = the company states in its own filing that the year-on-year comparison is not like for like. Label 0 = no such note found on the GHG table, its footnotes or by keyword scan. A 0 is not proof that no break exists.',
 'Sheet "Restatement audit": 18 comparisons of a filing\'s printed values with the prior-year column of the next filing (Coal India, Indian Oil, HDFC Bank; Section 8.5). None of the 12 Scope 1 and 2 totals was restated.',
 'Sheet "Scores": the 19 companies of Section 8.6 with a year-on-year change, with the numeric scores used there. Restaurant Brands Asia has no previous-year Scope 2 and is not scored.',
 'Sheet "Results": ROC-AUC of each score for separating the 4 disclosed-break companies from the 15 others (real_ext/dcb_eval.py).',
 'Selection: the companies were chosen by the authors from large listed Indian companies, mostly heavy emitters, and downloaded from the exchange portal. They are not a random sample. Page numbers are PDF page indexes from text extraction and may differ from printed page numbers. Nothing was estimated.']
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
tag = lambda f: f.replace('AR_', '').split('_A_')[0] if isinstance(f, str) else f
pairs = pd.DataFrame({
 'Company': obs.company, 'Sector group': obs.macro, 'Year of first filing (FY end)': obs.fy,
 'First filing (file name part)': obs.file_first, 'Page': obs.page_first, 'Next filing (file name part)': obs.file_next, 'Page (next)': obs.page_next,
 'Scope 1, first filing (tCO2e)': obs.s1_first, 'Scope 1, same year in next filing (tCO2e)': obs.s1_next, 'Scope 1 change (%)': obs.s1_pct,
 'Scope 2, first filing (tCO2e)': obs.s2_first, 'Scope 2, same year in next filing (tCO2e)': obs.s2_next, 'Scope 2 change (%)': obs.s2_pct,
 'Scope 1 restated (1/0)': obs.s1_restated, 'Scope 2 restated (1/0)': obs.s2_restated, 'Label (1 = restated)': obs.label.astype(int),
 'Note printed with the change': [NOTE.get((c, f), '') for c, f in zip(obs.company, obs.fy)],
 'Reading note': obs.note_next.fillna(''), 'Checked by author (fill in)': ''})
sheet('Restatement pairs', pairs, [26, 20, 12, 46, 7, 46, 8, 16, 18, 11, 16, 18, 11, 10, 10, 10, 60, 40, 14], '#,##0.00')
rs0 = pd.DataFrame([dict(Score=k, **{'ROC-AUC': v['auc'], '95% cluster-bootstrap low': v['lo'], '95% cluster-bootstrap high': v['hi'],
      'Permutation p (one-sided, 20,000 draws)': v['p'], 'Company-years': v['n'], 'Positives': v['positives'], 'Companies': v['companies']}) for k, v in rres.items()])
sheet('Restatement results', rs0, [36, 12, 16, 16, 24, 12, 10, 10], '0.000')
srows = []
for r in sens:
    for k in ('Raw year-on-year change (size)', 'Peer-centred change (size)', 'Peer-centred level (size)', 'Peer z-score (change and level)', 'Raw emissions level'):
        if k in r: srows.append({'Label rule': r['rule'], 'Threshold': r['threshold'], 'Positives': r['positives'], 'Score': k, 'ROC-AUC': r[k][0], 'Permutation p': r[k][1]})
sheet('Rule sensitivity', pd.DataFrame(srows), [14, 10, 10, 36, 10, 14], '0.000')
prow = [dict(Item='power at 9 positives and 54 negatives', **{'True ROC-AUC': p['true_auc'], 'Power (one-sided 5%)': p['power']}) for p in pw['power']]
prow += [dict(Item='positives and pairs needed for 80% power at 16% prevalence', **{'True ROC-AUC': n['true_auc'], 'Power (one-sided 5%)': n['power'], 'Positives': n['positives'], 'Company-years': n['pairs']}) for n in pw['needed_for_80pct_power']]
sheet('Power', pd.DataFrame(prow), [56, 14, 20, 12, 14], '0.00')
sheet('Labels', lab, [34, 7, 60, 16, 90, 44])
sheet('Restatement audit', aud, [28, 30, 11, 46, 16, 46, 18, 18, 34])
sc = ev[['company', 'dcb', 'y', 'yr', 'lr', 'lvl', 'peer_z']].copy()
sc.columns = ['Company', 'Disclosed break (1/0)', 'Year-on-year log change', 'Peer-centred YoY change', 'Peer-centred level (log10)', 'Level (log10)', 'Peer z-score']
sheet('Scores', sc, [40, 16, 18, 18, 18, 14, 14], '0.000')
rs = pd.DataFrame([dict(Score=k, **{'ROC-AUC': v['auc'], '95% bootstrap low': v['lo'], '95% bootstrap high': v['hi'],
      'Permutation p (one-sided, exact)': v['p'], 'Companies': v['n'], 'Positives': v['positives']}) for k, v in res.items()])
sheet('Results', rs, [36, 12, 16, 16, 28, 12, 10], '0.000')
wb.save('Supplementary_Table_S2_Label_Audit.xlsx'); print('saved')
