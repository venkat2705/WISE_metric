"""
Perturbation-type analysis of the 1,000-case set.

Each GT/HYP pair is auto-labelled by the type of perturbation applied to the GT:
  surface   : fillers (uh/um/well/so/yeah), contractions/colloquialisms (what's<->what is,
              wanna, gonna, ya), dropped function words, singular/plural  -> meaning preserved
  homophone : function-word homophones (for->four, to->two, here->hear, buy->by) -> spoken form preserved
  content   : named-entity / number / content-word corruption (jake->jay, fifteen->fifty,
              pizza->pisa, weather->leather)                                -> meaning altered
  flip      : negation insertion (not/don't) or antonym / action reversal
              (play->pause, open->close, to->from, set->delete, transfer->withdraw) -> intent reversed

For each metric it reports per-class mean/std, the AUC of separating meaning-preserving
(surface) from meaning-altering (content+flip) pairs, and Spearman rho against the ordinal
label (surface=3 > homophone=2 > content=1 > flip=0).

Usage:
    pip install pandas scipy
    python analyze_1000.py wise_1000_with_bertscore.csv
"""
import sys, difflib, pandas as pd, numpy as np
from scipy.stats import spearmanr, mannwhitneyu

FILL = {'uh','um','well','so','yeah','and','hmm','oh','like'}
NEG  = {'not',"don't",'never',"won't","can't"}
HOMO = {frozenset(p) for p in [('four','for'),('two','to'),('hear','here'),('by','buy'),('too','to'),('fore','for'),('bout','about')]}
PLUR = {('recipes','recipe'),('minutes','minute'),('degrees','degree'),('directions','direction'),('sandwiches','sandwich')}
ANT  = {('to','from'),('play','pause'),('open','close'),('lock','unlock'),('on','off'),('off','on'),('set','delete'),('set','stop'),
        ('set','snooze'),('transfer','withdraw'),('send','cancel'),('wake','lull'),('up','sleep'),('start','stop'),('add','remove'),
        ('schedule','cancel'),('remind','tell'),('from','to'),('play','stop')}
FUNC = {'a','an','the','to','i','is','it','you','do','of','me','my','please','can','could','for','up','on','in','at','that','this',
        'would','will','are','am'}

def expand(s):
    s = ' '.join(str(s).lower().split())
    for a, b in [("what's","what is"),("i'm","i am"),(" 'm "," am "),("wanna","want to"),("gonna","going to"),(" ya "," you ")]:
        s = (' ' + s + ' ').replace(a, b).strip(); s = ' '.join(s.split())
    return s.split()

def classify(g, h):
    if str(g).split() == str(h).split(): return 'identical'
    a = expand(g); b = [w for w in expand(h) if w not in FILL]
    if a == b: return 'surface'
    sm = difflib.SequenceMatcher(a=a, b=b); flip = content = homo = False
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == 'equal': continue
        A = a[i1:i2]; B = b[j1:j2]
        if any(w in NEG for w in B) and not any(w in NEG for w in A): flip = True; continue
        if any((x, y) in ANT for x in A for y in B): flip = True; continue
        B2 = []
        for w in B:
            m = [x for x in A if frozenset((x, w)) in HOMO]
            if m and w != m[0]: homo = True; B2.append(m[0])
            else: B2.append(w)
        pa = [w for w in A if w not in FUNC]; pb = [w for w in B2 if w not in FUNC]
        if pa == pb: continue
        if len(pa) == 1 and len(pb) == 1 and (pa[0], pb[0]) in PLUR: continue
        content = True
    if flip: return 'flip'
    if content: return 'content'
    return 'homophone' if homo else 'surface'

def auc(pos, neg):
    return mannwhitneyu(pos, neg, alternative='greater').statistic / (len(pos) * len(neg))

def main(path):
    df = pd.read_csv(path)
    df['label'] = [classify(g, h) for g, h in zip(df['GT'], df['Hyp'])]
    print(df['label'].value_counts().to_string(), '\n')
    d = df[df.label != 'identical']
    metrics = [c for c in ['WRR', 'WISE_gemma', 'WISE', 'BS_F1_raw', 'BS_F1_rescaled', 'SeMaScore']
               if c in d.columns and pd.to_numeric(d[c], errors='coerce').notna().any()]
    for c in metrics: d[c] = pd.to_numeric(d[c], errors='coerce')
    d = d.dropna(subset=metrics)
    print(f"rows used: {len(d)}\n")
    ordm = d.label.map({'surface': 3, 'homophone': 2, 'content': 1, 'flip': 0})
    rows = []
    for m in metrics:
        g = d.groupby('label')[m].agg(['mean', 'std'])
        pres = d[d.label == 'surface'][m]; alt = d[d.label.isin(['flip', 'content'])][m]
        rows.append(dict(metric=m,
            surface=f"{g.loc['surface','mean']:.1f}±{g.loc['surface','std']:.1f}",
            homophone=f"{g.loc['homophone','mean']:.1f}±{g.loc['homophone','std']:.1f}",
            content=f"{g.loc['content','mean']:.1f}±{g.loc['content','std']:.1f}",
            flip=f"{g.loc['flip','mean']:.1f}±{g.loc['flip','std']:.1f}",
            AUC=f"{auc(pres, alt):.3f}",
            AUC_flip=f"{auc(pres, d[d.label=='flip'][m]):.3f}",
            AUC_content=f"{auc(pres, d[d.label=='content'][m]):.3f}",
            rho=f"{spearmanr(d[m], ordm).statistic:.3f}"))
    out = pd.DataFrame(rows)
    print(out.to_string(index=False))
    out.to_csv('perturbation_analysis.csv', index=False)
    df.to_csv('wise_1000_labeled.csv', index=False)

if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'wise_1000_with_bertscore.csv')
