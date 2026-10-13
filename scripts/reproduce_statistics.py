import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    inventory = root / 'DATA_INVENTORY.json'
    entries = (json.loads(inventory.read_text()) if inventory.exists()
               else json.loads((root / 'manifest.json').read_text())['files'])
    for name, entry in entries.items():
        if not (root / name).resolve().is_relative_to(root):
            raise ValueError('inventory path escapes data directory')
        if sha(root / name) != entry['sha256']:
            raise ValueError(f'data checksum mismatch: {name}')
    p = np.load(root / 'data/all_oof_predictions.npz')
    loss = np.load(root / 'data/all_oof_spectral_losses.npz')
    rep = np.load(root / 'data/representation_oof.npz')
    expected = json.loads((root / 'results/analysis_completion.json').read_text())
    rep_expected = json.loads((root / 'results/representation_analysis.json').read_text())
    unique, inverse = np.unique(p['session'], return_inverse=True)
    if len(unique) != 86 or len(p['global_index']) != 2291:
        raise ValueError('unexpected acquisition or object count')
    if len(np.unique(p['global_index'])) != 2291:
        raise ValueError('duplicated out-of-fold object')
    if not np.array_equal(p['global_index'], rep['global_index']):
        raise ValueError('representation and decoder row ordering differ')
    counts = np.random.default_rng(20261012).multinomial(86, np.full(86, 1/86), 10000)

    def totals(values):
        return np.bincount(inverse, weights=values, minlength=86)

    def bacc(predictions, labels):
        correct = (predictions == labels[None]).mean(0)
        value, draws = [], []
        for category in np.unique(labels):
            use = labels == category
            value.append(correct[use].mean())
            support = np.bincount(inverse[use], minlength=86)
            good = np.bincount(inverse[use], weights=correct[use], minlength=86)
            if not ((counts @ support) > 0).all():
                raise ValueError('bootstrap draw has no support for a class')
            draws.append((counts @ good)/(counts @ support))
        return float(np.mean(value)), np.mean(draws, axis=0)

    scores = {name: bacc(p[name], p['y_action']) for name in p.files if name.startswith(('reference_fixed_', 'refit_'))}
    for name in p.files:
        if name.startswith('category_'):
            scores[name] = bacc(p[name], p['y_material'])
    spectral = {}
    sst = loss['addition_zero_sst'][0]
    for cell in ['addition_zero', 'replacement_zero', 'addition_correct', 'addition_wrong', 'replacement_correct', 'replacement_wrong']:
        mean_sse = loss[cell+'_sse'].mean(0)
        spectral[cell] = (float(1-mean_sse.sum()/sst.sum()), 1-(counts @ totals(mean_sse))/(counts @ totals(sst)))
    recomputed=[]
    for row in expected['intervals']:
        comparison = row['comparison']
        if row['metric'] == 'within-category decoder spectral R2':
            coefficients = {
                'content by composition': {'addition_correct':1,'addition_wrong':-1,'replacement_correct':-1,'replacement_wrong':1},
                'addition correct minus donor': {'addition_correct':1,'addition_wrong':-1},
                'replacement correct minus donor': {'replacement_correct':1,'replacement_wrong':-1},
                'correct addition minus replacement': {'addition_correct':1,'replacement_correct':-1}}[comparison]
            values=spectral
        elif comparison == 'addition minus replacement':
            values=scores;coefficients={'category_addition_correct':1,'category_replacement_correct':-1}
        else:
            readout, clause=comparison.split(': '); values=scores
            if clause=='injection by composition':
                coefficients={readout+'_'+name:sign for name,sign in [('addition_correct',1),('replacement_correct',-1),('addition_zero',-1),('replacement_zero',1)]}
            else:
                a,b=clause.split(' minus ');coefficients={readout+'_'+a:1,readout+'_'+b:-1}
        point=sum(sign*values[name][0] for name,sign in coefficients.items())
        ci=np.quantile(sum(sign*values[name][1] for name,sign in coefficients.items()),[.025,.975])
        np.testing.assert_allclose([point,*ci],[row['delta'],row['ci95_low'],row['ci95_high']],atol=1e-12)
        recomputed.append({'comparison':comparison,'point':float(point),'ci95':ci.tolist()})
    rvalues={}
    for arm in ['continuous','posterior_code']:
        sse=rep[arm+'_sse'].mean(0);rst=rep[arm+'_sst'].mean(0)
        rvalues[arm]=(float(1-sse.sum()/rst.sum()),1-(counts @ totals(sse))/(counts @ totals(rst)))
    point=rvalues['continuous'][0]-rvalues['posterior_code'][0]
    ci=np.quantile(rvalues['continuous'][1]-rvalues['posterior_code'][1],[.025,.975])
    np.testing.assert_allclose([point,*ci],[rep_expected['continuous_minus_posterior_R2'],*rep_expected['paired_session_ci95']],atol=1e-12)
    result={'status':'PASS','objects':2291,'sessions':86,'paired_decoder_comparisons':recomputed,
            'continuous_minus_posterior_R2':point,'paired_session_ci95':ci.tolist(),
            'analysis_unit':'acquisition session; optimization seeds averaged first',
            'full_model_retraining_performed':False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({'status':'PASS','comparisons':len(recomputed),'objects':2291,'sessions':86}))


if __name__=='__main__':
    main()
