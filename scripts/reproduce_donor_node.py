import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();root=args.root.resolve()
    data=root/'data' if (root/'data').exists() else root
    results=root/'results' if (root/'results').exists() else root
    donors=np.load(data/'donor_oof.npz')
    loss=np.load(data/'all_oof_spectral_losses.npz')
    pred=np.load(data/'all_oof_predictions.npz')
    expected=json.loads((results/'donor_node_summary.json').read_text())
    nodes=np.unique(donors['recipient_session'])
    if not np.isin(donors['donor_session'], nodes).all():
        raise ValueError('donor session is not present among acquisition nodes')
    u=np.searchsorted(nodes,donors['recipient_session'])
    v=np.searchsorted(nodes,donors['donor_session'])
    if len(nodes) != 86 or len(u) != 2291:
        raise ValueError('unexpected acquisition or object count')
    xi=np.random.default_rng(20261012).exponential(1.,size=(10000,len(nodes)))
    contrasts={
        'content by composition':-loss['addition_correct_sse']+loss['addition_wrong_sse']+loss['replacement_correct_sse']-loss['replacement_wrong_sse'],
        'addition correct minus donor':-loss['addition_correct_sse']+loss['addition_wrong_sse'],
        'replacement correct minus donor':-loss['replacement_correct_sse']+loss['replacement_wrong_sse']}
    draws={name:np.empty(10000) for name in contrasts}
    state_draws={f'{readout}: {comp}_correct minus {comp}_wrong':np.empty(10000)
                 for readout in ['reference_fixed','refit'] for comp in ['addition','replacement']}
    sst=loss['addition_zero_sst'][0]
    for start in range(0,10000,128):
        sample=xi[start:start+128]
        weights=[sample[:,u]*np.where(u==v[seed],1.,sample[:,v[seed]]) for seed in range(3)]
        denominator=sum(weight@sst/3 for weight in weights)
        for name,numerator in contrasts.items():
            draws[name][start:start+len(sample)]=sum(weights[seed]@numerator[seed]/3 for seed in range(3))/denominator
        for readout in ['reference_fixed','refit']:
            for comp in ['addition','replacement']:
                a=(pred[readout+'_'+comp+'_correct']==pred['y_action'][None]).astype(float)
                b=(pred[readout+'_'+comp+'_wrong']==pred['y_action'][None]).astype(float)
                recalls=[]
                for category in range(4):
                    use=pred['y_action']==category
                    n=sum(weights[seed][:,use]@(a-b)[seed,use]/3 for seed in range(3))
                    d=sum(weights[seed][:,use].sum(1)/3 for seed in range(3))
                    recalls.append(n/d)
                key=f'{readout}: {comp}_correct minus {comp}_wrong'
                state_draws[key][start:start+len(sample)]=np.mean(recalls,axis=0)
    all_draws={**draws,**state_draws}
    checked=[]
    for row in expected['intervals']:
        if row['comparison'] not in all_draws:
            continue
        interval=np.quantile(all_draws[row['comparison']],[.025,.975])
        np.testing.assert_allclose(interval,[row['ci95_low'],row['ci95_high']],atol=1e-12)
        checked.append({'comparison':row['comparison'],'ci95':interval.tolist()})
    if len(checked) != 7:
        raise ValueError('the expected seven intervals were not all checked')
    output={'status':'PASS','method':'independent row-wise shared-node multipliers; no edge aggregation reused',
            'acquisition_nodes':86,'objects':2291,'checked_intervals':checked,
            'no_training_or_parameter_selection':True}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output,indent=2),encoding='utf-8')
    print(json.dumps({'status':'PASS','independent_intervals':len(checked)}))


if __name__=='__main__':
    main()
