"""KneeScan web interface (Gradio). Upload a sagittal knee MRI and get ACL tear status and level, ACL location,
ACL length and width, and the femur-fibula distance, each from the model that validated best for that job.
Run locally:  python app.py   (then open http://localhost:7860)"""
import os, json, html, tempfile, traceback
import numpy as np
import gradio as gr
from kneescan.io import load_volume, LoadError
from kneescan.engine import KneeScan
from kneescan.render import acl_figure, bones_figure

ENGINE = KneeScan(threads=int(os.environ.get('KNEESCAN_THREADS', os.cpu_count() or 2)))
try: LEVEL_AUC = json.load(open(os.path.join(ENGINE.D, 'calibration.json')))['validation']['level']['partial_vs_complete_auc']
except Exception: LEVEL_AUC = '?'
EXAMPLES = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'examples')

CSS = """
.ks-cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 12px; }
.ks-card { border: 1px solid var(--border-color-primary); border-radius: 10px; padding: 14px 16px; background: var(--background-fill-secondary); }
.ks-card h4 { margin: 0 0 6px; font-size: .78rem; text-transform: uppercase; letter-spacing: .04em; color: var(--body-text-color-subdued); font-weight: 600; }
.ks-card .big { font-size: 1.55rem; font-weight: 650; line-height: 1.2; }
.ks-card .sub { font-size: .82rem; color: var(--body-text-color-subdued); margin-top: 4px; }
.ks-torn { color: #e5484d; } .ks-ok { color: #30a46c; } .ks-warn { color: #d97706; }
.ks-badge { display: inline-block; font-size: .7rem; padding: 1px 7px; border-radius: 99px; border: 1px solid currentColor; margin-left: 6px; vertical-align: middle; }
.ks-warnings { margin-top: 12px; font-size: .86rem; }
.ks-warnings li { margin: 3px 0; }
"""


def card(title, big, sub='', cls=''):
    return f'<div class="ks-card"><h4>{html.escape(title)}</h4><div class="big {cls}">{big}</div><div class="sub">{sub}</div></div>'


def summary_html(r):
    t, a, b, loc = r['tear'], r.get('acl', {}), r.get('femur_fibula'), r['acl_location']
    torn = t['status'] == 'torn'
    cards = [card('ACL tear', 'Torn' if torn else 'Intact', f"tear probability {t['probability']:.2f} (cut-off {t['threshold']:.2f})", 'ks-torn' if torn else 'ks-ok')]
    lvl = t.get('level', '')
    cards.append(card('Tear level', html.escape(lvl.capitalize()) + ('<span class="ks-badge ks-warn">low reliability</span>' if torn else ''),
                      f"partial vs complete is the weakest output (AUC {LEVEL_AUC})" if torn else 'no tear predicted'))
    cards.append(card('ACL location', f"Slice {loc['slice'] + 1}", f"box x {loc['box'][0]}-{loc['box'][2]}, y {loc['box'][1]}-{loc['box'][3]} px · localizer score {loc['score']:.2f}"))
    if 'length_px' in a:
        unit_mm = a.get('length_mm') is not None
        L = f"{a['length_mm']} mm" if unit_mm else f"{a['length_px']} px"; W = f"{a['width_mm']} mm" if unit_mm else f"{a['width_px']} px"
        src = a.get('mm_source', ''); flag = '' if a.get('passed_checks') else '<span class="ks-badge ks-warn">check outline</span>'
        cards.append(card('ACL length', L + flag, f"{a['length_px']} px · {src}"))
        cards.append(card('ACL width (mid-substance)', W + flag, f"{a['width_px']} px · {src}"))
    else:
        cards.append(card('ACL length / width', '—', html.escape(a.get('error', 'not measured'))))
    if b is not None:
        if 'distance_mm' in b:
            flag = '<span class="ks-badge ks-warn">experimental</span>' + ('' if b.get('passed_checks') else '<span class="ks-badge ks-warn">check outline</span>')
            cards.append(card('Femur → fibular head', f"{b['distance_mm']} mm{flag}", f"{b['distance_px']} px · slice {b['slice'] + 1} · {html.escape(b['mm_source'])}"))
        else:
            cards.append(card('Femur → fibular head', '—', html.escape(b.get('error', 'not measured'))))
    w = ''.join(f'<li>{html.escape(x)}</li>' for x in r.get('warnings', []))
    return f'<div class="ks-cards">{"".join(cards)}</div>' + (f'<ul class="ks-warnings">{w}</ul>' if w else '') + f'<div class="sub" style="margin-top:8px">Analysis took {r["seconds"]} s.</div>'


def validation_md():
    try: v = json.load(open(os.path.join(ENGINE.D, 'calibration.json')))['validation']
    except Exception: return ''
    t, l, lo, a = v.get('tear', {}), v.get('level', {}), v.get('location', {}), v.get('acl_size', {})
    ci = t.get('auc_95ci', ['?', '?'])
    return f"""
All tear numbers are for the **full automatic pipeline** (ACL found by the localizer, knee side unknown), each scan scored by a model that never trained on it.

| Output | Model | How it was checked | Result |
|---|---|---|---|
| ACL tear | MONAI 3D ResNet10, 5-model ensemble | KneeMRI, 917 scans (227 torn), 5-fold cross-validation grouped by patient | AUC {t.get('auc')} (95% CI {ci[0]}–{ci[1]}); sensitivity {t.get('sensitivity')}, specificity {t.get('specificity')}; complete vs healthy AUC {t.get('auc_complete_vs_healthy')}. With the hand-drawn ACL box and known knee side: AUC {t.get('auc_with_hand_drawn_acl_box_and_known_knee_side')} |
| Tear level | MONAI tear probability | Same scans, partial vs complete among torn | AUC {l.get('partial_vs_complete_auc')}; complete ruptures caught {l.get('complete_rupture_caught')}, partial tears labelled partial {l.get('partial_labelled_partial')} |
| ACL location | SegFormer ACL localizer (Hugging Face) + centre-slice prior | KneeMRI hand-marked ACL slices and boxes | slice inside the hand-marked ACL slices {lo.get('slice_inside_hand_marked_acl_slices')}, within one slice {lo.get('slice_within_one_of_them')}; median box overlap (IoU) {lo.get('box_overlap_iou_median')} (optimistic: the localizer was likely trained on these boxes) |
| ACL length / width | MedSAM (Meta SAM-based) + decoder fine-tuned on 110 expert ACL outlines | Held-out expert slices (CEMRR), 5-fold | Dice {a.get('dice')}; median length error {a.get('length_error_median_pct')}%, width error {a.get('width_error_median_pct')}%. Outlines can take in neighbouring fluid on fat-suppressed scans: check the overlay |
| Femur → fibular head | Registration to 3 reference knees + MedSAM outlines | No public ground truth exists | Experimental: check the overlay |

Research prototype, not for clinical use."""


def analyze(file, do_bones, progress=gr.Progress()):
    if file is None: raise gr.Error('Upload an MRI file first.')
    path = file if isinstance(file, str) else file.name
    try:
        progress(0.05, desc='Reading the MRI')
        d = load_volume(path, os.path.basename(path))
        if not ENGINE.loaded:
            progress(0.1, desc='Loading models (first run only)'); ENGINE.load()
        progress(0.2, desc='Finding the ACL, checking for a tear, outlining and measuring' + (' (femur-fibula adds ~1 min)' if do_bones else ''))
        r = ENGINE.analyze(d['vol'], d['spacing'], do_bones=do_bones)
    except LoadError as e:
        raise gr.Error(str(e))
    except Exception as e:
        traceback.print_exc(); raise gr.Error(f'Analysis failed: {e}')
    if 'error' in r: raise gr.Error(r['error'])
    r['input'] = dict(file=os.path.basename(path), format=d['fmt'], notes=d['notes'])
    eng = [w for w in r['warnings'] if not (w.startswith('No pixel spacing') and any('pixel spacing' in n for n in d['notes']))]
    r['warnings'] = d['notes'] + eng
    progress(0.95, desc='Drawing results')
    img_acl = acl_figure(d['vol'], r); img_b = bones_figure(d['vol'], r) if do_bones else None
    out = tempfile.NamedTemporaryFile(prefix='kneescan_', suffix='.json', delete=False, mode='w'); json.dump(r, out, indent=1); out.close()
    return summary_html(r), img_acl, img_b, out.name


with gr.Blocks(title='KneeScan', css=CSS, theme=gr.themes.Soft(primary_hue='red', neutral_hue='slate'), analytics_enabled=False) as demo:
    gr.Markdown('# KneeScan\nUpload a **sagittal knee MRI** to get ACL tear status and level, where the ACL is, its length and width, '
                'and the femur → fibular head distance. Accepts a **.zip of a DICOM series**, **.dcm**, **.nii / .nii.gz**, '
                'KneeMRI **.pck** or MRNet **.npy**. DICOM and NIfTI give real millimetres; the others are estimated. '
                '*Research prototype, not for clinical use.*')
    with gr.Row():
        with gr.Column(scale=1, min_width=280):
            f = gr.File(label='MRI file', file_types=['.zip', '.dcm', '.nii', '.gz', '.npy', '.pck'], type='filepath')
            bones = gr.Checkbox(value=True, label='Measure femur → fibular head (experimental, adds about a minute)')
            go = gr.Button('Analyze', variant='primary')
            ex = sorted(os.path.join(EXAMPLES, x) for x in os.listdir(EXAMPLES)) if os.path.isdir(EXAMPLES) else []
            if ex: gr.Examples(examples=[[e] for e in ex], inputs=[f], label='Example scans (KneeMRI, Clinical Hospital Centre Rijeka, CC BY-NC-ND 4.0)')
        with gr.Column(scale=2):
            summary = gr.HTML()
            gr.Markdown('**ACL** — located region (left), outline and length line (right)')
            img_acl = gr.Image(show_label=False, type='pil')
            gr.Markdown('**Femur → fibular head** — femur (yellow), tibia (blue), fibula (pink), shortest gap (white)')
            img_b = gr.Image(show_label=False, type='pil')
            js = gr.File(label='Full results (JSON)')
    with gr.Accordion('How each result is produced and how accurate it is', open=False):
        gr.Markdown(validation_md())
    go.click(analyze, inputs=[f, bones], outputs=[summary, img_acl, img_b, js])

if __name__ == '__main__':
    import sys
    share = '--share' in sys.argv or bool(os.environ.get('GRADIO_SHARE'))
    demo.queue(max_size=8).launch(server_name='0.0.0.0', server_port=int(os.environ.get('PORT', 7860)), share=share)
