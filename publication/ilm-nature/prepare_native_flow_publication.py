#!/usr/bin/env python3
"""Stage an editable actual-model diagram and bounded failed-cohort result."""
import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import zipfile

from PIL import Image
from generate_assignment_editable_figures import NS, Slide, element, q, relpath, resolve, xml
from prepare_native_qa_publication import active_manifest, sha
from render_native_qa_publication import verify_stage

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
COHORT = ROOT/'references/evidence/native_visual_flow_final_normal'
COPY = ROOT/'references/evidence/native_visual_flow_final_copy'
TRAINING = ROOT/'references/evidence/native_visual_flow_training'


def stage(source, output):
    before = active_manifest()
    verify_stage(source)
    cohort = json.loads((COHORT/'report.json').read_text())
    copying = json.loads((COPY/'report.json').read_text())
    trained = json.loads((TRAINING/'training_report.json').read_text())
    audit = json.loads((TRAINING/'training_audit.json').read_text())
    for name, digest in cohort['files_sha256'].items():
        if sha(COHORT/name) != digest:
            raise ValueError('complete cohort evidence changed: '+name)
    for name, digest in copying['files_sha256'].items():
        if sha(COPY/name) != digest:
            raise ValueError('complete copying evidence changed: '+name)
    if (copying['cases'] != 64 or not copying['complete_copy_cohort_raw_review']
            or copying['semantic_learning_validated'] or copying['full384_assessment_completed']
            or copying['final_checkpoint_sha256'] != cohort['final_checkpoint_sha256']
            or copying['deployment_sha256'] != cohort['deployment_sha256']):
        raise ValueError('owning complete final copying cohort required')
    for split, exact, reference, english in [('train', 2, 27, 0), ('validation', 3, 29, 1)]:
        rows = [r for r in copying['rows'] if r['split'] == split]
        if (len(rows) != 32 or sum(r['canonical_exact'] for r in rows) != exact
                or sum(r['reference_OCR_exact'] for r in rows) != reference
                or copying['summary'][split]['canonical_exact'] != exact
                or copying['summary'][split]['reference_OCR_exact'] != reference):
            raise ValueError('complete copying counts differ')
        for language, count, clean_count in [('zh', 2, 14), ('en', english, reference-14)]:
            selected = [r for r in rows if r['language'] == language]
            if (len(selected) != 16 or sum(r['canonical_exact'] for r in selected) != count
                    or sum(r['reference_OCR_exact'] for r in selected) != clean_count
                    or copying['summary'][split]['languages'][language] != dict(cases=16,
                        canonical_exact=count, reference_OCR_exact=clean_count)):
                raise ValueError('complete copying language counts differ')
    if (cohort['cases'] != 64 or not cohort['complete_normal_cohort_raw_review']
            or cohort['full384_assessment_completed'] or cohort['semantic_learning_validated']
            or audit['completed_report_sha256'] != sha(TRAINING/'training_report.json')
            or trained['updates'] != 1536 or audit['actual_optimizer_updates'] != 1536
            or trained['checkpoint_sha256'] != cohort['final_checkpoint_sha256']):
        raise ValueError('owning complete training and bounded normal cohort required')
    for split in ('train', 'validation'):
        rows = [r for r in cohort['rows'] if r['split'] == split]
        pairs = {}
        for row in rows:
            pairs.setdefault(row['pair'], []).append(row)
        if (len(rows) != 32 or len(pairs) != 16 or any(len(v) != 2 for v in pairs.values())
                or any(r['accepted_exact'] or r['canonical_exact'] for r in rows)):
            raise ValueError('complete failed normal counts differ')
    counts = trained['runtime_parameters']['components']
    if (sum(counts.values()) != trained['runtime_parameters']['total']
            or trained['runtime_parameters'] != audit['actual_runtime_parameters']
            or trained['runtime_parameters']['total'] != 6240279775
            or trained['actual_trainable_parameters'] != 37380992 or audit['actual_adapter_parameters'] != 37380992):
        raise ValueError('actual full-runtime or adapter count differs')
    with zipfile.ZipFile(source/'draft.pptx') as archive:
        original = {name: archive.read(name) for name in archive.namelist()}
    parts = dict(original)
    presentation = ET.fromstring(parts['ppt/presentation.xml'])
    slides = presentation.find('p:sldIdLst', NS)
    if len(slides) != 45:
        raise ValueError('complete45-slide predecessor required')
    types = ET.fromstring(parts['[Content_Types].xml'])
    relationships = ET.fromstring(parts['ppt/_rels/presentation.xml.rels'])
    template = 'ppt/slides/slide43.xml'
    template_rels = ET.fromstring(parts[relpath(template)])
    slide = Slide(parts[template])
    pictures = []

    def picture(row, key, x, y, width):
        path = COHORT/row['images'][key]
        raw = path.read_bytes()
        part = 'ppt/media/native_flow_'+key+'.png'
        if part in parts:
            raise ValueError('new media collision')
        parts[part] = raw
        with Image.open(path) as image:
            height = width*image.height/image.width
        slide.picture(slide.relation('image', '../media/'+Path(part).name), key, x, y, width, height)
        pictures.append(dict(part=part, source=row['images'][key], sha256=sha(path), crop_box=None))

    slide.shape('Native image-answer flow: trained execution, failed answering', 45, 24, 1510, 60, 30, True)
    slide.shape('a  Actual input and sampling path', 45, 108, 1510, 35, 22, True)
    slide.shape('Complete input page x', 45, 155, 235, 35, 18, True)
    row = cohort['rows'][32]
    if row['id'] != 'TRAIN_2295_QUERY_0':
        raise ValueError('first fixed held-out case differs')
    picture(row, 'input', 45, 205, 195)
    slide.shape('1024 × 1024 RGB', 45, 405, 235, 32, 17)
    nodes = [(300, 215, 'Frozen image /\nlanguage reader\n'+f'{counts["lmm"]/1e9:.3f}B parameters', 'F0F3F5'),
             (555, 190, 'Connector\n'+f'{counts["connector_module"]/1e9:.3f}B parameters'+'\nLoRA, rank 16', 'EAF1F6'),
             (785, 220, 'Image denoiser\n'+f'{counts["transformer"]/1e9:.3f}B parameters'+'\nLoRA, rank 16', 'EAF1F6'),
             (1045, 170, 'Frozen VAE\n'+f'{counts["vae"]/1e9:.3f}B parameters', 'F0F3F5')]
    for x, width, words, color in nodes:
        slide.shape(words, x, 225, width, 130, 18, center=True, fill=color)
    for a, b in [(245, 295), (520, 550), (750, 780), (1010, 1040), (1220, 1260)]:
        slide.arrow(a, 290, b, 290)
    slide.shape('50 steps', 995, 170, 120, 30, 16)
    slide.shape('Actual answer image', 1265, 155, 290, 35, 18, True)
    picture(row, 'answer', 1270, 205, 260)
    slide.shape('Full unrepaired page', 1270, 470, 280, 32, 16)
    slide.arrow(140, 400, 140, 455, head=False)
    slide.arrow(140, 455, 895, 455, head=False)
    slide.arrow(895, 455, 895, 360)
    slide.shape('Reference latent r(x): same frozen VAE encoder', 295, 405, 620, 36, 18)
    slide.shape('b  Training-only image supervision', 45, 510, 880, 36, 22, True)
    slide.shape('Answer image y → frozen VAE encoder E(y)', 45, 558, 880, 38, 21)
    slide.shape('zσ = (1 − σ) E(y) + σ ε        v* = ε − E(y)', 45, 610, 880, 40, 23)
    slide.shape('Loss = MSE(vθ(zσ, t, x), v*)', 45, 662, 880, 40, 23)
    slide.shape('One logical denoiser call per update; positive task branch.\nLoss updates connector / writer adapters; reader and VAE stay frozen.',
                45, 720, 895, 75, 18)
    slide.shape('c  Final question and copying cohorts', 975, 510, 580, 36, 22, True)
    for i, (label, count) in enumerate([('Seen questions', '0/32'), ('Held-out questions', '0/32'),
            ('Both held-out answers correct', '0/16'), ('Seen copying strings', '2/32'),
            ('Held-out copying strings', '3/32')]):
        slide.shape(label, 975, 556+32*i, 420, 30, 18)
        slide.shape(count, 1400, 556+32*i, 150, 30, 21, True)
    slide.shape(f'{trained["elapsed_seconds_this_attempt"]/3600:.2f} h training; {trained["peak_allocated_vram_gib_this_attempt"]:.3f} GiB peak\n'
                '37.381M trained adapters; 6.240B retained runtime\nQA reference OCR: 23/32 seen; 26/32 held out\n'
                'Copy reference OCR: 27/32 seen; 29/32 held out', 975, 728, 580, 100, 17)
    slide.shape('1,536 updates on 384 pairs. Inherited language/image pretraining; source/control comparison pending. Zero counts use source-accepted OCR; no reliable language learning established.',
                45, 844, 1510, 48, 15)
    layout = next(n.get('Target') for n in template_rels if n.get('Type').endswith('/slideLayout'))
    slide.relation('slideLayout', layout)
    number = 46
    note = 'ppt/notesSlides/native_flow_notes46.xml'
    part = 'ppt/slides/slide46.xml'
    slide.relation('notesSlide', '../notesSlides/'+Path(note).name)
    old_note = next(resolve(template, n.get('Target')) for n in template_rels if n.get('Type').endswith('/notesSlide'))
    notes = ET.fromstring(parts[old_note])
    caption = ('Figure S31. Actual retained DeepGen image-answer path and completed final question/copying cohorts. '
        'The input image supplies semantic features and a separate reference-image latent. The same frozen VAE encodes supervised target images and decodes sampled images. '
        'One native positive-branch velocity MSE call is used per update; fifty-step guidance-four inference is separate. Only connector/writer low-rank adapters learn. '
        'First held-out case TRAIN_2295_QUERY_0, selected by original index, is shown as complete unmodified input/output PNGs. '
        'All64 final normal pages and all64 final copying pages were inspected with source access; no quantitative manual accuracy is assigned. '
        '0/32 seen,0/32 held,0/16 both-correct held pairs are source-accepted offline OCR counts. Reference OCR23/32 and26/32 remains imperfect. '
        'Final copying OCR counts are2/32 seen and3/32 held out; clean references give27/32 and29/32. '
        'Copying uses synthetic strings sharing vocabulary, font and prompt template; it does not test sentence understanding. '
        'Full initial/final erasure/copying comparison and later diagnostics are pending. No foundation-training, historical-script, novelty or language-efficiency claim. '
        'Sources: references/evidence/native_visual_flow_training/training_report.json, references/evidence/native_visual_flow_final_normal/report.json '
        'and references/evidence/native_visual_flow_final_copy/report.json.')
    for shape in notes.findall('p:cSld/p:spTree/p:sp', NS):
        placeholder = shape.find('p:nvSpPr/p:nvPr/p:ph', NS)
        if placeholder is not None and placeholder.get('type') == 'body':
            for i, text in enumerate(shape.findall('.//a:t', NS)):
                text.text = caption if i == 0 else ''
    note_rels = ET.fromstring(parts[relpath(old_note)])
    for n in note_rels:
        if n.get('Type').endswith('/slide'):
            n.set('Target', '../slides/slide46.xml')
    for name, kind in [(part, 'slide'), (note, 'notesSlide')]:
        if name in parts:
            raise ValueError('new slide/notes collision')
        element('ct:Override', types, PartName='/'+name, ContentType='application/vnd.openxmlformats-officedocument.presentationml.'+kind+'+xml')
    parts[part], parts[relpath(part)] = xml(slide.root), xml(slide.rels)
    parts[note], parts[relpath(note)] = xml(notes), xml(note_rels)
    used = {n.get('Id') for n in relationships}
    rid = next(f'rId{i}' for i in range(1, 1000) if f'rId{i}' not in used)
    element('pkg:Relationship', relationships, Id=rid, Type=NS['r']+'/slide', Target='slides/slide46.xml')
    element('p:sldId', slides, id=max(int(n.get('id')) for n in slides)+1).set(q('r:id'), rid)
    for name, node in [('[Content_Types].xml', types), ('ppt/presentation.xml', presentation), ('ppt/_rels/presentation.xml.rels', relationships)]:
        parts[name] = xml(node)
    output.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(output/'draft.pptx', 'w', zipfile.ZIP_DEFLATED) as archive:
        for name, raw in parts.items():
            archive.writestr(name, raw)
    (output/'results_native_flow.tex').write_text(r'''A separate native image-answer flow adaptation completes 1,536 updates on
384 image pairs using 37.381M low-rank adapter parameters within a retained
6.240B-parameter pretrained system (Supplementary Note 53; Fig.~S31).
The complete final normal-question cohort yields 0/32 source-accepted
OCR-exact seen answers, 0/32 held-out answers and 0/16 both-correct held-out
passage pairs. Copying gives 2/32 seen and 3/32 held-out exact strings;
both complete 64-page output cohorts are inspected. Training takes
3.18\,h at 15.697\,GiB peak allocation, including loading, saving and fixed
monitoring. Reference OCR remains imperfect. Initial/source controls and
later field/sampling diagnostics are separate pending observations; this
completed cohorts do not establish useful language learning.
''')
    (output/'supplementary_note_53.tex').write_text(r'''\section*{Supplementary Note 53 | Native image-answer flow adaptation}
\label{note:native-visual-flow}
The retained pretrained reader supplies image-derived semantic conditions;
a separate frozen VAE encoding supplies reference latents from the input page.
Changing questions enter only as image pixels, with a fixed generic task
condition and removed unused lexical table, output head and tokenizer.
The frozen language/image body retains its external pretraining. This is
adapter learning on an existing system, not foundation training from pixels.

For supervised answer image $y$, the frozen VAE gives $z_y=E(y)$.
The native shifted schedule defines
\[
z_\sigma=(1-\sigma)z_y+\sigma\epsilon,\qquad
v^*=\epsilon-z_y,\qquad
\mathcal L=\operatorname{MSE}(v_\theta(z_\sigma,t_\sigma,x),v^*).
\]
Each update uses one logical positive-task denoiser call, with native
checkpoint recomputation in backward. Independent-noise inference uses
fifty native steps and guidance four; it is not denoising a supplied answer.
No answer transcript, teacher, OCR engine or font lookup enters deployment.
Supervised targets are loaded separately from input-only conditions.

Rank-16 connector/writer adapters use AdamW with rate $5\times10^{-5}$,
betas $(.9,.95)$, decay $.05$ and norm clipping at one. Four complete
permutations of 384 image pairs give 1,536 updates: 256 Chinese QA pairs from 128
articles and 128 Chinese/English copying pairs. Sixteen held-out QA articles
never enter this optimizer schedule. Both canvases are $1024\times1024$; original
$512\times256$ answer images are padded with white without resizing their writing.
There are 852 verified adapter tensors and optimizer states, all at step 1,536.
The complete retained runtime has 6,240,279,775 parameters; 37,380,992 adapt.
Whole-attempt time is 11,450.839\,s at 15.697\,GiB peak allocated memory,
including source loading, saving and 24 fixed seen monitoring images.

The completed final normal cohort contains 32 seen questions and 32 questions
from disjoint development articles. Canonical and source-accepted OCR exact
counts are zero in both splits; both-correct passage-pair counts are 0/16 in
both. Every generated PNG authenticates against its actual decoder tensor.
All 64 complete pages were inspected, with malformed writing and repeated
fragments retained. Source and OCR access prevent an independent blinded
review claim, and no manual answer-accuracy score is assigned. Clean reference
OCR gives 23/32 seen and 26/32 held out. Development cases have been reused;
these observations are not a new untouched benchmark.

The completed final copying cohort contains 32 seen strings and 32 held-out
strings, each split balanced between Chinese and English. Every original
output page was inspected and authenticated against its actual decoder tensor.
Counts under the unchanged OCR exact-match rule are:
\begin{center}
\begin{tabular}{lrrrr}
 & Seen & Held out & Reference seen & Reference held out \\
\hline
Chinese & 2/16 & 2/16 & 14/16 & 14/16 \\
English & 0/16 & 1/16 & 13/16 & 15/16 \\
Combined & 2/32 & 3/32 & 27/32 & 29/32 \\
\end{tabular}
\end{center}
The copying strings share a character/word inventory, rendering font and
instruction template. Held-out strings do not enter optimizer updates, but
these reused development examples do not test unseen scripts or sentence
understanding. Visible errors include omitted forms, merged strokes and extra
punctuation. No manual accuracy or repaired OCR score is assigned.

The complete source/final erasure/copying assessment and subsequent diagnostics
are pending and cannot be inferred from these final cohorts. They supply
no complete initial/final learning-gain estimate or causal attribution.
The original complete evidence remains under
\texttt{references/evidence/native\_visual\_flow\_final\_normal/}, with
the corresponding copying export under
\texttt{references/evidence/native\_visual\_flow\_final\_copy/}.
All input, answer and reference PNG bytes and the full training audit are retained.
Historical-script semantics and efficiency at matched language quality remain
unverified. Figure~S31 distinguishes the implemented training path from these
unachieved goals.
''')
    mutable = {'[Content_Types].xml', 'ppt/presentation.xml', 'ppt/_rels/presentation.xml.rels'}
    with zipfile.ZipFile(output/'draft.pptx') as archive:
        for name, raw in original.items():
            if name not in mutable and archive.read(name) != raw:
                raise ValueError('predecessor part changed: '+name)
        for name in archive.namelist():
            if name.endswith(('.xml', '.rels')):
                ET.fromstring(archive.read(name))
        if len(ET.fromstring(archive.read('ppt/presentation.xml')).find('p:sldIdLst', NS)) != 46:
            raise ValueError('46 staged slides required')
        for picture in pictures:
            if hashlib.sha256(archive.read(picture['part'])).hexdigest() != picture['sha256']:
                raise ValueError('original evidence PNG changed')
    if active_manifest() != before:
        raise ValueError('active publication files changed during isolated preparation')
    record = dict(builder_sha256=sha(__file__), source_stage_preparation_sha256=sha(source/'preparation.json'),
        source_cohort_sha256=sha(COHORT/'report.json'), source_training_report_sha256=sha(TRAINING/'training_report.json'),
        source_copy_cohort_sha256=sha(COPY/'report.json'),
        source_training_audit_sha256=sha(TRAINING/'training_audit.json'), preserved_native_slides=45,
        staged_native_slides=46, preserved_native_charts=52, new_native_chart=False,
        new_figure='S31_native_visual_flow', pictures=pictures, native_diagram_and_count_table=True,
        old_package_parts_byte_identical_except_required_package_index=True,
        output_files_sha256={p.name: sha(p) for p in output.iterdir() if p.is_file()},
        active_publication_files_unchanged=True, bundled_artifact_tool_available=False,
        Office_or_LaTeX_started=False, neural_models_loaded=0, rendered=False,
        visual_review_completed=False, active_manuscript_updated=False, Nutstore_updated=False,
        full_model_assessment_complete=False, semantic_learning_validated=False, general_ilm_goal_complete=False)
    (output/'preparation.json').write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps({k: record[k] for k in ('preserved_native_slides', 'staged_native_slides', 'preserved_native_charts', 'rendered')}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    stage(args.source.resolve(), args.output.resolve())
