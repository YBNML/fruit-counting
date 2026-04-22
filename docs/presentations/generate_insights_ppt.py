"""Generate a research-oriented PowerPoint summarizing theoretical background
and key insights from the fruit-counting project (Plans 1-4).

Focus: theory (SAM, CLIP, PseCo architecture) + insights (what we discovered
about fine-tuning, coordinate systems, CLIP alignment fragility).

Run: python docs/presentations/generate_insights_ppt.py
Output: docs/presentations/fruit_counting_insights.pptx
"""

from __future__ import annotations

from pathlib import Path

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN
from pptx.dml.color import RGBColor

OUT = Path(__file__).resolve().parent / "fruit_counting_insights.pptx"

# Muted research colors
C_TITLE = RGBColor(0x1F, 0x3B, 0x5C)       # navy
C_HEADER = RGBColor(0x2E, 0x5F, 0x8A)      # steel blue
C_BODY = RGBColor(0x2A, 0x2A, 0x2A)        # near-black
C_ACCENT = RGBColor(0xC2, 0x56, 0x2D)      # rust
C_MUTED = RGBColor(0x6B, 0x6B, 0x6B)       # gray


def add_title_slide(prs, title, subtitle):
    slide = prs.slides.add_slide(prs.slide_layouts[6])  # blank
    # Title
    tb = slide.shapes.add_textbox(Inches(0.7), Inches(2.2), Inches(12), Inches(1.6))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.LEFT
    r = p.add_run()
    r.text = title
    r.font.size = Pt(40)
    r.font.bold = True
    r.font.color.rgb = C_TITLE

    # Subtitle
    tb2 = slide.shapes.add_textbox(Inches(0.7), Inches(4.0), Inches(12), Inches(1.5))
    tf2 = tb2.text_frame
    tf2.word_wrap = True
    p2 = tf2.paragraphs[0]
    p2.alignment = PP_ALIGN.LEFT
    r2 = p2.add_run()
    r2.text = subtitle
    r2.font.size = Pt(20)
    r2.font.color.rgb = C_MUTED

    # Footer
    tb3 = slide.shapes.add_textbox(Inches(0.7), Inches(6.8), Inches(12), Inches(0.5))
    tf3 = tb3.text_frame
    r3 = tf3.paragraphs[0].add_run()
    r3.text = "Internal research notes · 2026-04-22"
    r3.font.size = Pt(11)
    r3.font.color.rgb = C_MUTED
    r3.font.italic = True


def add_section_slide(prs, section_number, section_title):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    tb = slide.shapes.add_textbox(Inches(0.7), Inches(3.0), Inches(12), Inches(1.2))
    tf = tb.text_frame
    p = tf.paragraphs[0]
    r = p.add_run()
    r.text = f"Part {section_number}"
    r.font.size = Pt(16)
    r.font.color.rgb = C_MUTED

    p = tf.add_paragraph()
    r = p.add_run()
    r.text = section_title
    r.font.size = Pt(36)
    r.font.bold = True
    r.font.color.rgb = C_TITLE


def add_content_slide(prs, title, bullets, footer=None):
    """bullets: list of (level:int, text:str) or dict with 'run-style' info.
    Level 0 = top bullet, level 1 = sub-bullet."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])

    # Title bar
    tb = slide.shapes.add_textbox(Inches(0.5), Inches(0.3), Inches(12.5), Inches(0.8))
    tf = tb.text_frame
    r = tf.paragraphs[0].add_run()
    r.text = title
    r.font.size = Pt(26)
    r.font.bold = True
    r.font.color.rgb = C_HEADER

    # Body
    body = slide.shapes.add_textbox(Inches(0.7), Inches(1.2), Inches(12.0), Inches(5.8))
    bt = body.text_frame
    bt.word_wrap = True
    first = True
    for item in bullets:
        if isinstance(item, tuple):
            level, text = item
        else:
            level, text = 0, item
        if first:
            p = bt.paragraphs[0]
            first = False
        else:
            p = bt.add_paragraph()
        p.level = level
        r = p.add_run()
        r.text = ("• " if level == 0 else "– ") + text
        r.font.size = Pt(18 if level == 0 else 15)
        r.font.color.rgb = C_BODY if level == 0 else C_MUTED

    if footer:
        tb2 = slide.shapes.add_textbox(Inches(0.7), Inches(7.0), Inches(12), Inches(0.4))
        tf2 = tb2.text_frame
        r2 = tf2.paragraphs[0].add_run()
        r2.text = footer
        r2.font.size = Pt(11)
        r2.font.italic = True
        r2.font.color.rgb = C_MUTED


def add_table_slide(prs, title, headers, rows, footer=None, col_widths=None):
    slide = prs.slides.add_slide(prs.slide_layouts[6])

    # Title
    tb = slide.shapes.add_textbox(Inches(0.5), Inches(0.3), Inches(12.5), Inches(0.8))
    r = tb.text_frame.paragraphs[0].add_run()
    r.text = title
    r.font.size = Pt(26)
    r.font.bold = True
    r.font.color.rgb = C_HEADER

    n_rows = len(rows) + 1
    n_cols = len(headers)
    left, top = Inches(0.7), Inches(1.3)
    width = Inches(12.0)
    height = Inches(min(5.5, 0.5 + 0.45 * n_rows))

    table_shape = slide.shapes.add_table(n_rows, n_cols, left, top, width, height)
    table = table_shape.table

    if col_widths:
        for i, w in enumerate(col_widths):
            table.columns[i].width = Inches(w)

    # Header
    for j, h in enumerate(headers):
        cell = table.cell(0, j)
        cell.text = ""
        p = cell.text_frame.paragraphs[0]
        run = p.add_run()
        run.text = h
        run.font.size = Pt(14)
        run.font.bold = True
        run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        cell.fill.solid()
        cell.fill.fore_color.rgb = C_HEADER

    for i, row in enumerate(rows, start=1):
        for j, val in enumerate(row):
            cell = table.cell(i, j)
            cell.text = ""
            p = cell.text_frame.paragraphs[0]
            run = p.add_run()
            run.text = str(val)
            run.font.size = Pt(12)
            run.font.color.rgb = C_BODY
            if i % 2 == 0:
                cell.fill.solid()
                cell.fill.fore_color.rgb = RGBColor(0xF2, 0xF2, 0xF2)

    if footer:
        tb2 = slide.shapes.add_textbox(Inches(0.7), Inches(7.0), Inches(12), Inches(0.4))
        r2 = tb2.text_frame.paragraphs[0].add_run()
        r2.text = footer
        r2.font.size = Pt(11)
        r2.font.italic = True
        r2.font.color.rgb = C_MUTED


def add_two_col_slide(prs, title, left_heading, left_bullets,
                      right_heading, right_bullets):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    # Title
    tb = slide.shapes.add_textbox(Inches(0.5), Inches(0.3), Inches(12.5), Inches(0.8))
    r = tb.text_frame.paragraphs[0].add_run()
    r.text = title
    r.font.size = Pt(26)
    r.font.bold = True
    r.font.color.rgb = C_HEADER

    def _col(left_in, heading, items):
        tb = slide.shapes.add_textbox(
            Inches(left_in), Inches(1.3), Inches(6.1), Inches(5.8)
        )
        tf = tb.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        hr = p.add_run()
        hr.text = heading
        hr.font.size = Pt(18)
        hr.font.bold = True
        hr.font.color.rgb = C_ACCENT
        for it in items:
            if isinstance(it, tuple):
                level, text = it
            else:
                level, text = 0, it
            p = tf.add_paragraph()
            p.level = level
            r2 = p.add_run()
            r2.text = ("• " if level == 0 else "– ") + text
            r2.font.size = Pt(14 if level == 0 else 12)
            r2.font.color.rgb = C_BODY if level == 0 else C_MUTED

    _col(0.7, left_heading, left_bullets)
    _col(7.0, right_heading, right_bullets)


def main() -> None:
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)

    # === Title ===
    add_title_slide(
        prs,
        "Fruit Counting with PseCo:\nA Case Study in CLIP-Aligned Classification",
        "Theoretical background + insights from four iterations of training, "
        "diagnosis, and production engineering on orchard imagery",
    )

    # =============================================================
    # Part 1: Problem & Theoretical Background
    # =============================================================
    add_section_slide(prs, 1, "Problem · Architecture · Theory")

    add_content_slide(
        prs,
        "The Problem",
        [
            "Goal: count protective-bagged fruits in orchard photos from a deploy camera.",
            "Original code path: 5-stage pipeline (Deblur → PseCo → Super-Resolution → Classifier → Defect).",
            (1, "Each stage was a thin wrapper around an external model — inference-only."),
            (1, "No training code, no labeled orchard data, no systematic evaluation."),
            "Our task: redesign + add training + validate on real orchard imagery.",
            "Core research question: can an open-world counting model be domain-adapted without sacrificing generality?",
        ],
        footer="Spec: docs/superpowers/specs/2026-04-19-fruit-counting-redesign-design.md",
    )

    add_content_slide(
        prs,
        "Theoretical Components (1/3) — SAM (Segment Anything Model)",
        [
            "Meta AI, 2023. Foundation model for class-agnostic segmentation.",
            (1, "ViT-H image encoder: 1024×1024 input → (256, 64, 64) feature map."),
            (1, "Promptable mask decoder: accepts points / boxes / text as prompts."),
            (1, "Trained on 1B+ masks (SA-1B). Strong generalization to unseen domains."),
            "For counting: SAM provides object proposals given a candidate point.",
            (1, "`forward_sam_with_embeddings(features, points=...)` returns k=5 plausible masks per point."),
            (1, "Anchor boxes (point ± half_side) refine the mask hypotheses."),
            "Key property: SAM doesn't know WHAT an object is — only where an object-like blob is.",
        ],
    )

    add_content_slide(
        prs,
        "Theoretical Components (2/3) — CLIP Text Alignment",
        [
            "OpenAI, 2021. Image encoder + text encoder jointly trained on 400M (image, caption) pairs.",
            (1, "Text encoder maps class name → 512-dim embedding (for ViT-B/32 backbone)."),
            (1, "Image encoder maps crops to the same 512-dim space."),
            (1, "Zero-shot classification via cosine similarity in shared space."),
            "In PseCo: the ROIHeadMLP's 512-dim output is compared against CLIP text features as the classifier.",
            (1, "dot(ROI_feature, text_feature) → score. Higher = matches the class name."),
            "Subtle point: OpenAI CLIP weights were trained with QuickGELU activation, not standard GELU.",
            (1, "`open_clip.create_model('ViT-B-32')` defaults to standard GELU → text features drift silently."),
            (1, "Must use `'ViT-B-32-quickgelu'` variant. We learned this the hard way in Plan 3."),
        ],
    )

    add_content_slide(
        prs,
        "Theoretical Components (3/3) — PseCo Pipeline (ICLR 2024)",
        [
            "\"Point, Segment, Count\" (Huang et al., 2024).",
            "Three-stage inference on cached SAM features:",
            (1, "PointDecoder: dense heatmap → top-K candidate points (post-NMS)."),
            (1, "SAM-as-proposal: for each point, SAM emits 5 mask/box hypotheses (varying anchor sizes)."),
            (1, "ROIHeadMLP + CLIP: score each (region, text) pair → NMS → threshold."),
            "Why it works for counting:",
            (1, "Class-agnostic SAM handles novel domains (orchard) without retraining its encoder."),
            (1, "Text prompt decides what to count — no re-training needed to switch from \"apples\" to \"bread rolls\"."),
            (1, "Decoupling: localization (SAM + PointDecoder) is shared; classification (MLP + CLIP) is swappable."),
            "Training target in upstream: BCE on pseudo-box IoU + CLIP-consistency loss (preserves alignment).",
        ],
        footer="github.com/Hzzone/PseCo — reports MAE 16–20 on FSC-147 test split",
    )

    # =============================================================
    # Part 2: Our Experimental Iterations
    # =============================================================
    add_section_slide(prs, 2, "Four Iterations on FSC-147 + Orchard")

    add_content_slide(
        prs,
        "Our Plan Timeline",
        [
            "Plan 1 (Foundation): scaffolding, CLI, config, data diagnostics. No training.",
            "Plan 2 (Training infrastructure): SAM feature cache (8.7 GB fp16 for 4945 FSC-147 images), Runner, TensorBoard, checkpoints. Placeholder objective.",
            "Plan 3 (First real fine-tune): CLIP text prompts + pos/neg labels from box-point containment.",
            (1, "Three bugs caught: CUDA device mismatch, coordinate-system mismatch, QuickGELU variant mismatch."),
            (1, "Final FSC-147 val/mae = 47.62 with K=32 proposals. Beat 'predict-zero' baseline of 50.58."),
            (1, "Looked like a success at this point."),
            "Plan 4 (Orchard reality check): four hand-counted orchard images exposed Plan 3's hidden failure.",
            (1, "Pivoted to upstream MLP + prompt engineering + SAM anchor tuning."),
            (1, "Final orchard MAE = 11.75 without any fine-tuning."),
        ],
    )

    add_table_slide(
        prs,
        "Iteration-by-Iteration Numbers",
        headers=[
            "Version", "FSC-147 val/mae",
            "Orchard MAE (4 imgs)", "Takeaway",
        ],
        rows=[
            ("Plan 2 (placeholder)", "50.58 (deg.)", "—", "degenerate: predicts 0 always"),
            ("Plan 3 v1 (no fix)", "62.90", "—", "coord-system bug"),
            ("Plan 3 v2 (+ QuickGELU)", "62.79", "—", "marginal"),
            ("Plan 3 v3 (+ coord fix)", "54.26", "—", "coord fix dominant"),
            ("Plan 3 v4 (K=32)", "47.62", "400+ ❌", "prompt-agnostic in prod"),
            ("Plan 4 upstream MLP (default)", "—", "14.25", "zero training, still competitive"),
            ("Plan 4 + threshold tune", "—", "13.50", "small grid-search gain"),
            ("Plan 4 + anchor_size=24", "—", "11.75 ⭐", "scale matters"),
        ],
        col_widths=[3.4, 2.4, 2.7, 3.4],
    )

    # =============================================================
    # Part 3: Key Insights — the core research takeaways
    # =============================================================
    add_section_slide(prs, 3, "Insights from the Failure Modes")

    add_content_slide(
        prs,
        "Insight 1 — \"Predict Zero\" is a Deceptive Baseline",
        [
            "On count datasets, predicting 0 for every image gives MAE = E[GT count].",
            (1, "For FSC-147 val, mean GT count ≈ 50.58."),
            (1, "Plan 2's placeholder trainer produced `val/mae = 50.58` constant across epochs."),
            (1, "Naive reading: \"training is stuck at a non-trivial loss\". True reading: model is collapsed."),
            "Plan 3 produced `val/mae = 47.62` — only marginally better than zero-prediction.",
            (1, "We celebrated this as beating baseline; it was actually still ~93% of the zero-prediction error."),
            "Research implication: for counting tasks, report MAE relative to predict-zero AND relative to predict-mean.",
            (1, "A model that beats predict-zero but not predict-mean is probably memorizing the dataset mean."),
            (1, "A convergence metric should subtract that trivial baseline explicitly."),
        ],
    )

    add_content_slide(
        prs,
        "Insight 2 — CLIP Alignment is Fragile Under Naive Fine-tuning",
        [
            "Plan 3 training: one class per image, binary cross-entropy on (positive / negative) proposal labels.",
            "Result: fine-tuned MLP became prompt-agnostic.",
            (1, "`apples`, `apple in paper bag`, `protective fruit bag` all produced ~500 predictions on same apple image."),
            (1, "Upstream MLP in contrast distinguishes: 6 / 25 / 8 on the same image."),
            "Mechanism: binary CE with only one active class drives the head to ignore text features.",
            (1, "Gradient only penalizes 'visual feature says positive/negative', never 'text agrees/disagrees'."),
            (1, "CLIP alignment in the pretrained weights slowly decays during fine-tuning."),
            "Upstream PseCo's training uses a secondary CLIP consistency loss (`cls_loss2`) — we skipped this.",
            (1, "Research question: can we quantify CLIP alignment loss during fine-tuning and stop early?"),
            (1, "Candidate metric: prompt-discrimination score on a held-out triplet (image, correct class, wrong class)."),
        ],
    )

    add_content_slide(
        prs,
        "Insight 3 — Coordinate-System Bugs Silently Waste Training",
        [
            "Plan 3 labels came from FSC-147 point annotations in original image space (longest side = 384 px).",
            "PointDecoder predictions came from SAM's 1024-px padded space.",
            (1, "Box-point containment check compared them as if they were in the same coordinate frame."),
            (1, "Almost no containment hits. Nearly all proposals labeled negative."),
            "Symptom: training loss went down (easy to say 'all negative'), val/mae stuck at 62–63.",
            (1, "A bug that SAVES MONEY on loss by making the problem trivially learnable."),
            "Fix: scale GT points to 1024 space via `1024 / max(H, W)`. Single commit, val/mae dropped 62.79 → 54.26.",
            "Research-engineering lesson:",
            (1, "Multi-module pipelines mix coordinate frames silently. Write a single test that asserts frame equality."),
            (1, "\"Losses going down\" alone is NEVER evidence of correctness."),
        ],
    )

    add_content_slide(
        prs,
        "Insight 4 — QuickGELU vs GELU: a 0.3% Activation Difference with 100% Semantic Drift",
        [
            "OpenAI's original CLIP was trained with QuickGELU: `x * sigmoid(1.702 * x)`.",
            "`open_clip.create_model('ViT-B-32', pretrained='openai')` defaults to standard GELU.",
            (1, "Text embeddings change by only ~2% in L2 distance."),
            (1, "But the fine-tuned MLP head, trained in downstream tasks against the QuickGELU-derived embeddings, is spec-mismatched."),
            "Upstream PseCo's `MLP_small_box_w1_zeroshot.tar` was trained with QuickGELU CLIP features.",
            (1, "We loaded it against standard-GELU features for 2 training runs before spotting the warning."),
            (1, "Fixing this alone was not enough to recover CLIP alignment (see Insight 2), but it is a necessary condition."),
            "Research-engineering lesson:",
            (1, "Treat activation-variant tags in pretrained models as semver-breaking. Document them in checkpoints."),
        ],
    )

    add_content_slide(
        prs,
        "Insight 5 — SAM Anchor Size Must Match Object Scale",
        [
            "PseCo's inference uses anchor boxes (point ± half_side) as prompts for SAM's box-refinement head.",
            "Upstream demo: anchor_size = 8 on 1024-padded input (implies ~8 px half-side, ~16 px full side).",
            (1, "Fine for FSC-147 where images are 384 longest side — objects are ~10–30 px."),
            (1, "Breaks on phone-taken orchard photos where objects span ~80–120 px in 1024 space."),
            "Grid search over {4, 8, 12, 16, 24, 32}: apple detection monotone-improves with larger anchors up to ~24.",
            (1, "apple_1 (GT 48): 22 → 28 predictions as anchor goes 8 → 24."),
            (1, "Pears (smaller relative scale) unchanged."),
            "Why SAM is scale-sensitive here:",
            (1, "SAM was trained to segment from prompts that are INSIDE the object. Small anchors around a correct point still lie inside apples, but the mask-decoder's best hypothesis matches the anchor's scale."),
            (1, "Research question: scale-adaptive anchor ensembles? Multi-scale voting?"),
        ],
    )

    add_content_slide(
        prs,
        "Insight 6 — Prompt Engineering Can Replace Fine-tuning When Fine-tuning Would Break Alignment",
        [
            "On orchard images, upstream MLP MAE by prompt:",
            (1, "`pears` → 35.0 · `pear in paper bag` → 6.0"),
            (1, "`apples` → 42.5 · `apple in paper bag` → 22.5"),
            (1, "`protective fruit bag` → 40.0"),
            "A three-word prompt change saves ~30 MAE points on pears. No retraining required.",
            "Mechanism: CLIP's text encoder captures compositional semantics (fruit-class × environment).",
            (1, "\"apple in paper bag\" activates BOTH the apple cluster and the packaging context."),
            (1, "\"apples\" alone activates the apple cluster — less specific → overshoots on wide scenes."),
            "Implication: for class-agnostic counting pipelines, prompt search should be the FIRST intervention, not fine-tuning.",
            (1, "Fine-tuning risks CLIP alignment loss; prompt search is free and reversible."),
        ],
    )

    # =============================================================
    # Part 4: Research directions
    # =============================================================
    add_section_slide(prs, 4, "Open Questions · Future Work")

    add_two_col_slide(
        prs,
        "What We Could Not Answer in This Sprint",
        "Data-dependent",
        [
            "How well does our default pipeline work on actual protective-bag imagery?",
            (1, "The captures we had were blank JPEGs (all pixels = 255)."),
            "Domain-specific labels: we have none. 50 hand-counted images would be enough for a proper MAE curve.",
            "Is the remaining apple undercount (GT ~47, pred ~30) a SAM-scale problem or a CLIP-semantics problem?",
            (1, "Needs multi-scale anchor ensemble ablation."),
        ],
        "Theory-dependent",
        [
            "Can we preserve CLIP alignment during fine-tuning with a single auxiliary loss?",
            (1, "PseCo's `cls_loss2` is one candidate — an offline CLIP-region consistency term."),
            (1, "Could a simpler text-prompt-discrimination loss work?"),
            "Scale-adaptive proposal generation — currently manual anchor tuning per domain.",
            "Is there a principled metric for \"CLIP alignment retained after fine-tuning\"?",
        ],
    )

    add_content_slide(
        prs,
        "Suggested Research Directions",
        [
            "Direction A — Alignment-preserving fine-tuning.",
            (1, "Port PseCo's `cls_loss2` (CLIP-consistency on separate batches of CLIP-detected regions)."),
            (1, "Evaluate against our Plan 3 trainer, which lost alignment."),
            (1, "Target: fine-tune on orchard data while keeping prompt-discrimination > threshold."),
            "Direction B — Scale-adaptive counting.",
            (1, "Multi-anchor ensemble: run SAM with anchors ∈ {4, 8, 16, 24, 32}, aggregate scores, NMS jointly."),
            (1, "Or learn a tiny anchor-size predictor from image statistics."),
            "Direction C — Baseline-aware counting metrics.",
            (1, "Report MAE minus 'predict mean' and MAE minus 'predict zero' jointly."),
            (1, "Helps distinguish real learning from distribution memorization on skewed count datasets."),
            "Direction D — Prompt optimization as a first-class step.",
            (1, "Systematic prompt search before any fine-tuning is considered."),
            (1, "A learned prompt-generator conditioned on image embedding could hybridize this."),
        ],
    )

    add_content_slide(
        prs,
        "Engineering Takeaways (Non-Research but Worth Keeping)",
        [
            "Invalidation-aware caches save hours: SAM features cached with `(ckpt_hash, image_size, dtype)` hash.",
            "Every ML pipeline should have a 'predict-0' reference run; otherwise you may celebrate dataset-mean imitation.",
            "Save a config snapshot INSIDE every checkpoint. We had several \"this ckpt no longer loads\" moments.",
            "Integration tests marked `slow` with `skipif(weights_missing)` are better than no tests — but they need to actually run in CI at least weekly.",
            "When adopting an upstream repo as a submodule, write a 20-minute 'does `demo.ipynb` actually import cleanly' check before trusting the API surface.",
            (1, "Our `PseCoStage.prepare()` had an import of a non-existent module for three plans."),
        ],
    )

    add_content_slide(
        prs,
        "References",
        [
            "PseCo — Huang et al., \"Point, Segment, and Count: A Generalized Framework for Object Counting\", ICLR 2024. github.com/Hzzone/PseCo",
            "SAM — Kirillov et al., \"Segment Anything\", ICCV 2023. segment-anything.com",
            "CLIP — Radford et al., \"Learning Transferable Visual Models From Natural Language Supervision\", ICML 2021.",
            "OpenCLIP — Ilharco et al. github.com/mlfoundations/open_clip",
            "FSC-147 — Ranjan et al., \"Learning To Count Everything\", CVPR 2021. github.com/cvlab-stonybrook/LearningToCountEverything",
            "This project — github.com/YBNML/fruit-counting",
            (1, "Specs under docs/superpowers/specs/"),
            (1, "Per-plan implementation logs under docs/superpowers/plans/"),
        ],
    )

    prs.save(str(OUT))
    print(f"[done] wrote {OUT}")


if __name__ == "__main__":
    main()
