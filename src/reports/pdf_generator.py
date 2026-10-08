"""
Module M12: PDF Report Generator

Generates clinical DSS reports in PDF format for stroke prediction results.
Includes patient info, prediction results, EEG feature values,
LIME explanations, and diagnostic recommendations.

Uses fpdf2 (pure Python, no external dependencies).
"""
from fpdf import FPDF
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional


class StrokeReportPDF(FPDF):
    """Custom PDF class with header/footer for stroke DSS reports."""

    def header(self):
        """Render page header with system name and date."""
        self.set_font('Helvetica', 'B', 15)
        self.set_text_color(102, 126, 234)
        self.cell(0, 10, 'EEG Stroke Decision Support System', align='C',
                  new_x="LMARGIN", new_y="NEXT")
        self.set_font('Helvetica', '', 9)
        self.set_text_color(128, 128, 128)
        self.cell(0, 5,
                  f'Report generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}',
                  align='C', new_x="LMARGIN", new_y="NEXT")
        self.set_text_color(0, 0, 0)
        # Gradient-like line
        self.set_draw_color(102, 126, 234)
        self.set_line_width(0.8)
        self.line(10, self.get_y() + 2, 200, self.get_y() + 2)
        self.set_line_width(0.2)
        self.set_draw_color(0, 0, 0)
        self.ln(8)

    def footer(self):
        """Render page footer with page number and disclaimer."""
        self.set_y(-22)
        self.set_font('Helvetica', 'I', 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 5, f'Page {self.page_no()}/{{nb}}', align='C',
                  new_x="LMARGIN", new_y="NEXT")
        self.cell(0, 5,
                  'DISCLAIMER: For research purposes only. '
                  'Not a certified medical device.',
                  align='C')

    def section_header(self, title: str):
        """Render a styled section header."""
        self.set_font('Helvetica', 'B', 12)
        self.set_fill_color(240, 242, 248)
        self.set_draw_color(102, 126, 234)
        self.cell(0, 8, f'  {title}', fill=True, border='L',
                  new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(0, 0, 0)
        self.ln(3)

    def table_header(self, columns: List[tuple]):
        """
        Render a table header row.

        Args:
            columns: list of (label, width) tuples
        """
        self.set_font('Helvetica', 'B', 9)
        self.set_fill_color(102, 126, 234)
        self.set_text_color(255, 255, 255)
        for label, width in columns:
            self.cell(width, 7, label, border=1, fill=True, align='C')
        self.ln()
        self.set_text_color(0, 0, 0)
        self.set_font('Helvetica', '', 9)


def _safe_text(text: str) -> str:
    """Sanitize text for Latin-1 encoding (fpdf2 built-in fonts only support Latin-1).
    Uses NFKD normalization to decompose accented characters first, then encodes
    to Latin-1, replacing any remaining non-Latin-1 characters with '?'.
    """
    import unicodedata
    normalized = unicodedata.normalize('NFKD', text)
    return normalized.encode('latin-1', 'replace').decode('latin-1')



def generate_report(
    patient_id: int,
    prediction: str,
    stroke_probability: float,
    risk_level: str,
    features: Dict[str, float],
    normalized_features: Dict[str, float],
    recommendations: Dict,
    lime_contributions: Optional[List] = None,
    true_label: Optional[str] = None,
    lime_fig_path: Optional[str] = None,
    physician_decisions: Optional[Dict] = None
) -> bytes:
    """
    Generate a PDF clinical report for a patient.

    Args:
        patient_id: Patient index (0-based)
        prediction: 'Stroke' or 'Control'
        stroke_probability: Probability of stroke (0-1)
        risk_level: 'high', 'medium', or 'low'
        features: {feature_name: raw_value}
        normalized_features: {feature_name: normalized_value}
        recommendations: Dict from RecommendationEngine.get_recommendations()
        lime_contributions: List of (feature_condition, weight) from LIME
        true_label: True class name or None
        lime_fig_path: Path to LIME explanation figure PNG

    Returns:
        PDF file contents as bytes
    """
    pdf = StrokeReportPDF()
    pdf.alias_nb_pages()
    pdf.add_page()

    # ── Patient Information ──────────────────────────────────
    pdf.set_font('Helvetica', 'B', 13)
    pdf.cell(0, 8, f'Patient #{patient_id + 1}',
             new_x="LMARGIN", new_y="NEXT")
    if true_label:
        pdf.set_font('Helvetica', '', 10)
        pdf.cell(0, 6, f'Known Clinical Status: {_safe_text(true_label)}',
                 new_x="LMARGIN", new_y="NEXT")
    pdf.ln(5)

    # ── Prediction Result ────────────────────────────────────
    pdf.section_header('PREDICTION RESULT')

    # Colored prediction box
    if prediction == 'Stroke':
        pdf.set_fill_color(255, 107, 107)
    else:
        pdf.set_fill_color(46, 213, 115)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font('Helvetica', 'B', 16)
    pdf.cell(55, 12, f'  {prediction}', fill=True)
    pdf.set_text_color(0, 0, 0)

    # Probability and risk level
    pdf.set_font('Helvetica', '', 11)
    pdf.cell(0, 12,
             f'   Probability: {stroke_probability:.1%}'
             f'  |  Risk: {risk_level.upper()}',
             new_x="LMARGIN", new_y="NEXT")
    pdf.ln(5)

    # ── EEG Feature Values ───────────────────────────────────
    pdf.section_header('EEG FEATURES')

    pdf.table_header([
        ('Feature', 60),
        ('Raw Value', 45),
        ('Normalized [0,1]', 45),
    ])

    fill = False
    for fname in features:
        if fill:
            pdf.set_fill_color(247, 248, 252)
        else:
            pdf.set_fill_color(255, 255, 255)
        pdf.cell(60, 6, _safe_text(fname), border=1, fill=True)
        pdf.cell(45, 6, f'{features[fname]:.4f}', border=1, fill=True,
                 align='C')
        norm_val = normalized_features.get(fname, 0.0)
        pdf.cell(45, 6, f'{norm_val:.4f}', border=1, fill=True, align='C')
        pdf.ln()
        fill = not fill

    pdf.ln(4)

    # ── LIME Explanation ─────────────────────────────────────
    if lime_contributions:
        pdf.section_header('AI EXPLANATION (LIME)')

        pdf.table_header([
            ('Feature Condition', 85),
            ('Impact', 35),
            ('Direction', 30),
        ])

        for feat_name, weight in lime_contributions:
            direction = 'Stroke +' if weight > 0 else 'Control +'

            pdf.set_font('Helvetica', '', 8)
            pdf.cell(85, 6, _safe_text(str(feat_name)[:50]), border=1)
            pdf.cell(35, 6, f'{abs(weight):.4f}', border=1, align='C')

            if weight > 0:
                pdf.set_text_color(220, 60, 60)
            else:
                pdf.set_text_color(46, 160, 80)
            pdf.set_font('Helvetica', 'B', 8)
            pdf.cell(30, 6, direction, border=1, align='C')
            pdf.set_text_color(0, 0, 0)
            pdf.set_font('Helvetica', '', 8)
            pdf.ln()

    # ── LIME Figure ──────────────────────────────────────────
    if lime_fig_path:
        pdf.ln(4)
        try:
            pdf.image(lime_fig_path, x=10, w=190)
        except Exception:
            pass  # Skip if image loading fails

    # ── Diagnostic Recommendations (new page) ────────────────
    pdf.add_page()
    pdf.section_header('DIAGNOSTIC RECOMMENDATIONS')

    pdf.set_font('Helvetica', 'B', 11)
    urgency = recommendations.get('urgency', 'N/A')
    action = recommendations.get('action', 'N/A')
    pdf.cell(0, 7, f'Urgency Level: {_safe_text(urgency)}',
             new_x="LMARGIN", new_y="NEXT")
    pdf.set_font('Helvetica', '', 10)
    pdf.cell(0, 7, f'Recommended Action: {_safe_text(action)}',
             new_x="LMARGIN", new_y="NEXT")
    pdf.ln(5)

    pdf.set_font('Helvetica', 'B', 10)
    n_tests = recommendations.get('n_tests', 0)
    pdf.cell(0, 7, f'Recommended Tests ({n_tests}):',
             new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)

    for test in recommendations.get('recommended_tests', []):
        # Test name with left accent bar
        pdf.set_draw_color(102, 126, 234)
        pdf.set_line_width(1.0)
        y_pos = pdf.get_y()
        pdf.line(12, y_pos, 12, y_pos + 11)
        pdf.set_line_width(0.2)
        pdf.set_draw_color(0, 0, 0)

        pdf.set_font('Helvetica', 'B', 9)
        pdf.cell(5, 6, '')  # indent
        pdf.cell(0, 6, _safe_text(test.get('full_name', test.get('name', ''))),
                 new_x="LMARGIN", new_y="NEXT")

        pdf.set_font('Helvetica', '', 8)
        pdf.set_text_color(100, 100, 100)
        pdf.cell(5, 5, '')  # indent
        pdf.cell(0, 5, _safe_text(test.get('description', '')),
                 new_x="LMARGIN", new_y="NEXT")
        pdf.set_text_color(0, 0, 0)
        pdf.ln(3)

    # ── Clinical Reasoning ───────────────────────────────────
    reasoning = recommendations.get('reasoning', [])
    if reasoning:
        pdf.ln(3)
        pdf.section_header('CLINICAL REASONING')
        pdf.set_font('Helvetica', '', 9)
        for reason in reasoning:
            pdf.cell(5, 6, '')
            pdf.cell(0, 6, f'- {_safe_text(reason)}',
                     new_x="LMARGIN", new_y="NEXT")

    # ── Physician Decision & Audit Log (Human-in-the-Loop) ───
    if physician_decisions and 'decisions' in physician_decisions:
        pdf.ln(4)
        pdf.section_header('PHYSICIAN REVIEW & AUDIT LOG (HUMAN-IN-THE-LOOP)')
        pdf.set_font('Helvetica', '', 8)
        pdf.cell(0, 5, 'Diagnostic test orders reviewed, confirmed, or overridden by attending physician:',
                 new_x="LMARGIN", new_y="NEXT")
        pdf.ln(2)

        pdf.table_header([
            ('Diagnostic Test', 55),
            ('Decision Status', 35),
            ('Action Timestamp', 40),
            ('Physician Notes / Reason', 60)
        ])

        for dec in physician_decisions['decisions']:
            name = _safe_text(dec.get('name', ''))
            status = dec.get('status', 'PENDING')
            ts = _safe_text(dec.get('timestamp') or 'Pending Review')
            notes = _safe_text(dec.get('physician_notes') or dec.get('reason') or '-')

            pdf.cell(55, 6, name, border=1)
            if status == 'CONFIRMED':
                pdf.set_text_color(46, 125, 50)  # Green
            elif status == 'CANCELLED':
                pdf.set_text_color(198, 40, 40)  # Red
            else:
                pdf.set_text_color(230, 81, 0)   # Orange
            pdf.set_font('Helvetica', 'B', 8)
            pdf.cell(35, 6, status, border=1, align='C')
            pdf.set_text_color(0, 0, 0)
            pdf.set_font('Helvetica', '', 8)
            pdf.cell(40, 6, ts, border=1, align='C')
            pdf.cell(60, 6, notes[:35], border=1)
            pdf.ln()

        pdf.ln(4)
        pdf.set_font('Helvetica', 'I', 8)
        pdf.cell(0, 5, 'Attending Physician Signature: ___________________________    Date: ______________',
                 new_x="LMARGIN", new_y="NEXT")

    # ── Model Information ────────────────────────────────────
    pdf.ln(6)
    pdf.section_header('MODEL INFORMATION')
    pdf.set_font('Helvetica', '', 9)

    model_info = [
        ('Architecture', 'Ensemble Echo State Network (E-ESN)'),
        ('Ensemble Size', '7 ESN models with soft voting'),
        ('Reservoir Size', '200 neurons per ESN'),
        ('Validation', 'Leave-One-Out Cross-Validation (LOOCV)'),
        ('Explainability', 'KernelSHAP (global) + LIME (local)'),
        ('Reference', 'Bouazizi & Ltifi (2024), Decision Support Systems 178'),
    ]

    for key, value in model_info:
        pdf.set_font('Helvetica', 'B', 9)
        pdf.cell(40, 6, f'{key}:')
        pdf.set_font('Helvetica', '', 9)
        pdf.cell(0, 6, _safe_text(value), new_x="LMARGIN", new_y="NEXT")

    # Return PDF as bytes
    return bytes(pdf.output())
