"""
PDF Report Generator
Produces a formatted, restricted-classification incident report.
Uses ReportLab — no internet required.
"""

import os
from datetime import datetime
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, KeepTogether,
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT

from core.schemas import AnalysisResult, SeverityLevel


# ── Colour palette ─────────────────────────────────────────────────────────────
NSG_DARK   = colors.HexColor("#0F1923")
NSG_GREEN  = colors.HexColor("#1D9E75")
NSG_RED    = colors.HexColor("#A32D2D")
NSG_AMBER  = colors.HexColor("#854F0B")
NSG_LIGHT  = colors.HexColor("#F4F4F4")
NSG_BORDER = colors.HexColor("#CCCCCC")

SEVERITY_COLOURS = {
    SeverityLevel.HIGH:   (colors.HexColor("#FCEBEB"), colors.HexColor("#A32D2D")),
    SeverityLevel.MEDIUM: (colors.HexColor("#FAEEDA"), colors.HexColor("#854F0B")),
    SeverityLevel.LOW:    (colors.HexColor("#E1F5EE"), colors.HexColor("#0F6E56")),
}


def generate_report(result: AnalysisResult, output_dir: str) -> str:
    """
    Generate a PDF report from AnalysisResult.
    Returns the absolute path to the saved PDF.
    """
    filename = f"NSG_Report_{result.job_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    filepath = os.path.join(output_dir, filename)

    doc = SimpleDocTemplate(
        filepath,
        pagesize=A4,
        topMargin=20*mm, bottomMargin=20*mm,
        leftMargin=20*mm, rightMargin=20*mm,
    )

    styles = getSampleStyleSheet()
    story  = []

    # ── Helper styles ──────────────────────────────────────────────────────────
    title_style = ParagraphStyle("Title", parent=styles["Normal"],
                                 fontSize=18, fontName="Helvetica-Bold",
                                 textColor=NSG_DARK, alignment=TA_CENTER)
    sub_style   = ParagraphStyle("Sub",   parent=styles["Normal"],
                                 fontSize=10, textColor=colors.grey, alignment=TA_CENTER)
    class_style = ParagraphStyle("Class", parent=styles["Normal"],
                                 fontSize=9, fontName="Helvetica-Bold",
                                 textColor=NSG_RED, alignment=TA_CENTER)
    h2_style    = ParagraphStyle("H2",    parent=styles["Normal"],
                                 fontSize=12, fontName="Helvetica-Bold",
                                 textColor=NSG_DARK, spaceBefore=10, spaceAfter=4)
    body_style  = ParagraphStyle("Body",  parent=styles["Normal"],
                                 fontSize=9, leading=14)
    mono_style  = ParagraphStyle("Mono",  parent=styles["Normal"],
                                 fontSize=8, fontName="Courier", leading=12)

    # ── Header ─────────────────────────────────────────────────────────────────
    story.append(Paragraph("⬛ NATIONAL SECURITY GUARD", title_style))
    story.append(Paragraph("Integrated Command Post · AI Surveillance Analysis", sub_style))
    story.append(Spacer(1, 4*mm))
    story.append(Paragraph("⚠ CLASSIFICATION: RESTRICTED ⚠", class_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=NSG_GREEN))
    story.append(Spacer(1, 4*mm))

    # ── Meta table ─────────────────────────────────────────────────────────────
    meta_data = [
        ["Report ID",    result.job_id,
         "Generated",    datetime.now().strftime("%d %b %Y  %H:%M:%S")],
        ["Video File",   Path(result.video_file).name,
         "Audio File",   Path(result.audio_file).name],
        ["Duration",     result.summary.duration_analysed,
         "Frames",       str(result.summary.frames_processed)],
        ["Threat Level", result.summary.overall_threat_level.value.upper(),
         "Persons Detected", str(result.summary.persons_detected)],
    ]
    meta_table = Table(meta_data, colWidths=[35*mm, 60*mm, 35*mm, 45*mm])
    meta_table.setStyle(TableStyle([
        ("FONTNAME",   (0,0), (-1,-1), "Helvetica"),
        ("FONTSIZE",   (0,0), (-1,-1), 8),
        ("FONTNAME",   (0,0), (0,-1), "Helvetica-Bold"),
        ("FONTNAME",   (2,0), (2,-1), "Helvetica-Bold"),
        ("BACKGROUND", (0,0), (-1,-1), NSG_LIGHT),
        ("GRID",       (0,0), (-1,-1), 0.5, NSG_BORDER),
        ("PADDING",    (0,0), (-1,-1), 4),
        ("TEXTCOLOR",  (1,3), (1,3),
         NSG_RED if result.summary.overall_threat_level == SeverityLevel.HIGH else NSG_AMBER),
        ("FONTNAME",   (1,3), (1,3), "Helvetica-Bold"),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 5*mm))

    # ── Summary metrics ────────────────────────────────────────────────────────
    story.append(Paragraph("Executive Summary", h2_style))
    s = result.summary
    summary_text = (
        f"Analysis of <b>{Path(result.video_file).name}</b> and "
        f"<b>{Path(result.audio_file).name}</b> has been completed. "
        f"The system processed <b>{s.frames_processed}</b> frames over a duration of "
        f"<b>{s.duration_analysed}</b>, detecting <b>{s.persons_detected}</b> person(s), "
        f"<b>{s.faces_recognised}</b> face(s), and "
        f"<b>{s.watchlist_matches}</b> watchlist match(es). "
        f"A total of <b>{len(result.alerts)}</b> alert(s) were generated, with "
        f"<b>{len([a for a in result.alerts if a.severity == SeverityLevel.HIGH])}</b> classified as HIGH severity. "
        f"Overall threat assessment: <b>{s.overall_threat_level.value.upper()}</b>."
    )
    story.append(Paragraph(summary_text, body_style))
    story.append(Spacer(1, 4*mm))

    # ── Alerts table ───────────────────────────────────────────────────────────
    story.append(Paragraph("Alerts Log", h2_style))
    if result.alerts:
        alert_rows = [["Time", "Severity", "Type", "Category", "Description", "Conf."]]
        for a in sorted(result.alerts, key=lambda x: x.timestamp):
            bg, fg = SEVERITY_COLOURS[a.severity]
            alert_rows.append([
                a.timestamp,
                a.severity.value.upper(),
                a.alert_type.value,
                a.category.replace("_", " ").title(),
                Paragraph(a.description[:120], mono_style),
                f"{a.confidence:.0%}",
            ])

        alert_table = Table(
            alert_rows,
            colWidths=[18*mm, 18*mm, 14*mm, 28*mm, 70*mm, 14*mm],
            repeatRows=1,
        )
        # Build per-row background colours
        style_cmds = [
            ("FONTNAME",   (0,0), (-1,0),  "Helvetica-Bold"),
            ("FONTSIZE",   (0,0), (-1,-1), 7),
            ("BACKGROUND", (0,0), (-1,0),  NSG_DARK),
            ("TEXTCOLOR",  (0,0), (-1,0),  colors.white),
            ("GRID",       (0,0), (-1,-1), 0.3, NSG_BORDER),
            ("VALIGN",     (0,0), (-1,-1), "TOP"),
            ("PADDING",    (0,0), (-1,-1), 3),
        ]
        for row_idx, a in enumerate(result.alerts, start=1):
            bg, fg = SEVERITY_COLOURS[a.severity]
            style_cmds += [
                ("BACKGROUND", (1, row_idx), (1, row_idx), bg),
                ("TEXTCOLOR",  (1, row_idx), (1, row_idx), fg),
                ("FONTNAME",   (1, row_idx), (1, row_idx), "Helvetica-Bold"),
            ]
        alert_table.setStyle(TableStyle(style_cmds))
        story.append(alert_table)
    else:
        story.append(Paragraph("No alerts generated.", body_style))

    story.append(Spacer(1, 5*mm))

    # ── Timeline ───────────────────────────────────────────────────────────────
    story.append(Paragraph("Event Timeline", h2_style))
    if result.timeline:
        tl_rows = [["Timestamp", "Source", "Event"]]
        for ev in sorted(result.timeline, key=lambda x: x.timestamp):
            tl_rows.append([
                ev.timestamp,
                ev.source.value.upper(),
                Paragraph(ev.event[:180], mono_style),
            ])
        tl_table = Table(tl_rows, colWidths=[20*mm, 18*mm, 130*mm], repeatRows=1)
        tl_table.setStyle(TableStyle([
            ("FONTNAME",   (0,0), (-1,0),  "Helvetica-Bold"),
            ("FONTSIZE",   (0,0), (-1,-1), 7),
            ("BACKGROUND", (0,0), (-1,0),  NSG_DARK),
            ("TEXTCOLOR",  (0,0), (-1,0),  colors.white),
            ("GRID",       (0,0), (-1,-1), 0.3, NSG_BORDER),
            ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, NSG_LIGHT]),
            ("VALIGN",     (0,0), (-1,-1), "TOP"),
            ("PADDING",    (0,0), (-1,-1), 3),
        ]))
        story.append(tl_table)
    story.append(Spacer(1, 5*mm))

    # ── Audio anomalies ────────────────────────────────────────────────────────
    story.append(Paragraph("Audio Anomalies", h2_style))
    if result.audio_anomalies:
        ao_rows = [["Time", "Type", "Confidence", "Detail"]]
        for ao in result.audio_anomalies:
            ao_rows.append([
                ao.timestamp,
                ao.anomaly_type.replace("_", " ").title(),
                f"{ao.confidence:.0%}",
                Paragraph(ao.detail[:140], mono_style),
            ])
        ao_table = Table(ao_rows, colWidths=[20*mm, 30*mm, 20*mm, 98*mm], repeatRows=1)
        ao_table.setStyle(TableStyle([
            ("FONTNAME",   (0,0), (-1,0),  "Helvetica-Bold"),
            ("FONTSIZE",   (0,0), (-1,-1), 7),
            ("BACKGROUND", (0,0), (-1,0),  NSG_DARK),
            ("TEXTCOLOR",  (0,0), (-1,0),  colors.white),
            ("GRID",       (0,0), (-1,-1), 0.3, NSG_BORDER),
            ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, NSG_LIGHT]),
            ("PADDING",    (0,0), (-1,-1), 3),
        ]))
        story.append(ao_table)
    else:
        story.append(Paragraph("No audio anomalies detected.", body_style))

    story.append(Spacer(1, 5*mm))

    # ── Recommendations ────────────────────────────────────────────────────────
    story.append(Paragraph("Recommendations", h2_style))
    recs = _build_recommendations(result)
    for i, rec in enumerate(recs, 1):
        story.append(Paragraph(f"{i}. {rec}", body_style))
    story.append(Spacer(1, 4*mm))

    # ── Footer ─────────────────────────────────────────────────────────────────
    story.append(HRFlowable(width="100%", thickness=0.5, color=NSG_BORDER))
    footer_style = ParagraphStyle("Footer", parent=styles["Normal"],
                                  fontSize=7, textColor=colors.grey, alignment=TA_CENTER)
    story.append(Paragraph(
        f"NSG AI Surveillance System · Auto-generated report · {datetime.now().strftime('%d %b %Y %H:%M')} · "
        "Handle as per NSG information security policy · RESTRICTED",
        footer_style,
    ))

    doc.build(story)
    logger.info(f"PDF report saved: {filepath}")
    return filepath


def _build_recommendations(result: AnalysisResult) -> list[str]:
    recs = []
    high_alerts = [a for a in result.alerts if a.severity == SeverityLevel.HIGH]
    if any(a.category == "weapon_detected" for a in high_alerts):
        recs.append("Deploy QRT immediately to the flagged zone — weapon detected.")
    if any(a.category == "watchlist_match" for a in high_alerts):
        recs.append("Intercept and verify identity of watchlist-matched individual(s).")
    if any(a.category == "gunshot_detected" for a in result.alerts):
        recs.append("Corroborate acoustic gunshot event with visual footage at indicated timestamp.")
    if any(a.category == "crowd_formation" for a in result.alerts):
        recs.append("Monitor crowd formation zones — consider dispersal protocol.")
    if any(a.category == "loitering" for a in result.alerts):
        recs.append("Verify identity of loitering individual(s) in restricted zones.")
    if any(a.category == "unattended_object" for a in result.alerts):
        recs.append("Initiate EOD inspection of unattended object(s) per SOP.")
    if any(a.category == "keyword_alert" for a in result.alerts):
        recs.append("Review audio transcript — threat-level language detected.")
    if not recs:
        recs.append("No immediate action required — continue routine monitoring.")
    recs.append("Archive analysis artefacts (snapshots, transcript) per NSG retention policy.")
    return recs


import logging
logger = logging.getLogger(__name__)
