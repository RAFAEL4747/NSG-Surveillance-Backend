import os, logging
from datetime import datetime
from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.lib.enums import TA_CENTER
from core.schemas import AnalysisResult, SeverityLevel

logger = logging.getLogger(__name__)

BLACK  = colors.HexColor("#1A1A1A")
AMBER  = colors.HexColor("#B45309")
RED    = colors.HexColor("#991B1B")
GREEN  = colors.HexColor("#166534")
OFFWHT = colors.HexColor("#F5F4F0")
BORDER = colors.HexColor("#D6D3C8")

SEV_BG = {
    SeverityLevel.HIGH:   colors.HexColor("#FEE2E2"),
    SeverityLevel.MEDIUM: colors.HexColor("#FEF3C7"),
    SeverityLevel.LOW:    colors.HexColor("#F0FDF4"),
}
SEV_FG = {
    SeverityLevel.HIGH:   RED,
    SeverityLevel.MEDIUM: AMBER,
    SeverityLevel.LOW:    GREEN,
}

def generate_report(result: AnalysisResult, output_dir: str) -> str:
    fname = f"NSG_Report_{result.job_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    fpath = os.path.join(output_dir, fname)
    doc   = SimpleDocTemplate(fpath, pagesize=A4,
                              topMargin=18*mm, bottomMargin=18*mm,
                              leftMargin=20*mm, rightMargin=20*mm)
    st = getSampleStyleSheet()
    h1 = ParagraphStyle("h1", parent=st["Normal"], fontSize=15, fontName="Helvetica-Bold",
                         textColor=BLACK, alignment=TA_CENTER)
    h2 = ParagraphStyle("h2", parent=st["Normal"], fontSize=11, fontName="Helvetica-Bold",
                         textColor=BLACK, spaceBefore=8, spaceAfter=4)
    body = ParagraphStyle("body", parent=st["Normal"], fontSize=8, leading=13)
    mono = ParagraphStyle("mono", parent=st["Normal"], fontSize=7, fontName="Courier", leading=11)
    sub  = ParagraphStyle("sub",  parent=st["Normal"], fontSize=8, textColor=colors.grey, alignment=TA_CENTER)
    cls  = ParagraphStyle("cls",  parent=st["Normal"], fontSize=8, fontName="Helvetica-Bold",
                           textColor=RED, alignment=TA_CENTER)

    story = []
    story.append(Paragraph("NATIONAL SECURITY GUARD", h1))
    story.append(Paragraph("Integrated Command Post · AI Surveillance Analysis System", sub))
    story.append(Spacer(1,3*mm))
    story.append(Paragraph("CLASSIFICATION: RESTRICTED", cls))
    story.append(HRFlowable(width="100%", thickness=1, color=BLACK))
    story.append(Spacer(1,4*mm))

    s = result.summary
    meta = [
        ["Report ID", result.job_id, "Generated", datetime.now().strftime("%d %b %Y %H:%M")],
        ["Video",     Path(result.video_file).name, "Audio", Path(result.audio_file).name],
        ["Duration",  s.duration_analysed, "Frames processed", str(s.frames_processed)],
        ["Threat level", s.overall_threat_level.value.upper(), "Total alerts", str(len(result.alerts))],
    ]
    mt = Table(meta, colWidths=[32*mm,63*mm,32*mm,48*mm])
    mt.setStyle(TableStyle([
        ("FONTSIZE",(0,0),(-1,-1),7), ("FONTNAME",(0,0),(0,-1),"Helvetica-Bold"),
        ("FONTNAME",(2,0),(2,-1),"Helvetica-Bold"),
        ("BACKGROUND",(0,0),(-1,-1),OFFWHT),
        ("GRID",(0,0),(-1,-1),0.4,BORDER), ("PADDING",(0,0),(-1,-1),3),
        ("TEXTCOLOR",(1,3),(1,3), RED if s.overall_threat_level==SeverityLevel.HIGH else AMBER),
        ("FONTNAME",(1,3),(1,3),"Helvetica-Bold"),
    ]))
    story.extend([mt, Spacer(1,5*mm)])

    # Capabilities note
    caps = result.capabilities
    missing = []
    if not caps.get("object_detection"): missing.append("object/weapon detection (install ultralytics)")
    if not caps.get("transcription"):    missing.append("speech transcription (install openai-whisper)")
    if missing:
        story.append(Paragraph(f"⚠ Limited analysis — modules unavailable: {'; '.join(missing)}", body))
        story.append(Spacer(1,3*mm))

    story.append(Paragraph("Alerts", h2))
    if result.alerts:
        rows = [["Time","Severity","Type","Category","Description","Conf."]]
        for a in sorted(result.alerts, key=lambda x: x.timestamp):
            rows.append([a.timestamp, a.severity.value.upper(), a.alert_type.value,
                         a.category.replace("_"," ").title(),
                         Paragraph(a.description[:110], mono), f"{a.confidence:.0%}"])
        at = Table(rows, colWidths=[17*mm,16*mm,13*mm,27*mm,86*mm,12*mm], repeatRows=1)
        cmds = [("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),("FONTSIZE",(0,0),(-1,-1),7),
                ("BACKGROUND",(0,0),(-1,0),BLACK),("TEXTCOLOR",(0,0),(-1,0),colors.white),
                ("GRID",(0,0),(-1,-1),0.3,BORDER),("VALIGN",(0,0),(-1,-1),"TOP"),("PADDING",(0,0),(-1,-1),3)]
        for ri,a in enumerate(result.alerts,1):
            cmds += [("BACKGROUND",(0,ri),(0,ri),SEV_BG[a.severity]),
                     ("TEXTCOLOR",(0,ri),(0,ri),SEV_FG[a.severity]),
                     ("FONTNAME",(0,ri),(0,ri),"Helvetica-Bold"),
                     ("BACKGROUND",(1,ri),(1,ri),SEV_BG[a.severity]),
                     ("TEXTCOLOR",(1,ri),(1,ri),SEV_FG[a.severity])]
        at.setStyle(TableStyle(cmds))
        story.append(at)
    else:
        story.append(Paragraph("No alerts generated.", body))

    story.extend([Spacer(1,4*mm), Paragraph("Event Timeline", h2)])
    if result.timeline:
        tr = [["Timestamp","Source","Event"]]
        for ev in result.timeline:
            tr.append([ev.timestamp, ev.source.value.upper(), Paragraph(ev.event[:160], mono)])
        tt = Table(tr, colWidths=[18*mm,16*mm,141*mm], repeatRows=1)
        tt.setStyle(TableStyle([
            ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),("FONTSIZE",(0,0),(-1,-1),7),
            ("BACKGROUND",(0,0),(-1,0),BLACK),("TEXTCOLOR",(0,0),(-1,0),colors.white),
            ("GRID",(0,0),(-1,-1),0.3,BORDER),
            ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,OFFWHT]),
            ("VALIGN",(0,0),(-1,-1),"TOP"),("PADDING",(0,0),(-1,-1),3),
        ]))
        story.append(tt)

    story.extend([Spacer(1,4*mm), HRFlowable(width="100%",thickness=0.4,color=BORDER)])
    story.append(Paragraph(
        f"NSG AI Surveillance System · Auto-generated · {datetime.now().strftime('%d %b %Y %H:%M')} · RESTRICTED",
        ParagraphStyle("ft",parent=st["Normal"],fontSize=6,textColor=colors.grey,alignment=TA_CENTER)
    ))
    doc.build(story)
    return fpath
