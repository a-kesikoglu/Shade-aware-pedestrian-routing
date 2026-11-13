"""Dump paragraph/table/section structure of a .docx (used to inspect the journal template)."""
import sys
import docx
from docx.text.paragraph import Paragraph
from docx.table import Table

d = docx.Document(sys.argv[1])
out = open(sys.argv[2], "w", encoding="utf8")
for el in d.element.body.iterchildren():
    tag = el.tag.split("}")[1]
    if tag == "p":
        p = Paragraph(el, d)
        pf = p.paragraph_format
        runs = [(r.text[:30], r.font.name, r.font.size.pt if r.font.size else None, r.bold, r.italic,
                 r.font.superscript) for r in p.runs][:6]
        out.write(f"P style={p.style.name} align={p.alignment} sb={pf.space_before} sa={pf.space_after} "
                  f"ls={pf.line_spacing} ind={pf.first_line_indent} | {p.text[:220]}\n   runs={runs}\n")
    elif tag == "tbl":
        t = Table(el, d)
        out.write("TABLE\n")
        for r in t.rows:
            out.write("   ROW: " + " || ".join(c.text[:90].replace("\n", " / ") for c in r.cells) + "\n")
    else:
        out.write(tag + "\n")
for i, s in enumerate(d.sections):
    cols = s._sectPr.xpath("./w:cols")
    out.write(f"SECTION {i} top={s.top_margin.cm:.2f} bot={s.bottom_margin.cm:.2f} left={s.left_margin.cm:.2f} "
              f"right={s.right_margin.cm:.2f} w={s.page_width.cm:.2f} h={s.page_height.cm:.2f} "
              f"cols={dict(cols[0].attrib) if cols else None} diff_first={s.different_first_page_header_footer}\n")
    for name, hf in [("HEADER", s.header), ("FOOTER", s.footer), ("FPHEADER", s.first_page_header),
                     ("FPFOOTER", s.first_page_footer)]:
        for p in hf.paragraphs:
            if p.text.strip():
                out.write(f"  {name}: {p.text}\n")
        for t in hf.tables:
            for r in t.rows:
                out.write(f"  {name}TBL: " + " || ".join(c.text for c in r.cells) + "\n")
st = d.styles["Normal"]
out.write(f"NORMAL font={st.font.name} size={st.font.size}\n")
out.close()
