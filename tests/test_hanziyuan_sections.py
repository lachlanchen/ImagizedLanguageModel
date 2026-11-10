"""Stage boundaries must preserve the source's empty historical sections."""
import unittest

from ilm.etymology.hanziyuan import parse_hanziyuan_entries, parse_page


def page(body):
    labels=['J00886','B00610','B00611','S00273','L00585']
    styles=' '.join(f'#{label}, #etymology{label} {{ background-image: url("data:image/svg+xml;base64,PHN2Zy8+") }}' for label in labels)
    return f'<div id="etymologyCharacters">{body}<style>{styles}</style></div>'


def glyph_list(*labels):
    return '<ul>'+''.join(f'<li><div id="etymology{x}">{x}</div><button>{x}</button></li>' for x in labels)+'</ul>'


class HistoricalSectionTests(unittest.TestCase):
    def test_empty_oracle_does_not_inherit_bronze_or_seal(self):
        html=page('<h3>Oracle characters 甲骨文 (0)</h3><p>Not available.</p><hr>'
            '<h3>Bronze characters 金文 (2)</h3>'+glyph_list('B00610','B00611')+
            '<h3>Seal characters 篆字 (1)</h3>'+glyph_list('S00273')+
            '<h3>Liushutong characters 六书通 (0)</h3><p>Not available.</p>')
        _,glyphs=parse_page(html,'https://hanziyuan.net/etymology')
        self.assertEqual({(g.stage,g.label) for g in glyphs},
                         {('bronze','B00610'),('bronze','B00611'),('seal','S00273')})

    def test_unrecognized_heading_also_ends_the_previous_section(self):
        html=page('<h3>Oracle characters (0)</h3><p>Not available.</p>'
                  '<h3>Further information</h3>'+glyph_list('B00610'))
        self.assertEqual(parse_page(html,'https://hanziyuan.net/etymology')[1],[])

    def test_populated_lists_keep_their_own_stage_and_unique_labels(self):
        html=page('<h3>Oracle characters (1)</h3>'+glyph_list('J00886')+
                  '<hr><h3>Bronze characters (1)</h3>'+glyph_list('B00610')+
                  '<h3>Liushutong characters (1)</h3>'+glyph_list('L00585'))
        _,glyphs=parse_page(html,'https://hanziyuan.net/etymology')
        self.assertEqual([(g.stage,g.label) for g in glyphs],
                         [('oracle','J00886'),('bronze','B00610'),('liushutong','L00585')])

    def test_ajax_entries_keep_separate_fields_and_stylesheets(self):
        first_svg = "data:image/svg+xml;base64,PHN2Zy8+"
        second_svg = "data:image/svg+xml;base64,PHN2Zz48L3N2Zz4="
        html = f"""
        <div class="etymology-nav" id="&#x53EA;">
          <div id="E08169">
            <p><b>Main pronunciation 主要发音:</b> zhǐ</p>
            <p><b>Original meaning 本义:</b> First description.</p>
            <div id="etymologyCharacters">
              <h3>Bronze characters (1)</h3>{glyph_list('B00610')}
            </div>
            <style>#B00610 {{ background-image: url('{first_svg}') }}</style>
          </div>
          <div id="E08172">
            <p><b>Main pronunciation 主要发音:</b> zhī</p>
            <p><b>Original meaning 本义:</b> Second description.</p>
            <div id="etymologyCharacters">
              <h3>Bronze characters (1)</h3>{glyph_list('B00610')}
            </div>
            <style>#B00610 {{ background-image: url('{second_svg}') }}</style>
          </div>
        </div>
        """
        entries = parse_hanziyuan_entries(html)
        self.assertEqual([entry.entry_id for entry in entries], ["E08169", "E08172"])
        self.assertEqual([entry.fields["original_meaning"] for entry in entries],
                         ["First description.", "Second description."])
        self.assertEqual([entry.glyphs[0].src for entry in entries], [first_svg, second_svg])
        meta, glyphs = parse_page(html, "https://hanziyuan.net/etymology")
        self.assertEqual((meta.char, meta.codepoint, meta.pinyin, meta.main_meaning),
                         ("只", "U+53EA", "zhǐ", "First description."))
        self.assertEqual([glyph.src for glyph in glyphs], [first_svg, second_svg])


if __name__=='__main__':unittest.main()
