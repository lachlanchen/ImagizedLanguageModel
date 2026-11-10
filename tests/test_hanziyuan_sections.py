"""Stage boundaries must preserve the source's empty historical sections."""
import unittest

from ilm.etymology.hanziyuan import parse_page


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


if __name__=='__main__':unittest.main()
