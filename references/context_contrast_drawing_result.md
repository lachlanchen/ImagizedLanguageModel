# Earlier context changes freely drawn strokes

On 1,024 held-out passage prefixes the native model drew zero exact next-source glyphs with stock greedy decoding, matched full-density decoding and full-versus-last-glyph contrast. All three methods drew a nonempty image for every prefix. Each image came from the model's own eight-component writer. Reference successors entered only the subsequent image comparison. [Protocol](context_contrast_drawing_protocol.md) · [Measurements](evidence/context_contrast_drawing_20261118/result.json)

| Decoder | Distinct images | Pixels changed between the two contexts, mean of 1,024 pixels | Mean source ink F1 | Exact matches anywhere among 9,581 observed training shapes |
| --- | ---: | ---: | ---: | ---: |
| Stock greedy | 213 | 2.30 | .3490 | 0 |
| Full-density control | 308 | 7.08 | .3152 | 2 |
| Context contrast | 849 | 60.18 | .2367 | 0 |

The contrast changed 10,796 of 49,152 local choices relative to full density on its own generated prefixes. Its improvement in the source-versus-exchanged F1 margin was −0.00050 (95% article interval −0.00469 to 0.00362). The drawing criterion failed. An observed text can have several sensible continuations, and exact bitmap identity also requires stroke fidelity. Neither metric counts all linguistically appropriate outputs.

In response to the pictured question about 言, both new decoders reached 128 cells. The control repeated a small set of box-like marks. The contrast produced more varied, fragmented marks. A single informed model reader could read no explanation in either full output. These cases sit outside the ongoing blind evaluation. This result retires the tested decoding rule for this checkpoint. [Full-density answer](evidence/context_contrast_drawing_20261118/density_yan.png) · [Context-contrast answer](evidence/context_contrast_drawing_20261118/contrast_yan.png)
