#!/usr/bin/env python3
from __future__ import annotations

import html
import urllib.parse
from pathlib import Path
from typing import Optional

import tornado.ioloop
import tornado.web

from ilm.etymology import db as dbm
from ilm.etymology.hanziyuan import (
    build_char_info,
    fetch_url,
    guess_source_site,
    parse_page,
    save_glyph_assets,
)


DEFAULT_DB = Path("data/historic/etymology.sqlite3")
DEFAULT_OUT = Path("data/historic/glyphs")
DEFAULT_CACHE = Path("data/historic/cache")


PINYIN_DEMO = {
    "zhong": "中",
    "che": "车",
}


def build_url_from_site(char: str, site: str) -> str:
    if site == "chineseetymology":
        q = urllib.parse.quote(char)
        return f"https://www.chineseetymology.org/CharacterEtymology.aspx?characterInput={q}"
    if site == "hanziyuan":
        # Note: many pages are dynamically rendered; may not parse server-side
        q = urllib.parse.quote(char)
        return f"https://hanziyuan.net/#{q}"
    return ""


class IndexHandler(tornado.web.RequestHandler):
    def get(self):
        self.write(
            """
            <html><head><title>ILM Etymology Ingest</title></head>
            <body>
              <h2>ILM Etymology Ingest</h2>
              <form method="POST" action="/ingest">
                <div>
                  <label>Query (char or pinyin demo like 'zhong'):</label>
                  <input type="text" name="query" placeholder="中 or zhong"/>
                </div>
                <div>
                  <label>Or direct page URL:</label>
                  <input type="text" name="url" size="80" placeholder="https://www.chineseetymology.org/..."/>
                </div>
                <div>
                  <label>Site helper (if only char provided):</label>
                  <select name="site">
                    <option value="chineseetymology" selected>chineseetymology.org</option>
                    <option value="hanziyuan">hanziyuan.net (JS-heavy)</option>
                  </select>
                </div>
                <div>
                  <label>DB path:</label>
                  <input type="text" name="db" value="data/historic/etymology.sqlite3" size="60"/>
                </div>
                <div>
                  <label>Output dir:</label>
                  <input type="text" name="out" value="data/historic/glyphs" size="60"/>
                </div>
                <div>
                  <label>Cache dir:</label>
                  <input type="text" name="cache" value="data/historic/cache" size="60"/>
                </div>
                <div>
                  <label>Delay (seconds):</label>
                  <input type="number" step="0.1" name="delay" value="0.5"/>
                </div>
                <div><button type="submit">Ingest</button></div>
              </form>
            </body></nhtml>
            """
        )


class IngestHandler(tornado.web.RequestHandler):
    def post(self):
        url = (self.get_body_argument("url", default="") or "").strip()
        query = (self.get_body_argument("query", default="") or "").strip()
        site = (self.get_body_argument("site", default="chineseetymology") or "").strip()
        db_path = Path(self.get_body_argument("db", default=str(DEFAULT_DB)))
        out_root = Path(self.get_body_argument("out", default=str(DEFAULT_OUT)))
        cache_dir = Path(self.get_body_argument("cache", default=str(DEFAULT_CACHE)))
        delay_s = float(self.get_body_argument("delay", default="0.5"))

        char_override: Optional[str] = None
        if not url and query:
            if len(query) == 1:
                char_override = query
            else:
                char_override = PINYIN_DEMO.get(query.lower())
            if not char_override:
                self.write(f"<p>Unknown pinyin demo '{html.escape(query)}'. Try 'zhong' or enter a URL.</p>")
                return
            url = build_url_from_site(char_override, site)
        if not url:
            self.write("<p>Provide either a URL or a character/pinyin query.</p>")
            return

        try:
            html_text = fetch_url(url, cache_dir=cache_dir, delay=delay_s)
            meta, glyphs = parse_page(html_text, base_url=url)
            meta = build_char_info(char_override, meta)
            if not meta:
                raise RuntimeError("Failed to detect character; try specifying the char in the query field.")
            saved = save_glyph_assets(glyphs=glyphs, out_root=out_root, char=meta.char, base_url=url)

            # DB upsert
            conn = dbm.connect(db_path)
            try:
                dbm.ensure_schema(conn)
                char_id = dbm.upsert_char(
                    conn,
                    meta.char,
                    codepoint=meta.codepoint,
                    pinyin=meta.pinyin,
                    main_meaning=meta.main_meaning,
                    importance_freq=meta.importance_freq,
                    sources=guess_source_site(url),
                )
                for g, local_path, w, h in saved:
                    dbm.add_glyph(
                        conn,
                        char_id=char_id,
                        stage=g.stage or "unknown",
                        label=g.label,
                        source_site=guess_source_site(url),
                        url=url if not (g.src or "").startswith("data:") else None,
                        local_path=str(local_path),
                        width=w,
                        height=h,
                    )
            finally:
                conn.close()

            files = "<br/>".join(html.escape(str(p)) for _, p, _, _ in saved)
            self.write(
                f"""
                <h3>Ingest OK</h3>
                <p>Char: {html.escape(meta.char)} ({html.escape(meta.codepoint or '')})</p>
                <p>Source: {html.escape(url)}</p>
                <p>Saved {len(saved)} glyph files:</p>
                <div style='font-family:monospace'>{files}</div>
                <p><a href='/'>Back</a></p>
                """
            )
        except Exception as e:
            self.write(f"<h3>Error</h3><pre>{html.escape(str(e))}</pre><p><a href='/'>Back</a></p>")


def make_app():
    return tornado.web.Application([
        (r"/", IndexHandler),
        (r"/ingest", IngestHandler),
    ])


if __name__ == "__main__":
    app = make_app()
    app.listen(8888)
    print("Serving on http://127.0.0.1:8888 …")
    tornado.ioloop.IOLoop.current().start()

