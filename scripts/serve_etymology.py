#!/usr/bin/env python3
from __future__ import annotations

import html
import urllib.parse
from pathlib import Path
from typing import Optional

import tornado.ioloop
import tornado.web

import logging
from ilm.etymology import db as dbm
from ilm.etymology.hanziyuan import (
    build_char_info,
    fetch_url,
    fetch_hanziyuan_ajax,
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
            <html>
            <head>
              <meta charset="utf-8" />
              <title>ILM Etymology Ingest</title>
              <style>
                body { font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; margin: 2rem; color: #222; }
                .card { max-width: 900px; margin: 0 auto; padding: 1.25rem 1.5rem; border: 1px solid #e5e7eb; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.06); }
                h2 { margin-top: 0; }
                label { display: block; font-weight: 600; margin: .5rem 0 .25rem; }
                input[type=text], input[type=number], select { width: 100%; padding: .5rem .6rem; border: 1px solid #d1d5db; border-radius: 8px; }
                .row { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; }
                .actions { margin-top: 1rem; }
                button { background: #111827; color: white; border: 0; padding: .6rem 1rem; border-radius: 8px; cursor: pointer; }
                .note { color: #6b7280; font-size: .9rem; }
              </style>
            </head>
            <body>
              <div class="card">
                <h2>ILM Etymology Ingest</h2>
                <form method="POST" action="/ingest">
                  <label>Query (char or demo pinyin like 'zhong')</label>
                  <input type="text" name="query" placeholder="中 or zhong"/>
                  <label>Or direct page URL</label>
                  <input type="text" name="url" placeholder="https://www.chineseetymology.org/..."/>
                  <div class="row">
                    <div>
                      <label>Site helper</label>
                      <select name="site">
                        <option value="hanziyuan" selected>hanziyuan.net (default)</option>
                        <option value="chineseetymology">chineseetymology.org</option>
                      </select>
                    </div>
                    <div>
                      <label>Delay (seconds)</label>
                      <input type="number" step="0.1" name="delay" value="0.5"/>
                    </div>
                  </div>
                  <div class="row">
                    <div>
                      <label>DB path</label>
                      <input type="text" name="db" value="data/historic/etymology.sqlite3"/>
                    </div>
                    <div>
                      <label>Output dir</label>
                      <input type="text" name="out" value="data/historic/glyphs"/>
                    </div>
                  </div>
                  <label>Cache dir</label>
                  <input type="text" name="cache" value="data/historic/cache"/>
                  <div class="actions"><button type="submit">Ingest</button></div>
                  <p class="note">Tip: Try pinyin "zhong" (maps to 中) or paste a direct URL.</p>
                </form>
              </div>
            </body>
            </html>
            """
        )


class IngestHandler(tornado.web.RequestHandler):
    def post(self):
        url = (self.get_body_argument("url", default="") or "").strip()
        query = (self.get_body_argument("query", default="") or "").strip()
        site = (self.get_body_argument("site", default="hanziyuan") or "").strip()
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

        logs = []
        def log(msg: str):
            logs.append(msg)
            try:
                print(msg)
            except Exception:
                pass

        try:
            if site == "hanziyuan" and char_override and (not url or url.startswith("https://hanziyuan.net/#")):
                log(f"AJAX POST to hanziyuan/etymology for char='{char_override}'")
                html_text, base_url = fetch_hanziyuan_ajax(char=char_override, cache_dir=cache_dir, delay=delay_s)
            else:
                base_url = url
                log(f"Fetch: {url}")
                html_text = fetch_url(url, cache_dir=cache_dir, delay=delay_s)
            dbg = []
            meta, glyphs = parse_page(html_text, base_url=base_url, filter_related=True, debug=dbg)
            for entry in dbg:
                log(entry)
            meta = build_char_info(char_override, meta)
            if not meta:
                raise RuntimeError("Failed to detect character; try specifying the char in the query field.")
            log(f"Detected char: {meta.char} {meta.codepoint or ''}")
            log(f"Glyph candidates: {len(glyphs)}")
            saved = save_glyph_assets(glyphs=glyphs, out_root=out_root, char=meta.char, base_url=base_url)
            log(f"Saved files: {len(saved)} → {out_root}")

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
                    sources=guess_source_site(base_url if site == "hanziyuan" and char_override else url),
                )
                for g, local_path, w, h in saved:
                    dbm.add_glyph(
                        conn,
                        char_id=char_id,
                        stage=g.stage or "unknown",
                        label=g.label,
                        source_site=guess_source_site(base_url if site == "hanziyuan" and char_override else url),
                        url=(base_url if (site == "hanziyuan" and char_override) else url) if not (g.src or "").startswith("data:") else None,
                        local_path=str(local_path),
                        width=w,
                        height=h,
                    )
            finally:
                conn.close()

            # Group by stage for display
            by_stage = {}
            for g, p, w, h in saved:
                by_stage.setdefault(g.stage or "unknown", []).append((g, p, w, h))

            # Build gallery HTML
            stages_html = []
            for stage, items in sorted(by_stage.items()):
                rows = []
                for g, p, w, h in items:
                    rel = str(p.relative_to(out_root)) if str(p).startswith(str(out_root)) else str(p)
                    img = f"<img src='/glyph?p={urllib.parse.quote(rel)}' alt='{html.escape(g.label or '')}' style='max-width:128px;max-height:128px;display:block;margin:auto;'/>"
                    cap = html.escape((g.label or "").strip() or p.name)
                    rows.append(f"<div class='tile'>{img}<div class='cap'>{cap}</div></div>")
                stages_html.append(
                    f"<section><h4>{html.escape(stage.title())}</h4><div class='grid'>{''.join(rows)}</div></section>"
                )

            log_html = html.escape("\n".join(logs))
            self.write(
                f"""
                <html>
                <head>
                  <meta charset='utf-8'/>
                  <title>Ingest OK</title>
                  <style>
                    body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; margin: 2rem; color: #222; }}
                    .wrap {{ max-width: 1100px; margin: 0 auto; }}
                    .hdr {{ display: flex; justify-content: space-between; align-items: baseline; }}
                    .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(140px, 1fr)); gap: 1rem; }}
                    .tile {{ border: 1px solid #e5e7eb; border-radius: 10px; padding: .5rem; text-align: center; background: #f9fafb; }}
                    .cap {{ font-size: .85rem; color: #374151; margin-top: .4rem; word-break: break-all; }}
                    pre.logs {{ background: #0b1020; color: #cbd5e1; padding: .75rem 1rem; border-radius: 10px; overflow: auto; }}
                    a.btn {{ display: inline-block; padding: .5rem .8rem; background: #111827; color: #fff; border-radius: 8px; text-decoration: none; }}
                  </style>
                </head>
                <body>
                  <div class='wrap'>
                    <div class='hdr'>
                      <h2>Ingest OK — {html.escape(meta.char)} {html.escape(meta.codepoint or '')}</h2>
                      <a class='btn' href='/'>New Ingest</a>
                    </div>
                    <p>Source: <a href='{html.escape(url)}' target='_blank'>{html.escape(url)}</a></p>
                    {''.join(stages_html)}
                    <h3>Logs</h3>
                    <pre class='logs'>{log_html}</pre>
                  </div>
                </body>
                </html>
                """
            )
        except Exception as e:
            self.write(f"<h3>Error</h3><pre>{html.escape(str(e))}</pre><p><a href='/'>Back</a></p>")


class GlyphHandler(tornado.web.RequestHandler):
    def initialize(self, out_root: Path):
        self.out_root = out_root

    def get(self):
        p = self.get_query_argument("p", default="")
        if not p:
            self.set_status(404); return
        # Sanitize and enforce within out_root
        candidate = (self.out_root / p).resolve()
        try:
            out_root_res = self.out_root.resolve()
        except Exception:
            out_root_res = self.out_root
        if not str(candidate).startswith(str(out_root_res)):
            self.set_status(403); return
        if not candidate.exists() or not candidate.is_file():
            self.set_status(404); return
        ext = candidate.suffix.lower()
        ctype = {
            ".svg": "image/svg+xml",
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".gif": "image/gif",
        }.get(ext, "application/octet-stream")
        with open(candidate, "rb") as f:
            self.set_header("Content-Type", ctype)
            self.write(f.read())


def make_app():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    return tornado.web.Application([
        (r"/", IndexHandler),
        (r"/ingest", IngestHandler),
        (r"/glyph", GlyphHandler, {"out_root": DEFAULT_OUT}),
    ], debug=True)


if __name__ == "__main__":
    app = make_app()
    app.listen(8888)
    print("Serving on http://127.0.0.1:8888 …")
    tornado.ioloop.IOLoop.current().start()
