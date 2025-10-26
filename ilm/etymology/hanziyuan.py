from __future__ import annotations

import base64
import os
import re
import time
import urllib.parse
from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import requests
from bs4 import BeautifulSoup, Tag


# Canonical stage IDs and phrase patterns that map to them
STAGE_PATTERNS: List[Tuple[str, Iterable[str]]] = [
    ("oracle", ("oracle", "甲骨文", "甲骨")),
    ("bronze", ("bronze", "金文")),
    ("seal", ("seal", "篆書", "篆书", "小篆", "篆")),
    ("clerical", ("隸書", "隶书", "隶", "隸")),
    ("regular", ("楷书", "楷")),
    ("running", ("行书", "行")),
    ("cursive", ("草书", "草")),
    ("liushutong", ("六书通", "liushutong")),
]


def canonical_stage_from_text(text: str) -> Optional[str]:
    t = (text or "").lower()
    for stage, patterns in STAGE_PATTERNS:
        for p in patterns:
            if p.lower() in t:
                return stage
    return None


def guess_source_site(url: str) -> str:
    try:
        host = urllib.parse.urlparse(url).hostname or ""
        if "hanziyuan" in host:
            return "hanziyuan"
        if "chineseetymology" in host:
            return "chineseetymology"
        return host or "unknown"
    except Exception:
        return "unknown"


@dataclass
class Glyph:
    stage: str
    label: Optional[str]
    src: str  # data URI or absolute/relative URL
    width: Optional[int] = None
    height: Optional[int] = None


@dataclass
class CharInfo:
    char: str
    codepoint: Optional[str] = None
    pinyin: Optional[str] = None
    main_meaning: Optional[str] = None
    importance_freq: Optional[int] = None


def _sanitize_filename(name: str, *, fallback: str = "glyph") -> str:
    name = (name or "").strip()
    if not name:
        return fallback
    # Keep word chars, dash, underscore, dot; collapse others
    name = re.sub(r"[^\w\-_.]+", "_", name)
    name = name.strip("._-") or fallback
    return name[:128]


def _decode_data_uri(data_uri: str) -> Tuple[bytes, str]:
    """Return (bytes, extension) given a data URI.
    Supports image/svg+xml, image/png, image/jpeg; utf8 or base64.
    """
    if not data_uri.startswith("data:"):
        raise ValueError("Not a data URI")
    header, _, payload = data_uri.partition(",")
    # header: data:image/svg+xml;base64
    mime = header.split(";")[0][5:] if ";" in header else header[5:]
    is_b64 = ";base64" in header
    if is_b64:
        data = base64.b64decode(payload)
    else:
        data = urllib.parse.unquote_to_bytes(payload)
    ext = {
        "image/svg+xml": ".svg",
        "image/png": ".png",
        "image/jpeg": ".jpg",
        "image/jpg": ".jpg",
        "image/gif": ".gif",
    }.get(mime, ".bin")
    return data, ext


def _extract_data_url_from_style(style: str) -> Optional[str]:
    if not style:
        return None
    m = re.search(r"background(?:-image)?:\s*url\(([^)]+)\)", style, re.I)
    if not m:
        return None
    url = m.group(1).strip('"\'')
    return url


def fetch_url(url: str, *, headers: Optional[Dict[str, str]] = None, timeout: float = 20.0,
              cache_dir: Optional[Path] = None, delay: float = 0.0) -> str:
    """Fetch a URL with optional file cache."""
    headers = headers or {"User-Agent": "ILM-Etymology/1.0 (+https://github.com/lachlanchen/ImagizedLanguageModel)"}
    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)
        key = _sanitize_filename(url, fallback=str(abs(hash(url))))
        cache_path = cache_dir / f"{key}.html"
        if cache_path.exists():
            return cache_path.read_text("utf-8", errors="ignore")
    if delay:
        time.sleep(delay)
    logger.info("fetch: %s", url)
    r = requests.get(url, headers=headers, timeout=timeout)
    r.raise_for_status()
    html = r.text
    if cache_dir:
        cache_path.write_text(html, "utf-8")  # type: ignore[name-defined]
    return html


def _codepoint_decimal(ch: str) -> str:
    return str(ord(ch))


def fetch_hanziyuan_ajax(
    *,
    char: str,
    session: Optional[requests.Session] = None,
    timeout: float = 20.0,
    cache_dir: Optional[Path] = None,
    delay: float = 0.0,
) -> Tuple[str, str]:
    """Fetch HTML by POSTing to hanziyuan `/etymology` with `chinese` codepoint.

    Returns (html, base_url) used for joining relative resources.
    """
    sess = session or requests.Session()
    headers = {
        "User-Agent": "ILM-Etymology/1.0 (+https://github.com/lachlanchen/ImagizedLanguageModel)",
        "Accept": "text/html, */*; q=0.01",
        "Accept-Language": "en-US,en;q=0.9",
        "Origin": "https://hanziyuan.net",
        "Referer": "https://hanziyuan.net/",
    }
    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)
        key = _sanitize_filename(f"hanziyuan_ajax_{char}")
        cache_path = cache_dir / f"{key}.html"
        if cache_path.exists():
            return cache_path.read_text("utf-8", errors="ignore"), "https://hanziyuan.net/etymology"

    if delay:
        time.sleep(delay)
    # Prime cookies/session (ASP.NET often sets tokens on first GET)
    try:
        logger.info("hanziyuan: prime session cookies")
        sess.get("https://hanziyuan.net/", headers=headers, timeout=timeout)
    except Exception as e:
        logger.warning("hanziyuan: priming failed: %s", e)

    # Gather Bronze token from cookies (set by landing page)
    bronze_cookie = None
    try:
        bronze_cookie = sess.cookies.get("Bronze")
    except Exception:
        bronze_cookie = None
    data = {
        # The site expects the literal character in form data
        "chinese": char,
        # It also echoes the Bronze cookie value in the form body
        **({"Bronze": bronze_cookie} if bronze_cookie else {}),
    }
    url = "https://hanziyuan.net/etymology"
    post_headers = dict(headers)
    # Custom headers used by site scripts
    post_headers["Chinese"] = _codepoint_decimal(char)
    if bronze_cookie:
        post_headers["Seal"] = bronze_cookie
    logger.info("POST %s chinese=%s has_bronze_cookie=%s", url, _codepoint_decimal(char), bool(bronze_cookie))
    r = sess.post(url, headers=post_headers, data=data, timeout=timeout)
    r.raise_for_status()
    html = r.text
    if cache_dir:
        cache_path.write_text(html, "utf-8")  # type: ignore[name-defined]
    return html, url


def _nearest_stage_for(el: Tag) -> Optional[str]:
    # Walk up ancestors to find a heading or container with stage text
    node: Optional[Tag] = el
    for _ in range(0, 6):
        if not node:
            break
        # Look for heading siblings before this node
        prev = node.previous_sibling
        while prev:
            if isinstance(prev, Tag) and prev.name and prev.name.lower() in {"h1", "h2", "h3", "h4", "h5", "h6"}:
                st = canonical_stage_from_text(prev.get_text(" ", strip=True))
                if st:
                    return st
            prev = prev.previous_sibling
        # Check current node text/class
        txt = node.get_text(" ", strip=True) if isinstance(node, Tag) else ""
        st2 = canonical_stage_from_text(txt)
        if st2:
            return st2
        # Move up
        node = node.parent if isinstance(node, Tag) else None
    return None


def _extract_char_info(soup: BeautifulSoup) -> Optional[CharInfo]:
    # Try <title>
    title = (soup.title.string if soup.title and soup.title.string else "").strip()
    m = re.search(r"([\u4e00-\u9fff])", title)
    ch = m.group(1) if m else None

    # Fallback: first large header containing a CJK char
    if not ch:
        for tag in soup.find_all(["h1", "h2"]):
            t = tag.get_text(strip=True)
            m2 = re.search(r"([\u4e00-\u9fff])", t)
            if m2:
                ch = m2.group(1)
                break

    if not ch:
        return None

    # Try pinyin and meaning heuristics
    pinyin = None
    meaning = None
    # Common cases: elements labeled 拼音 or pinyin
    for el in soup.find_all(text=re.compile(r"拼音|pinyin", re.I)):
        try:
            parent = el.parent if isinstance(el, Tag) else None
            if parent:
                # Next text or sibling holds value
                txt = parent.get_text(" ", strip=True)
                # Remove label
                txt = re.sub(r"^\s*(拼音|pinyin)[:：]?\s*", "", txt, flags=re.I)
                if txt and len(txt) < 64:
                    pinyin = txt
                    break
        except Exception:
            pass
    # Meaning: look for 释义/意思/meaning labels
    for el in soup.find_all(text=re.compile(r"释义|意思|meaning", re.I)):
        try:
            parent = el.parent if isinstance(el, Tag) else None
            if parent:
                txt = parent.get_text(" ", strip=True)
                txt = re.sub(r"^\s*(释义|意思|meaning)[:：]?\s*", "", txt, flags=re.I)
                if 0 < len(txt) < 256:
                    meaning = txt
                    break
        except Exception:
            pass

    codepoint = f"U+{ord(ch):04X}"
    return CharInfo(char=ch, codepoint=codepoint, pinyin=pinyin, main_meaning=meaning)


def parse_page(html: str, base_url: Optional[str] = None, *, filter_related: bool = True) -> Tuple[Optional[CharInfo], List[Glyph]]:
    """Parse an etymology page and extract char info and glyphs.
    Designed to handle hanziyuan/chineseetymology-like structures, including
    <img> tags and elements with CSS background-image: url(data:...).
    """
    soup = BeautifulSoup(html, "html.parser")
    char_info = _extract_char_info(soup)

    glyphs: List[Glyph] = []

    # 1) IMG tags
    for img in soup.find_all("img"):
        src = img.get("src")
        if not src:
            continue
        label = img.get("alt") or img.get("title")
        try:
            w = int(img.get("width") or 0) or None
            h = int(img.get("height") or 0) or None
        except Exception:
            w = h = None
        stage = _nearest_stage_for(img) or "unknown"
        if base_url and src and not src.startswith("data:"):
            src = urllib.parse.urljoin(base_url, src)
        glyphs.append(Glyph(stage=stage, label=label, src=src, width=w, height=h))

    # 2) Elements with background-image data URIs
    for el in soup.find_all(True):  # all tags
        style = el.get("style")
        if not style:
            continue
        url = _extract_data_url_from_style(style)
        if not url:
            continue
        label = el.get("data-label") or el.get("title")
        stage = _nearest_stage_for(el) or "unknown"
        glyphs.append(Glyph(stage=stage, label=label, src=url))

    # Deduplicate identical sources within same stage/label
    seen = set()
    uniq: List[Glyph] = []
    for g in glyphs:
        key = (g.stage, g.label or "", g.src)
        if key in seen:
            continue
        seen.add(key)
        uniq.append(g)

    if filter_related:
        uniq = filter_glyphs_for_related(uniq, base_url)
        logger.info("filtered glyphs: %d", len(uniq))

    return char_info, uniq


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _ext_from_url(url: str) -> str:
    path = urllib.parse.urlparse(url).path
    _, ext = os.path.splitext(path)
    return ext or ""


def save_glyph_assets(
    *,
    glyphs: List[Glyph],
    out_root: Path,
    char: str,
    base_url: Optional[str] = None,
    session: Optional[requests.Session] = None,
    delay: float = 0.0,
) -> List[Tuple[Glyph, Path, Optional[int], Optional[int]]]:
    """Save glyph sources to files under out_root/<char>/<stage>/<label>.ext
    Returns list of (glyph, local_path, width, height).
    """
    sess = session or requests.Session()
    results: List[Tuple[Glyph, Path, Optional[int], Optional[int]]] = []

    for idx, g in enumerate(glyphs, 1):
        label = g.label or f"glyph{idx:03d}"
        stage = g.stage or "unknown"
        safe_label = _sanitize_filename(label, fallback=f"glyph{idx:03d}")
        stage_dir = out_root / char / stage
        _ensure_dir(stage_dir)

        src = g.src
        if src.startswith("data:"):
            data, ext = _decode_data_uri(src)
            ext = ext or ".bin"
            out_path = stage_dir / f"{safe_label}{ext}"
            out_path.write_bytes(data)
            results.append((g, out_path, g.width, g.height))
            continue

        # URL: resolve relative if base_url provided
        url = urllib.parse.urljoin(base_url, src) if (base_url and not src.startswith("http")) else src
        if delay:
            time.sleep(delay)
        logger.info("download: %s", url)
        r = sess.get(url, timeout=20)
        r.raise_for_status()
        # Determine extension
        ext = _ext_from_url(url)
        if not ext:
            ct = r.headers.get("Content-Type", "").lower()
            if "svg" in ct:
                ext = ".svg"
            elif "png" in ct:
                ext = ".png"
            elif "jpeg" in ct or "jpg" in ct:
                ext = ".jpg"
            else:
                ext = ".bin"
        out_path = stage_dir / f"{safe_label}{ext}"
        out_path.write_bytes(r.content)
        results.append((g, out_path, g.width, g.height))

    return results


def build_char_info(char: Optional[str], meta: Optional[CharInfo]) -> Optional[CharInfo]:
    if char and (not meta or meta.char != char):
        # Construct CharInfo overriding detection
        return CharInfo(
            char=char,
            codepoint=f"U+{ord(char):04X}",
            pinyin=meta.pinyin if meta else None,
            main_meaning=meta.main_meaning if meta else None,
            importance_freq=meta.importance_freq if meta else None,
        )
    return meta


def _url_host(u: Optional[str]) -> str:
    try:
        return urllib.parse.urlparse(u or "").hostname or ""
    except Exception:
        return ""


_EXCLUDE_NAME_RE = re.compile(
    r"(logo|wechat|alipay|first[_-]?class|banner|ad|research|net|favicon)", re.I
)
_LIKELY_LABEL_RE = re.compile(r"^[A-Za-z]?[0-9]{3,6}")


def filter_glyphs_for_related(glyphs: List[Glyph], base_url: Optional[str]) -> List[Glyph]:
    """Heuristically keep only glyph-like images, dropping generic page assets.
    Rules:
      - Always allow data: URIs.
      - Otherwise, host must match base_url host.
      - Prefer items with a plausible label (e.g., J12345) or stage != unknown.
      - Exclude names containing obvious non-glyph terms (logo, wechat, alipay, banner, ad...).
    """
    base_host = _url_host(base_url)
    kept: List[Glyph] = []
    for g in glyphs:
        src = g.src or ""
        if src.startswith("data:"):
            kept.append(g)
            continue
        host = _url_host(src)
        if base_host and host and host != base_host:
            # Cross-origin image — likely unrelated
            continue
        # File name checks
        path = urllib.parse.urlparse(src).path
        fname = os.path.basename(path)
        if _EXCLUDE_NAME_RE.search(fname):
            continue
        plausible = False
        if g.label and _LIKELY_LABEL_RE.match(g.label.strip()):
            plausible = True
        if (g.stage or "").lower() not in {"", "unknown"}:
            plausible = True
        if not plausible:
            # If file looks like svg/png and small-ish filename, still allow
            if fname.lower().endswith((".svg", ".png", ".jpg", ".jpeg", ".gif")):
                plausible = True
        if plausible:
            kept.append(g)
    return kept

logger = logging.getLogger(__name__)
