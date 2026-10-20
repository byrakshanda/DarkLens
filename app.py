from __future__ import annotations

import ipaddress
import os
import socket
from urllib.parse import urljoin, urlparse

import joblib
import requests
import streamlit as st
from bs4 import BeautifulSoup

from detector import detect_signals, classify_text, COMPILED

st.set_page_config(
    page_title="DarkLens | E-commerce Page Review",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "dark_pattern_model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "vectorizer.pkl")

# -----------------------------
# Simple, stakeholder-friendly UI
# -----------------------------
st.markdown("""
<style>
:root { color-scheme: dark; }
.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1180px;
}
.hero {
    padding: 1.7rem 1.8rem;
    border-radius: 18px;
    background: linear-gradient(120deg,#111827,#18264a);
    border: 1px solid #334155;
    margin-bottom: 1.2rem;
}
.hero h1 { margin-bottom: .35rem; }
.hero p { color: #cbd5e1; font-size: 1.02rem; margin-bottom: .2rem; }
.evidence {
    padding: .9rem 1rem;
    margin: .55rem 0;
    border: 1px solid #334155;
    border-radius: 10px;
    background: rgba(30,41,59,.55);
    line-height: 1.65;
}
.evidence mark {
    background: #ffd84d;
    color: #111827;
    padding: .08rem .22rem;
    border-radius: 4px;
    font-weight: 700;
}
.trigger-note {
    color: #aab6c8;
    font-size: .9rem;
    margin: .15rem 0 .7rem;
}
.plain-card {
    padding: 1rem 1.1rem;
    border: 1px solid #334155;
    border-radius: 12px;
    background: rgba(15,23,42,.45);
    height: 100%;
}
.muted { color: #aab6c8; font-size: .92rem; }
.result-good {
    padding: 1rem 1.1rem;
    border-radius: 12px;
    background: rgba(22,101,52,.22);
    border: 1px solid rgba(74,222,128,.35);
}
.result-review {
    padding: 1rem 1.1rem;
    border-radius: 12px;
    background: rgba(120,53,15,.20);
    border: 1px solid rgba(251,191,36,.35);
}
.model-card {
    margin: .9rem 0 1rem;
    padding: .95rem 1.1rem;
    border: 1px solid #334155;
    border-radius: 12px;
    background: rgba(30,41,59,.45);
}
.model-label {
    color: #aab6c8;
    font-size: .9rem;
    margin-bottom: .2rem;
}
.model-value {
    color: #f8fafc;
    font-size: 1.35rem;
    font-weight: 700;
    line-height: 1.3;
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def load_assets():
    model = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None
    vectorizer = joblib.load(VECTORIZER_PATH) if os.path.exists(VECTORIZER_PATH) else None
    return model, vectorizer


try:
    model, vectorizer = load_assets()
    model_error = None
except Exception as exc:
    model, vectorizer = None, None
    model_error = str(exc)


def public_http_url(url: str) -> str:
    """Validate a public HTTP(S) URL and reject local/private destinations."""
    parsed = urlparse(url.strip())
    if parsed.scheme not in ("http", "https"):
        raise ValueError("Please use a complete http:// or https:// webpage link.")
    if not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("This link is incomplete or contains unsupported login details.")
    if parsed.port not in (None, 80, 443):
        raise ValueError("For safety, links using custom ports are not supported.")

    host = parsed.hostname.rstrip(".").lower()
    if host in {"localhost", "localhost.localdomain"} or host.endswith(".local"):
        raise ValueError("Local or private-network addresses cannot be scanned.")

    try:
        addresses = {ipaddress.ip_address(host)}
    except ValueError:
        try:
            addresses = {
                ipaddress.ip_address(item[4][0])
                for item in socket.getaddrinfo(
                    host,
                    parsed.port or (443 if parsed.scheme == "https" else 80),
                    type=socket.SOCK_STREAM,
                )
            }
        except socket.gaierror:
            raise ValueError("The website address could not be found. Check the link and try again.")

    if not addresses or any(not address.is_global for address in addresses):
        raise ValueError("This link resolves to a non-public address and cannot be scanned.")

    return parsed.geturl()


def scrape_public_page(url: str) -> str:
    """Fetch a small public page, validating each redirect destination."""
    current = public_http_url(url)
    headers = {"User-Agent": "DarkLensResearchTool/1.0 (public-page text analysis)"}

    with requests.Session() as session:
        session.trust_env = False

        for _ in range(4):
            response = session.get(
                current,
                headers=headers,
                timeout=(5, 10),
                allow_redirects=False,
                stream=True,
            )

            if response.is_redirect or response.is_permanent_redirect:
                location = response.headers.get("Location")
                response.close()
                if not location:
                    raise ValueError("The website sent an incomplete redirect response.")
                current = public_http_url(urljoin(current, location))
                continue

            response.raise_for_status()
            content_type = response.headers.get("Content-Type", "").lower()

            if content_type and not any(
                kind in content_type
                for kind in ("text/html", "text/plain", "application/xhtml+xml")
            ):
                response.close()
                raise ValueError("This link does not appear to be a readable webpage. Try a public product page.")

            chunks, total = [], 0
            try:
                for chunk in response.iter_content(chunk_size=16384, decode_unicode=True):
                    if not chunk:
                        continue
                    if isinstance(chunk, bytes):
                        chunk = chunk.decode(
                            response.encoding or "utf-8",
                            errors="replace",
                        )

                    total += len(chunk.encode("utf-8", errors="replace"))
                    if total > 2_000_000:
                        raise ValueError("This page is too large to scan safely (limit: about 2 MB).")
                    chunks.append(chunk)
            finally:
                response.close()

            html = "".join(chunks)
            soup = BeautifulSoup(html, "html.parser")

            for tag in soup(["script", "style", "noscript", "svg", "nav", "footer"]):
                tag.decompose()

            return soup.get_text(separator=" ", strip=True)[:100_000]

    raise ValueError("This page redirected too many times. Try a direct public page link.")


DISPLAY_NAMES = {
    "Scarcity claims": "Scarcity",
    "Urgency pressure": "Urgency",
    "Social-proof claims": "Popularity",
    "Discount pressure": "Discounts",
    "Possible extra-cost clues": "Extra costs",
    "Possible add-on / sneaking clues": "Add-ons",
    "Subscription / renewal clues": "Subscriptions",
}

PLAIN_MEANINGS = {
    "Scarcity claims": "The page suggests that availability may be limited, which can encourage a quicker purchase.",
    "Urgency pressure": "The page encourages the shopper to act quickly or suggests that an offer may end soon.",
    "Social-proof claims": "The page uses popularity or other shoppers' behaviour to influence the purchase decision.",
    "Discount pressure": "The page strongly highlights a saving, deal, coupon, or reduced price.",
    "Possible extra-cost clues": "The wording mentions an additional fee or charge that may need a closer look.",
    "Possible add-on / sneaking clues": "The wording mentions an extra product or service that may be added to the purchase.",
    "Subscription / renewal clues": "The wording mentions a subscription, trial, recurring payment, or automatic renewal.",
}

REVIEW_ACTIONS = {
    "Scarcity claims": "Check whether the stock message is clear and supported by the actual availability.",
    "Urgency pressure": "Check whether the deadline is genuine and whether the offer changes after the stated time.",
    "Social-proof claims": "Check whether the popularity claim is clear, current, and presented without misleading pressure.",
    "Discount pressure": "Check the original price, final price, and coupon conditions before treating the saving as meaningful.",
    "Possible extra-cost clues": "Check when the fee is shown and whether the customer can see it before payment.",
    "Possible add-on / sneaking clues": "Check whether the extra item/service is optional and whether it is clearly selected by the shopper.",
    "Subscription / renewal clues": "Check the price, renewal terms, and cancellation information before purchase.",
}


def friendly_name(category: str) -> str:
    return DISPLAY_NAMES.get(category, category)


def highlight_matches(snippet: str, category: str) -> str:
    """Highlight the exact phrase(s) that triggered a category."""
    matches = []
    for regex in COMPILED.get(category, []):
        matches.extend(m.span() for m in regex.finditer(snippet))
    if not matches:
        safe = (snippet.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))
        return safe

    matches.sort()
    merged = []
    for start, end in matches:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))

    parts = []
    cursor = 0
    for start, end in merged:
        parts.append(snippet[cursor:start])
        parts.append(f"<mark>{snippet[start:end]}</mark>")
        cursor = end
    parts.append(snippet[cursor:])
    return "".join(parts).replace("&", "&amp;").replace("<mark>", "___MARK_OPEN___").replace("</mark>", "___MARK_CLOSE___").replace("<", "&lt;").replace(">", "&gt;").replace("___MARK_OPEN___", "<mark>").replace("___MARK_CLOSE___", "</mark>")



def render_results(text: str, source_url: str):
    signals = detect_signals(text)
    prediction = classify_text(text, model, vectorizer)

    st.markdown("## Results")

    if not signals:
        st.success("No matching wording patterns found on this page.")
        st.caption("This only means DarkLens did not find the patterns it currently checks for.")
        return

    total_examples = sum(len(v["evidence"]) for v in signals.values())
    c1, c2 = st.columns(2)
    c1.metric("Patterns found", len(signals))
    c2.metric("Examples", total_examples)

    # Keep the model suggestion fully readable instead of squeezing it into a metric column.
    if prediction:
        st.markdown(
            f'<div class="model-card"><div class="model-label">Model suggestion</div>'
            f'<div class="model-value">{prediction}</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div class="trigger-note">Highlighted words are the exact phrases that triggered each check.</div>', unsafe_allow_html=True)

    # A filter is useful only when there are several detected patterns.
    # With 1–2 patterns, showing the results directly keeps the interface simpler.
    selected_categories = list(signals.keys())
    if len(signals) > 2:
        options = [friendly_name(k) for k in signals.keys()]
        selected = st.selectbox("Filter by pattern", options)
        selected_categories = [k for k in signals if friendly_name(k) == selected]

    for category in selected_categories:
        details = signals[category]
        name = friendly_name(category)
        with st.expander(f"{name}  •  {len(details['evidence'])} example(s)", expanded=True):
            for snippet in details["evidence"]:
                highlighted = highlight_matches(snippet, category)
                st.markdown(f'<div class="evidence">“{highlighted}”</div>', unsafe_allow_html=True)

    if prediction:
        st.caption(f"Model suggestion: {prediction} · review the highlighted evidence before drawing a conclusion.")



# -----------------------------
# Main page — URL only
# -----------------------------
st.markdown("""
<div class="hero">
  <h1>🔎 DarkLens</h1>
  <p><b>Review an e-commerce webpage for potentially manipulative wording.</b></p>
  <p>Paste a public product-page URL and DarkLens will show the wording it thinks is worth reviewing.</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.title("DarkLens")
    st.caption("E-commerce wording review")
    st.divider()
    st.markdown("**Simple workflow**")
    st.markdown("1. Paste a public page URL")
    st.markdown("2. Scan the page")
    st.markdown("3. Review the examples")
    st.markdown("4. Decide whether the wording needs attention")
    st.divider()
    st.caption("DarkLens highlights possible signals. It does not make a final judgement.")

if model_error:
    st.warning(
        "The trained model could not be loaded. The wording checks can still work."
    )
    with st.expander("Technical details"):
        st.code(model_error)

st.subheader("Scan a public webpage")
st.write("Use a public e-commerce/product page. You do not need to paste the page text yourself.")

url = st.text_input(
    "Website URL",
    placeholder="https://www.example.com/product",
    help="Only public pages are supported.",
)

st.caption("For safety, use a public page only. Never enter passwords, private links, OTPs, or payment details.")

if st.button("🔍 Scan website", type="primary", use_container_width=True):
    if not url.strip():
        st.warning("Please enter a webpage URL first.")
        st.session_state.pop("page_text", None)
        st.session_state.pop("scanned_url", None)
    else:
        try:
            with st.spinner("Reading the visible page text…"):
                page_text = scrape_public_page(url)

            if len(page_text.strip()) < 20:
                st.warning(
                    "Very little readable text was found. This page may rely heavily on JavaScript. "
                    "Try another public product page."
                )
                st.session_state.pop("page_text", None)
                st.session_state.pop("scanned_url", None)
            else:
                st.session_state.page_text = page_text
                st.session_state.scanned_url = url.strip()
                st.success("Page scanned successfully. Review the findings below.")

        except (ValueError, requests.RequestException) as exc:
            st.error(
                str(exc)
                if isinstance(exc, ValueError)
                else "The page could not be retrieved. It may be unavailable or block automated access."
            )
        except Exception:
            st.error(
                "Something unexpected happened while reading this page. "
                "Try another public product page."
            )

if st.session_state.get("page_text"):
    st.divider()
    st.caption(f"Scanned page: {st.session_state.get('scanned_url', 'Public webpage')}")
    render_results(
        st.session_state.page_text,
        st.session_state.get("scanned_url", "Public webpage"),
    )

st.divider()
st.caption("DarkLens flags wording for review; a match is not proof of deception.")
