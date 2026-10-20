# DarkLens — E-commerce Dark Pattern Review

DarkLens is a Streamlit research prototype for reviewing public e-commerce webpages for potentially manipulative wording.

## What this version does

This version intentionally keeps **one main workflow only**:

1. Paste a public e-commerce/product-page URL.
2. DarkLens reads the visible page text.
3. It checks the text for possible signals such as urgency, scarcity, popularity, discount pressure, extra costs, add-ons, and subscriptions.
4. It shows the **exact text examples** that triggered each signal.
5. It also shows the saved machine-learning model's suggested category.
6. A simple filter lets a non-technical reviewer focus on one area at a time.

The app does **not** include separate Home, Analyze Text, Compare, or How to Read Results pages.

## Technology

- Python
- Streamlit
- Requests
- BeautifulSoup
- Regular-expression pattern checks
- scikit-learn TF-IDF vectorizer + saved classifier
- Joblib for loading the saved model/vectorizer

## Important interpretation

A detected phrase is a **review signal**, not proof that a retailer is deceptive.

For example:
- "Only 2 left" may indicate scarcity pressure, but the stock claim may still be true.
- "73% off" may indicate discount pressure, but the discount may still be legitimate.
- "Best seller" may indicate popularity/social proof, but that does not by itself mean the wording is harmful.

The model prediction is also a suggestion based on patterns learned during training. It is not a final judgement.

## Run locally

1. Install Python 3.10 or 3.11.
2. Open a terminal in this folder.
3. Install dependencies:
   `pip install -r requirements.txt`
4. Start:
   `streamlit run app.py`

Keep these files together:
- `app.py`
- `detector.py`
- `dark_pattern_model.pkl`
- `vectorizer.pkl`

## Limitations

- Text-only detection can miss visual or interaction-based dark patterns.
- Phrase rules can flag legitimate wording.
- The model's training data and verified accuracy metrics are not included in this prototype.
- JavaScript-heavy or bot-blocking pages may not be readable.
- The public URL scanner is not a substitute for a production security-audited webpage fetching service.
- Do not enter credentials, OTPs, payment details, or private URLs.
