# TexnoBozor AI yordamchisi

"TexnoBozor" (o'ylab topilgan elektronika do'koni) uchun AI chat-yordamchi.
O'quv loyihasi: Gemini, LangChain, LangGraph, RAG (Chroma), SQLite, MCP, Streamlit.

**Holati: Phase 5 of 7** — terminal agenti va Streamlit interfeysi tayyor.
Keyingisi: 6-bosqich (`interrupt()` bilan tasdiqlash va MCP serverlar).

To'liq texnik topshiriq: [SPEC.md](SPEC.md).

## Ishga tushirish

1. Virtual muhit yarating va yoqing:
   `python -m venv venv && source venv/bin/activate` (Windows: `venv\Scripts\activate`)
2. Kutubxonalarni o'rnating: `pip install -r requirements.txt`
3. `.env.example` dan nusxa olib `.env` yarating va `GOOGLE_API_KEY`, `GEMINI_MODEL`, `EMBEDDING_MODEL` ni to'ldiring.
4. Soxta do'kon bazasini yarating: `python data/seed_data.py`
5. Hujjatlarni vektor bazaga yuklang: `python rag/ingest.py`
6. Interfeysni oching: `streamlit run app.py`
7. Yon panelda `+998901234567` (Aziz Karimov) raqami bilan kiring.

Testlar (API kalit kerak emas): `python -m pytest -v`
Terminalda suhbat: `python chat.py`
