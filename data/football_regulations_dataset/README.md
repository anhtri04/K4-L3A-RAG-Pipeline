# Professional Football Regulations Dataset Seed

Compiled: 2026-09-20

## Structure

- `data/landing/legal/ifab/` - IFAB playing-law references
- `data/landing/legal/fifa/` - FIFA legal and transfer-system references
- `data/landing/legal/uefa/` - UEFA competition and disciplinary references
- `data/landing/legal/afc/` - AFC licensing and competition references
- `data/landing/news/...` - official news/change/enforcement JSON records

## Legal files

The DOCX files are concise source-grounded reference summaries created from official federation/confederation sources. They preserve source URL, edition/effective-date metadata and retrieval topics. They are not substitutes for the complete official regulations.

## News JSON

Each news JSON contains at least `url`, `title`, `date_crawled`, and `content_markdown`, plus governance metadata useful for RAG. `content_markdown` is a concise summary rather than a full republication of the source article.

## Version-aware RAG

When answering "current rule" questions, filter by governing body, competition/jurisdiction, status and effective date before semantic similarity. Keep future-effective regulations separate from current rules.
