# Proofreading and editorial analysis

`/corrector` fixes errors without rewriting the author's voice. It processes
the entire extracted text in sequential, token-budgeted fragments with adjacent
context. The model proposes anchored edits, which are checked and applied to
the original; an invalid, ambiguous, overlapping or truncated response fails
the operation rather than returning a partial document. This is not a
guarantee that the model finds every error or makes no mistaken correction.
The separate change list contains up to 30 automatically generated diff
fragments, not a ranked or comprehensive explanation.

For uploaded TXT, the corrected file is UTF-8. For DOCX, body paragraphs and
tables are processed in order; headers, footers, notes and text in other
non-body objects are not analyzed. The original document structure is retained
where possible, but formatting in changed paragraphs can be lost. If replacing
a paragraph would destroy a hyperlink, drawing or field, the result is
returned as TXT instead. PDF is extracted to TXT; its layout is not retained.
File size and extraction limits are enforced by rejection, not silent
truncation. Scanned PDFs require OCR and may not contain extractable text.

`/editor` independently analyzes each part for contextual stylistic weaknesses.
Dictionary matches are candidates only, not verdicts; the model also looks
for issues outside the dictionary. The model is instructed to quote examples,
explain its judgment and offer alternatives without forcing a minimum number of
comments. RAG is optional. Both tools depend on the configured LLM, and their
quality should be reviewed by the author.

The original and full tool result are stored separately in memory for the
current user's discussion. A subsequent tool result replaces that context;
`/done` exits it and `/reset` clears it. For documents exceeding the discussion
budget, selected relevant passages are sent to the model and the model is
instructed to disclose the limitation. Ask about a specific quotation or
passage when it cannot identify the relevant section.
