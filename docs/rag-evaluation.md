# Ash RAG: operation and evaluation

## Corpus and provenance

The existing corpus and Chroma collections are retained. At evaluation time,
`craft.json` contained 3,270 chunks from seven craft books, `style.json` 561
fiction excerpts, and `editorial.json` 63 curated editorial entries. Fiction
is an example of literary style, not an instructional source. The editorial
entries have author/title/section labels, but the corresponding books are not
present here to substantiate their wording or attribution. They are labelled
**unverified summaries**, not book quotations. Other excerpts and their
metadata are also not independently checked against the books; in particular,
the existing craft file has almost no chapter metadata, and its one chapter
label appears to be an inline reference. The service preserves the supplied
metadata; it does not manufacture missing pages or chapters.

`methodique`, `feedback`, `block`, `develop`, `character` and `dialogue`
search craft; `style` and `praise` search fiction; `editor` and `roast`
search editorial material. `correct_text` is a separate proofreading call
and does not use RAG; normal chat and unrelated commands do not search either.
The no-argument methodique samples the shared craft corpus explicitly.
For commands using RAG, prompts forbid invented references and require exact
supplied text for direct quotations. This is an instruction to the LLM, not a
formal output verifier; generated responses and attributions still need review.

## Index and failure handling

One service is shared per resolved corpus path and one embedding model per
process. Model and Chroma initialization are lazy, serialized, and retried
after 60 seconds on failure. Each search checks corpus file timestamps and
sizes. Changed files are reloaded as a complete snapshot; Chroma records are
compared by stable ID, text and metadata, and only additions/changes are
encoded in bounded batches. Removed IDs are deleted from the corresponding
index. Existing collections are reused; corrupt HNSW collections can be
recovered into separately named collections without deleting the originals.
If initialization or querying fails, category-filtered keyword search is used.
An invalid corpus update leaves the last complete in-memory snapshot in use.
These locks are process-local; operate one indexing process per data directory.

Chroma 1.5.8 on the tested Windows workspace failed to reopen the craft HNSW
using its Unicode absolute path, but reopened the same collection via a path
relative to the working directory. The service uses a relative Chroma path
when possible. Keep the process working directory stable. During diagnosis
an additional `writers_recovered_craft` collection was created; it was not
deleted and is not needed when the original collection is readable.

Quoted explicit source requests such as `книге «... »` or `author "..."` are
restricted to matching available metadata; a missing title/author yields no
context. Other phrasings are not an exhaustive named-entity detector.

## Reproducible quality check

`tests/rag_queries.json` has 33 labelled requests: 25 with an expected category,
source type and topic evidence, plus eight irrelevant/missing-source requests.
The benchmark checks the top three for correct category and an expected
evidence substring, or requires no result for a negative case. It is stronger
than checking for nonempty Chroma output, but is a **coarse automated proxy**:
it is not a blind human relevance judgment, cannot establish attribution
accuracy, and can miss or accept semantically misleading results. The same
small set was used to tune thresholds; there is no held-out estimate.

Run from the repository root, after installing requirements and caching the
model. The baseline revision defaults to `593bf5b` and can be overridden with
`--legacy-revision`. `--data-dir data` also avoids the Windows path issue in
the old implementation.

```sh
python scripts/evaluate_rag.py --legacy --data-dir data
python scripts/evaluate_rag.py --data-dir data
python scripts/evaluate_rag.py --legacy --vector --data-dir data
python scripts/evaluate_rag.py --vector --data-dir data
python scripts/evaluate_rag.py --calibrate --data-dir data
python scripts/evaluate_rag.py --vector --calibrate --data-dir data
python -m pytest -q
```

For an offline VPS, pre-cache `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
and set `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`. If it is not cached,
the service falls back rather than downloading successfully. `OMP_NUM_THREADS`
and `MKL_NUM_THREADS` can limit CPU use; the measurements below used `2`.

Results on Windows, Python 3.13.12, Chroma 1.5.8, sentence-transformers 5.4.1,
PyTorch 2.11.0, existing original index:

| Search | Positive evidence | Negative rejection | Total |
| --- | ---: | ---: | ---: |
| Old keyword fallback | 16/25 | 3/8 | 19/33 |
| New keyword fallback | 22/25 | 8/8 | 30/33 |
| Old semantic search | 21/25 | 0/8 | 21/33 |
| New semantic + gated fallback | 23/25 | 8/8 | 31/33 |

The new semantic run still missed the expected evidence for
`кульминация развязка истории` and `пунктуация прямой речи и диалога`. The
former returned high-scoring but inadequate fragments. Confidence is not proof
of relevance. In a threshold sweep, pure semantic thresholds 0.50/0.55/0.60
scored respectively 20/18/15 of 25 positives and 8/8 negatives; with the
confidence-gated fallback they scored 22/22/23 positives and 8/8 negatives.
0.60 was selected on this set (0.75 also scored 23/25 but retained fewer
semantic results). Keyword coverage 0.30 scored 22/25 and 8/8; 0.25 allowed
one negative through. These thresholds should be recalibrated on larger,
independently judged Russian and English sets before claiming general quality.

Median warm search times in the comparison were 18 ms old vector, 16 ms new
vector/hybrid, 134 ms old keyword and 1 ms new keyword. The vector comparison
reused one loaded model across both implementations; peak process RSS was
1,023 MiB. The cold calibration took 86 s for the first search and peaked at
1,065 MiB, including model imports and index loading; keyword-only construction
used about 97 MiB RSS. These are local Windows measurements, not VPS guarantees.
A diagnostic recovery rebuild of 3,270 chunks took about 403 s on the same
machine with the earlier, larger encoding batch; current batches are bounded
to eight model inputs. No full re-embedding was needed when reopening the
original unchanged collections. Plan memory and first-index time accordingly.

The unit suite covers shared/lazy model initialization, concurrent acquisition,
real Chroma incremental changes/deletions, category routing and fallback,
missing sources, query failures, prompt assembly and the keyword benchmark.
No live Telegram or external LLM call was made in this evaluation.
