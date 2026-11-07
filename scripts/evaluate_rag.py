#!/usr/bin/env python3
"""Reproducible relevance benchmark. --legacy uses the committed pre-change code."""
import argparse
import json
import statistics
import subprocess
import sys
import time
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def rss_mb(peak=False):
    try:
        import psutil
        info = psutil.Process().memory_info()
        if not peak:
            return info.rss / 1048576
        if hasattr(info, "peak_wset"):
            return info.peak_wset / 1048576
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    except ImportError:
        if sys.platform == "win32":
            import ctypes
            from ctypes import wintypes
            class Counters(ctypes.Structure):
                _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                            ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                            ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
            c = Counters()
            c.cb = ctypes.sizeof(c)
            ctypes.windll.kernel32.GetCurrentProcess.restype = wintypes.HANDLE
            ctypes.windll.psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
            ctypes.windll.psapi.GetProcessMemoryInfo(ctypes.windll.kernel32.GetCurrentProcess(), ctypes.byref(c), c.cb)
            return (c.PeakWorkingSetSize if peak else c.WorkingSetSize) / 1048576
        import resource
        # Linux ru_maxrss is peak RSS in KiB, not current RSS.
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy", action="store_true")
    parser.add_argument("--legacy-revision", default="593bf5b", help="Pre-change revision")
    parser.add_argument("--vector", action="store_true", help="Load model and sync the complete index")
    parser.add_argument("--calibrate", action="store_true", help="Sweep thresholds on the labelled cases")
    parser.add_argument("--data-dir", default=str(ROOT / "data"))
    parser.add_argument("--cases", default=str(ROOT / "tests" / "rag_queries.json"))
    args = parser.parse_args()
    if args.legacy:
        code = subprocess.check_output(["git", "show", f"{args.legacy_revision}:src/writer_rag.py"], cwd=ROOT, text=True, encoding="utf-8")
        module = types.ModuleType("legacy_rag")
        sys.modules[module.__name__] = module
        exec(compile(code, "legacy_rag.py", "exec"), module.__dict__)
        cls = module.WriterRAG
    else:
        import writer_rag
        from writer_rag import WriterRAG
        cls = WriterRAG
        if args.calibrate:
            if args.vector:
                writer_rag.SEMANTIC_MIN_SIMILARITY = -1
            else:
                writer_rag.KEYWORD_MIN_COVERAGE = 0
    cases = json.loads(Path(args.cases).read_text(encoding="utf-8"))
    start_rss, start = rss_mb(), time.perf_counter()
    rag = cls(data_dir=args.data_dir, use_local_embeddings=args.vector)
    print(f"construct_seconds={time.perf_counter()-start:.3f} rss_before_mb={start_rss:.1f} rss_construct_mb={rss_mb():.1f}")
    times, passed, positives, negatives, positive_pass, negative_pass = [], 0, 0, 0, 0, 0
    evaluated = []
    for case in cases:
        start = time.perf_counter()
        results = rag.search_similar(case["query"], n_results=3, category=case["category"])
        evaluated.append((case, results))
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        is_negative = case["source_type"] == "none"
        category_ok = all(r.metadata.get("category") == case["category"] for r in results)
        evidence_ok = any(any(term.casefold() in r.content.casefold() for term in case["evidence"]) for r in results)
        ok = not results if is_negative else bool(results) and category_ok and evidence_ok
        passed += ok
        negatives += is_negative
        positives += not is_negative
        negative_pass += ok and is_negative
        positive_pass += ok and not is_negative
        print(f"{'PASS' if ok else 'FAIL'} {case['category']:9} {case['query']} "
              f"seconds={elapsed:.3f} results={[(r.source, round(r.relevance, 3), r.metadata.get('category')) for r in results]}")
    print(f"TOTAL {passed}/{len(cases)} positive={positive_pass}/{positives} negative={negative_pass}/{negatives} "
          f"median_seconds={statistics.median(times):.3f} max_seconds={max(times):.3f} "
          f"rss_end_mb={rss_mb():.1f} peak_rss_mb={rss_mb(peak=True):.1f}")
    if args.calibrate:
        if args.legacy or (args.vector and rag._indexed_signature is None):
            print("Calibration unavailable: requested index was not usable.")
            return
        for threshold in (0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75):
            pos = neg = hybrid_pos = hybrid_neg = 0
            for case, raw in evaluated:
                results = [r for r in raw if r.relevance >= threshold]
                def correct(items):
                    if case["source_type"] == "none":
                        return not items
                    return bool(items) and all(r.metadata.get("category") == case["category"] for r in items) and any(
                        any(term.casefold() in r.content.casefold() for term in case["evidence"]) for r in items)
                hybrid = results
                if args.vector and not results:
                    hybrid = rag._keyword_fallback(case["query"], 3, case["category"])
                if case["source_type"] == "none":
                    neg += correct(results)
                    hybrid_neg += correct(hybrid)
                else:
                    pos += correct(results)
                    hybrid_pos += correct(hybrid)
            print(f"THRESHOLD {threshold:.2f} positive={pos}/{positives} negative={neg}/{negatives}" +
                  (f" hybrid_positive={hybrid_pos}/{positives} hybrid_negative={hybrid_neg}/{negatives}"
                   if args.vector else ""))


if __name__ == "__main__":
    main()
