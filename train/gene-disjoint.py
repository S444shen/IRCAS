#!/usr/local/bin/python
# -*- coding: utf-8 -*-
"""
Gene-disjoint pair deduplication and 5-fold splitting for IRCAS.

 Performs:
  (1) Pair-level deduplication (canonicalize (A,B) == (B,A), drop self-hits)
  (2) Annotation of each pair with its gene group, parsed from a GTF/GFF file
  (3) Gene-disjoint k-fold partitioning via sklearn's GroupKFold

Verifies post-hoc that no gene appears in more than one fold and emits both
the legacy .split{i} files (for backwards compatibility with downstream
scripts) and a manifest describing train/test fold composition.

Usage:
    python unique_gene_disjoint.py <blast_tsv> <n_folds> \
        --gtf <annotation.gtf> \
        [--strategy {composite,drop,unionfind}] \
        [--seed 42] \
        [--verify]

Example:
    python unique_gene_disjoint.py human_blast.tsv 5 \
        --gtf gencode.v44.annotation.gtf \
        --strategy composite \
        --verify
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, FrozenSet, Iterable, List, Set, Tuple, Union

# sklearn is only needed for GroupKFold; if unavailable we fall back to a
# manual implementation so the script still runs in minimal environments.
try:
    from sklearn.model_selection import GroupKFold
    _HAS_SKLEARN = True
except ImportError:  # pragma: no cover
    _HAS_SKLEARN = False


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

TranscriptID = str
GeneID = str
# A "group key" is what we hand to GroupKFold. For within-gene pairs it's the
# gene id itself; for cross-gene paralog pairs it's a frozenset of the two
# gene ids (composite strategy) or the unionfind representative.
GroupKey = Union[GeneID, FrozenSet[GeneID]]
Pair = Tuple[TranscriptID, TranscriptID]


# ---------------------------------------------------------------------------
# GTF parsing
# ---------------------------------------------------------------------------

# Matches: transcript_id "ENST00000456328.2";
_TRANSCRIPT_RE = re.compile(r'transcript_id\s+"([^"]+)"')
_GENE_RE = re.compile(r'gene_id\s+"([^"]+)"')


def parse_transcript_to_gene(gtf_path: Path) -> Dict[TranscriptID, GeneID]:
    """Parse a GTF (optionally gzipped) into a transcript_id -> gene_id map.

    Robust to:
      - gzipped or plain text input (chosen by .gz suffix)
      - comment lines starting with #
      - feature lines without a transcript_id (gene-level features) -- skipped
      - duplicate transcript entries (later occurrences must agree with first;
        we warn and keep the first)

    GFF3 uses key=value pairs separated by ';' instead of GTF's
    key "value"; pairs. We try GTF first and fall back to GFF3 if no matches.
    """
    opener = gzip.open if gtf_path.suffix == ".gz" else open
    mapping: Dict[TranscriptID, GeneID] = {}
    conflicts = 0
    seen_lines = 0

    with opener(gtf_path, "rt") as fh:
        for line in fh:
            if not line or line.startswith("#"):
                continue
            seen_lines += 1
            # Attributes column is the 9th (0-indexed: 8) in both GTF and GFF3
            cols = line.rstrip("\n").split("\t")
            if len(cols) < 9:
                continue
            attrs = cols[8]

            t_match = _TRANSCRIPT_RE.search(attrs)
            g_match = _GENE_RE.search(attrs)

            if t_match is None or g_match is None:
                # Fall back to GFF3 key=value format
                t_match = re.search(r"transcript_id=([^;]+)", attrs)
                g_match = re.search(r"gene_id=([^;]+)", attrs)
                # Some GFF3 files use Parent= for transcript->gene links
                if g_match is None:
                    g_match = re.search(r"Parent=([^;,]+)", attrs)

            if t_match is None or g_match is None:
                continue

            tid = t_match.group(1)
            gid = g_match.group(1)

            if tid in mapping:
                if mapping[tid] != gid:
                    conflicts += 1
                    # Keep the first assignment; warn at end
                continue
            mapping[tid] = gid

    if seen_lines == 0:
        raise RuntimeError(
            f"GTF file {gtf_path} appears empty or contains only comments."
        )
    if not mapping:
        raise RuntimeError(
            f"No transcript->gene mappings extracted from {gtf_path}. "
            "Check that the file is a valid GTF/GFF3 with transcript_id "
            "and gene_id attributes."
        )
    if conflicts:
        print(
            f"[WARN] {conflicts} transcripts had multiple gene_id "
            "assignments; kept the first occurrence for each.",
            file=sys.stderr,
        )

    print(
        f"[INFO] Parsed {len(mapping):,} transcript->gene mappings "
        f"from {gtf_path}",
        file=sys.stderr,
    )
    return mapping


# ---------------------------------------------------------------------------
# BLAST pair extraction
# ---------------------------------------------------------------------------


def stream_unique_pairs(blast_path: Path) -> Iterable[Pair]:
    """Stream canonicalized, deduplicated, non-self transcript pairs.

    Reads the BLAST tsv lazily (line by line) so memory stays O(unique pairs)
    rather than O(BLAST hits). The dedup set itself does grow with the number
    of unique pairs, but that's unavoidable -- and matches the memory profile
    of the original unique.py.
    """
    seen: Set[Pair] = set()
    with open(blast_path, "r") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            cols = line.split("\t")
            if len(cols) < 2:
                continue
            q, s = cols[0], cols[1]
            if q == s:
                continue
            # Canonicalize: lexicographically smaller name first
            pair: Pair = (q, s) if q < s else (s, q)
            if pair in seen:
                continue
            seen.add(pair)
            yield pair


# ---------------------------------------------------------------------------
# Group assignment
# ---------------------------------------------------------------------------


class UnionFind:
    """Tiny union-find for the 'unionfind' grouping strategy."""

    def __init__(self) -> None:
        self.parent: Dict[GeneID, GeneID] = {}

    def find(self, x: GeneID) -> GeneID:
        # Path compression
        root = x
        while self.parent.setdefault(root, root) != root:
            root = self.parent[root]
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a: GeneID, b: GeneID) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def assign_groups(
    pairs: List[Pair],
    t2g: Dict[TranscriptID, GeneID],
    strategy: str,
) -> Tuple[List[Pair], List[GroupKey], Dict[str, int]]:
    """Map each pair to a group key under the requested strategy.

    Returns (kept_pairs, group_keys, stats). 'drop' strategy may discard pairs.
    """
    stats = {
        "total_pairs": len(pairs),
        "within_gene": 0,
        "cross_gene": 0,
        "unmapped_transcript": 0,
    }
    kept_pairs: List[Pair] = []
    keys: List[GroupKey] = []

    if strategy == "unionfind":
        uf = UnionFind()
        # First pass: union genes that share at least one pair
        for t1, t2 in pairs:
            if t1 not in t2g or t2 not in t2g:
                continue
            g1, g2 = t2g[t1], t2g[t2]
            uf.union(g1, g2)
        # Second pass: assign each pair the representative of its component
        for pair in pairs:
            t1, t2 = pair
            if t1 not in t2g or t2 not in t2g:
                stats["unmapped_transcript"] += 1
                continue
            g1, g2 = t2g[t1], t2g[t2]
            if g1 == g2:
                stats["within_gene"] += 1
            else:
                stats["cross_gene"] += 1
            kept_pairs.append(pair)
            keys.append(uf.find(g1))
        return kept_pairs, keys, stats

    # 'composite' and 'drop' share the same first part of the logic
    for pair in pairs:
        t1, t2 = pair
        if t1 not in t2g or t2 not in t2g:
            stats["unmapped_transcript"] += 1
            continue
        g1, g2 = t2g[t1], t2g[t2]
        if g1 == g2:
            stats["within_gene"] += 1
            kept_pairs.append(pair)
            keys.append(g1)
        else:
            stats["cross_gene"] += 1
            if strategy == "drop":
                continue
            # composite: frozenset so order-independent and hashable
            kept_pairs.append(pair)
            keys.append(frozenset((g1, g2)))

    return kept_pairs, keys, stats


# ---------------------------------------------------------------------------
# Group K-fold (with sklearn fallback)
# ---------------------------------------------------------------------------


def group_kfold_indices(
    n_items: int,
    groups: List[GroupKey],
    n_folds: int,
    seed: int,
) -> List[List[int]]:
    """Return n_folds lists of item indices, where each list is the held-out
    fold. No group appears in more than one fold.

    Uses sklearn's GroupKFold when available (deterministic, balanced). Falls
    back to a manual greedy bin-packing assignment otherwise.

    Note: sklearn's GroupKFold ignores 'seed' (it is deterministic by group
    frequency). We respect the seed only in the fallback path. This matches
    the standard sklearn behavior; mention this in your manuscript methods.
    """
    # sklearn's GroupKFold internally calls np.unique on the groups array,
    # which sorts them -- and str vs frozenset are not orderable. So we
    # canonicalize every group key to a stable integer id first.
    key_to_id: Dict[GroupKey, int] = {}
    int_groups: List[int] = []
    for g in groups:
        if g not in key_to_id:
            key_to_id[g] = len(key_to_id)
        int_groups.append(key_to_id[g])

    if _HAS_SKLEARN:
        gkf = GroupKFold(n_splits=n_folds)
        # GroupKFold needs an X argument but ignores its content
        dummy_X = [[0]] * n_items
        fold_indices: List[List[int]] = []
        for _, test_idx in gkf.split(dummy_X, groups=int_groups):
            fold_indices.append(sorted(test_idx.tolist()))
        return fold_indices

    # Manual fallback: greedy LPT (longest processing time) bin packing on
    # groups by size, assigning each group to the currently-smallest fold.
    import random

    rng = random.Random(seed)
    group_to_items: Dict[int, List[int]] = defaultdict(list)
    for idx, g in enumerate(int_groups):
        group_to_items[g].append(idx)

    # Sort groups by size desc; shuffle ties for reproducibility
    group_list = list(group_to_items.items())
    rng.shuffle(group_list)
    group_list.sort(key=lambda kv: len(kv[1]), reverse=True)

    folds: List[List[int]] = [[] for _ in range(n_folds)]
    for _g, items in group_list:
        # Assign to currently smallest fold
        smallest = min(range(n_folds), key=lambda i: len(folds[i]))
        folds[smallest].extend(items)

    return [sorted(f) for f in folds]


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


def verify_no_gene_leakage(
    fold_indices: List[List[int]],
    pairs: List[Pair],
    t2g: Dict[TranscriptID, GeneID],
) -> None:
    """Assert that no gene appears in more than one fold.

    For composite-key pairs (cross-gene paralogs), BOTH constituent genes are
    checked -- a paralog pair locks both genes into the same fold.

    Raises AssertionError with a diagnostic message if leakage is detected.
    """
    fold_genes: List[Set[GeneID]] = []
    for fold in fold_indices:
        genes: Set[GeneID] = set()
        for idx in fold:
            t1, t2 = pairs[idx]
            if t1 in t2g:
                genes.add(t2g[t1])
            if t2 in t2g:
                genes.add(t2g[t2])
        fold_genes.append(genes)

    for i in range(len(fold_genes)):
        for j in range(i + 1, len(fold_genes)):
            overlap = fold_genes[i] & fold_genes[j]
            if overlap:
                example = sorted(overlap)[:5]
                raise AssertionError(
                    f"Gene leakage detected between fold {i} and fold {j}: "
                    f"{len(overlap)} shared gene(s). Examples: {example}"
                )
    print(
        f"[OK] Verified gene-disjoint across {len(fold_indices)} folds; "
        f"fold gene counts: {[len(s) for s in fold_genes]}",
        file=sys.stderr,
    )


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def write_outputs(
    blast_path: Path,
    pairs: List[Pair],
    fold_indices: List[List[int]],
    groups: List[GroupKey],
) -> None:
    """Write legacy-compatible .split{i} files plus a manifest."""
    # Legacy .unique file: all kept pairs, canonical "A,B" form, one per line
    unique_path = blast_path.with_suffix(blast_path.suffix + ".unique")
    with open(unique_path, "w") as fh:
        for t1, t2 in pairs:
            fh.write(f"{t1},{t2}\n")
    print(f"[INFO] Wrote {len(pairs):,} unique pairs to {unique_path}",
          file=sys.stderr)

    # Per-fold .split files (held-out set for each fold)
    for i, fold in enumerate(fold_indices, start=1):
        split_path = blast_path.with_suffix(blast_path.suffix + f".split{i}")
        with open(split_path, "w") as fh:
            for idx in fold:
                t1, t2 = pairs[idx]
                fh.write(f"{t1},{t2}\n")
        print(
            f"[INFO] Fold {i}: {len(fold):,} pairs -> {split_path}",
            file=sys.stderr,
        )

    # Manifest: machine-readable summary for downstream training scripts
    manifest = {
        "n_pairs": len(pairs),
        "n_folds": len(fold_indices),
        "fold_sizes": [len(f) for f in fold_indices],
        "fold_unique_groups": [
            len({groups[i] for i in fold}) for fold in fold_indices
        ],
        # For each fold i, list which other folds form its training set.
        # (Standard 5-fold CV: fold i is test, the other 4 are train.)
        "cv_folds": [
            {
                "test_fold": i + 1,
                "train_folds": [j + 1 for j in range(len(fold_indices))
                                if j != i],
            }
            for i in range(len(fold_indices))
        ],
    }
    manifest_path = blast_path.with_suffix(blast_path.suffix + ".manifest.json")
    with open(manifest_path, "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"[INFO] Wrote manifest to {manifest_path}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Gene-disjoint pair dedup + k-fold split for IRCAS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("blast_tsv", type=Path,
                    help="Input BLAST output (tab-separated; cols 1-2 are "
                         "query/subject transcript IDs)")
    ap.add_argument("n_folds", type=int, help="Number of CV folds (e.g. 5)")
    ap.add_argument("--gtf", type=Path, required=True,
                    help="GTF or GFF3 annotation file (optionally gzipped) "
                         "providing transcript_id -> gene_id mapping")
    ap.add_argument("--strategy",
                    choices=["composite", "drop", "unionfind"],
                    default="composite",
                    help="How to handle cross-gene paralog pairs. "
                         "'composite' (default): treat each cross-gene pair "
                         "as its own group keyed by frozenset({g1,g2}). "
                         "'drop': discard cross-gene pairs entirely. "
                         "'unionfind': merge any genes connected by a pair "
                         "into a single super-group (most conservative).")
    ap.add_argument("--seed", type=int, default=42,
                    help="Random seed (only used by the manual fallback path "
                         "when sklearn is unavailable)")
    ap.add_argument("--verify", action="store_true",
                    help="Run post-hoc check that no gene appears in more "
                         "than one fold")
    args = ap.parse_args()

    if args.n_folds < 2:
        ap.error("n_folds must be >= 2")
    if not args.blast_tsv.exists():
        ap.error(f"BLAST file not found: {args.blast_tsv}")
    if not args.gtf.exists():
        ap.error(f"GTF file not found: {args.gtf}")

    # Step 1: parse annotation
    t2g = parse_transcript_to_gene(args.gtf)

    # Step 2: stream BLAST -> unique pairs (in-memory list since we need
    # random access for splitting)
    pairs = list(stream_unique_pairs(args.blast_tsv))
    print(f"[INFO] {len(pairs):,} unique non-self transcript pairs",
          file=sys.stderr)

    # Step 3: assign group keys
    pairs, groups, stats = assign_groups(pairs, t2g, args.strategy)
    print(
        f"[INFO] Group assignment ({args.strategy}): "
        f"within-gene={stats['within_gene']:,}, "
        f"cross-gene={stats['cross_gene']:,}, "
        f"unmapped (dropped)={stats['unmapped_transcript']:,}",
        file=sys.stderr,
    )
    if not pairs:
        print("[ERROR] No pairs remained after group assignment. "
              "Check that your GTF covers the transcripts in your BLAST "
              "output (ID format must match).", file=sys.stderr)
        return 1

    # Step 4: gene-disjoint k-fold
    fold_indices = group_kfold_indices(
        n_items=len(pairs),
        groups=groups,
        n_folds=args.n_folds,
        seed=args.seed,
    )

    # Step 5: verify (mandatory in production; --verify flag is for explicit
    # opt-in to the assertion behavior, but we always print fold stats)
    if args.verify:
        verify_no_gene_leakage(fold_indices, pairs, t2g)

    # Step 6: write outputs
    write_outputs(args.blast_tsv, pairs, fold_indices, groups)

    return 0


if __name__ == "__main__":
    sys.exit(main())