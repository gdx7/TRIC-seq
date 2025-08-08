#!/usr/bin/env python3
"""
TRIC-seq utility: expand bacterial GFF3 annotations, generate UTRs from gaps,
and (optionally) mask duplicate rRNA copies in the genome FASTA.

Key features vs. the Colab prototype:
- Multi-contig aware (chromosome/plasmids supported)
- Deterministic, CLI-driven, with clear I/O schemas
- Robust GFF3 attribute parsing (gene, locus_tag, Name, ID, product)
- Strand-aware gap handling with --gap-cutoff and --utr-max
- Avoids assigning UTRs to tRNA/ncRNA/rRNA features (with fallback rename logic)
- rRNA masking strategies:
    * deduplicate-by-gene  (mask 2nd+ occurrences of the SAME rrnX-16S/23S/5S label)
    * keep-first-per-type  (mask all but the first 16S, 23S, 5S per contig)
    * none                 (do not mask)
- Writes a clean CSV and an (optional) masked FASTA
"""

import argparse
import csv
import logging
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord

# ----------------------------- CLI / Logging -------------------------------- #

def get_args():
    p = argparse.ArgumentParser(
        description="Expand GFF3 annotations, infer UTRs, and mask duplicate rRNAs."
    )
    p.add_argument("--gff3", required=True, help="Input GFF3 file")
    p.add_argument("--fasta", required=True, help="Input genome FASTA (one or more contigs)")
    p.add_argument("--out-annotations", required=True, help="Output CSV of expanded annotations")
    p.add_argument("--out-masked-fasta", default=None,
                   help="Output FASTA with rRNA duplicates masked (optional)")
    p.add_argument("--gap-cutoff", type=int, default=40,
                   help="Max intergenic gap to fill by extending the smaller gene (default: 40)")
    p.add_argument("--utr-max", type=int, default=40,
                   help="Maximum length assigned to each UTR at a large gap (default: 40)")
    p.add_argument("--expand-ncrna", type=int, default=20,
                   help="Flank added to ncRNA (default: 20)")
    p.add_argument("--expand-trna", type=int, default=10,
                   help="Flank added to tRNA (default: 10)")
    p.add_argument("--rrna-mask-strategy",
                   choices=["deduplicate-by-gene", "keep-first-per-type", "none"],
                   default="deduplicate-by-gene",
                   help="Strategy for masking rRNA duplicates in FASTA (default: deduplicate-by-gene)")
    p.add_argument("--rrna-types", default="16S,23S,5S",
                   help="Comma-separated rRNA type keywords to consider (default: 16S,23S,5S)")
    p.add_argument("--verbose", action="store_true", help="Verbose logging")
    return p.parse_args()


def setup_logging(verbose: bool):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(message)s",
        level=level
    )


# ----------------------------- Utilities ------------------------------------ #

ATTR_SPLIT = re.compile(r";\s*")
KV_SPLIT = re.compile(r"=")

def parse_attributes(attr_str: str) -> Dict[str, str]:
    """Parse a GFF3 attributes column into a dict."""
    d = {}
    if pd.isna(attr_str):
        return d
    for field in ATTR_SPLIT.split(str(attr_str)):
        if not field:
            continue
        if "=" not in field:
            # Some GFFs use key value without '='; skip or store as flag
            d[field] = ""
            continue
        k, v = KV_SPLIT.split(field, 1)
        d[k] = v
    return d


def coalesce_name(attrs: Dict[str, str]) -> Optional[str]:
    """Prefer gene -> Name -> locus_tag -> ID for a stable display name."""
    for k in ("gene", "Name", "locus_tag", "ID"):
        if k in attrs and attrs[k]:
            return attrs[k]
    return None


def extract_product(attrs: Dict[str, str]) -> Optional[str]:
    for k in ("product", "Product", "note", "Note", "description"):
        if k in attrs and attrs[k]:
            return attrs[k]
    return None


def contig_lengths_from_fasta(fasta_path: str) -> Dict[str, int]:
    lengths = {}
    for rec in SeqIO.parse(fasta_path, "fasta"):
        lengths[rec.id] = len(rec.seq)
    return lengths


@dataclass
class Feature:
    gene: str
    start: int
    end: int
    ftype: str
    strand: str
    locus: str
    chrom: str
    product: Optional[str] = None

    def length(self) -> int:
        return abs(self.end - self.start) + 1


# ----------------------------- Parsing GFF3 --------------------------------- #

ALLOWED_TYPES = {"CDS", "tRNA", "rRNA", "ncRNA", "pseudogene"}

def load_gff3(gff3_path: str, expand_ncrna: int, expand_trna: int,
              contig_lengths: Dict[str, int]) -> pd.DataFrame:
    cols = ["Seqid", "Source", "Type", "Start", "End", "Score", "Strand", "Phase", "Attributes"]
    gff = pd.read_csv(gff3_path, sep="\t", comment="#", header=None, names=cols, dtype=str, quoting=csv.QUOTE_NONE)
    gff = gff[gff["Seqid"].notna()].copy()
    gff = gff[gff["Type"].isin(ALLOWED_TYPES)].copy()

    # Cast coords
    gff["Start"] = gff["Start"].astype(int)
    gff["End"] = gff["End"].astype(int)

    # Parse attributes
    parsed_attrs = gff["Attributes"].apply(parse_attributes)
    gff["gene_attr"] = parsed_attrs.apply(coalesce_name)
    gff["locus_tag_attr"] = parsed_attrs.apply(lambda a: a.get("locus_tag") or "")
    gff["product_attr"] = parsed_attrs.apply(extract_product)

    # Coalesce Gene/Locus/Product
    gff["Gene"] = gff["gene_attr"].where(gff["gene_attr"].notna(), gff["locus_tag_attr"])
    gff["Gene"] = gff["Gene"].fillna("unknown_gene").replace("", "unknown_gene")
    gff["Locus"] = gff["locus_tag_attr"].replace("", "unknown_locus")
    gff["Chromosome"] = gff["Seqid"]
    gff["product"] = gff["product_attr"]

    df = gff[["Gene", "Start", "End", "Type", "Strand", "Locus", "Chromosome", "product"]].copy()

    # Expand ncRNA/tRNA bounds
    if expand_ncrna or expand_trna:
        logging.info("Expanding ncRNA by ±%d and tRNA by ±%d", expand_ncrna, expand_trna)
    if expand_ncrna:
        mask = df["Type"] == "ncRNA"
        df.loc[mask, "Start"] = (df.loc[mask, "Start"] - expand_ncrna).clip(lower=1)
        df.loc[mask, "End"] = df.loc[mask, "End"] + expand_ncrna
    if expand_trna:
        mask = df["Type"] == "tRNA"
        df.loc[mask, "Start"] = (df.loc[mask, "Start"] - expand_trna).clip(lower=1)
        df.loc[mask, "End"] = df.loc[mask, "End"] + expand_trna

    # Bound end per contig length
    df["End"] = df.apply(lambda r: min(r["End"], contig_lengths.get(r["Chromosome"], r["End"])), axis=1)

    return df


# ----------------- UTR inference & small-gap adjustment --------------------- #

UTR_TYPES_BLOCKED = {"tRNA", "ncRNA", "rRNA"}

def annotate_utrs_and_adjust_genes(df: pd.DataFrame, gap_cutoff: int,
                                   utr_max: int, contig_lengths: Dict[str, int]) -> pd.DataFrame:
    # Sort within contig
    df = df.sort_values(["Chromosome", "Start", "End"]).reset_index(drop=True)

    adjusted = df.to_dict(orient="records")
    new_features: List[dict] = []

    for chrom, chrom_df in df.groupby("Chromosome", sort=False):
        chrom_df = chrom_df.sort_values(["Start", "End"]).reset_index(drop=True)
        n = len(chrom_df)
        for i in range(n - 1):
            cur = chrom_df.iloc[i]
            nxt = chrom_df.iloc[i + 1]

            gap_start = cur["End"] + 1
            gap_end = nxt["Start"] - 1
            gap = gap_end - gap_start + 1
            if gap <= 0:
                continue  # overlap/adjacent

            # Skip creating UTRs flanking features that are tRNA/ncRNA/rRNA
            cur_blocked = cur["Type"] in UTR_TYPES_BLOCKED
            nxt_blocked = nxt["Type"] in UTR_TYPES_BLOCKED

            if gap < gap_cutoff:
                # extend smaller feature into the gap
                cur_len = abs(cur["End"] - cur["Start"]) + 1
                nxt_len = abs(nxt["End"] - nxt["Start"]) + 1
                if cur_len <= nxt_len:
                    # extend current end
                    for rec in adjusted:
                        if rec["Gene"] == cur["Gene"] and rec["Chromosome"] == chrom and rec["Start"] == cur["Start"]:
                            rec["End"] = max(rec["End"], gap_end)
                            break
                else:
                    # extend next start
                    for rec in adjusted:
                        if rec["Gene"] == nxt["Gene"] and rec["Chromosome"] == chrom and rec["End"] == nxt["End"]:
                            rec["Start"] = min(rec["Start"], gap_start)
                            break
                continue

            # Large gap: assign UTRs (≤ utr_max each side, not more than half gap)
            half = gap // 2
            utr_len = min(utr_max, half)

            if cur["Strand"] == nxt["Strand"]:
                if cur["Strand"] == "+":
                    # + strand: 3' UTR for current, 5' UTR for next
                    if not cur_blocked:
                        new_features.append({
                            "Gene": f"3′{cur['Gene']}",
                            "Start": gap_start,
                            "End": gap_start + utr_len - 1,
                            "Type": "3′UTR",
                            "Strand": "+",
                            "Locus": cur["Locus"],
                            "Chromosome": chrom,
                            "product": cur.get("product")
                        })
                    if not nxt_blocked:
                        new_features.append({
                            "Gene": f"5′{nxt['Gene']}",
                            "Start": gap_start + utr_len,
                            "End": gap_end,
                            "Type": "5′UTR",
                            "Strand": "+",
                            "Locus": nxt["Locus"],
                            "Chromosome": chrom,
                            "product": nxt.get("product")
                        })
                else:
                    # - strand: 3' UTR for next, 5' UTR for current
                    if not nxt_blocked:
                        new_features.append({
                            "Gene": f"3′{nxt['Gene']}",
                            "Start": gap_end - utr_len + 1,
                            "End": gap_end,
                            "Type": "3′UTR",
                            "Strand": "-",
                            "Locus": nxt["Locus"],
                            "Chromosome": chrom,
                            "product": nxt.get("product")
                        })
                    if not cur_blocked:
                        new_features.append({
                            "Gene": f"5′{cur['Gene']}",
                            "Start": gap_start,
                            "End": gap_end - utr_len,
                            "Type": "5′UTR",
                            "Strand": "-",
                            "Locus": cur["Locus"],
                            "Chromosome": chrom,
                            "product": cur.get("product")
                        })
            else:
                # Opposite strands: split into same-type UTRs on each side
                mid = gap_start + half - 1
                if cur["Strand"] == "+" and nxt["Strand"] == "-":
                    # 3′ ends face: two 3′ UTRs
                    if not cur_blocked:
                        new_features.append({
                            "Gene": f"3′{cur['Gene']}",
                            "Start": gap_start,
                            "End": mid,
                            "Type": "3′UTR",
                            "Strand": "+",
                            "Locus": cur["Locus"],
                            "Chromosome": chrom,
                            "product": cur.get("product")
                        })
                    if not nxt_blocked:
                        new_features.append({
                            "Gene": f"3′{nxt['Gene']}",
                            "Start": mid + 1,
                            "End": gap_end,
                            "Type": "3′UTR",
                            "Strand": "-",
                            "Locus": nxt["Locus"],
                            "Chromosome": chrom,
                            "product": nxt.get("product")
                        })
                else:
                    # 5′ ends face: two 5′ UTRs
                    if not cur_blocked:
                        new_features.append({
                            "Gene": f"5′{cur['Gene']}",
                            "Start": gap_start,
                            "End": mid,
                            "Type": "5′UTR",
                            "Strand": "-",
                            "Locus": cur["Locus"],
                            "Chromosome": chrom,
                            "product": cur.get("product")
                        })
                    if not nxt_blocked:
                        new_features.append({
                            "Gene": f"5′{nxt['Gene']}",
                            "Start": mid + 1,
                            "End": gap_end,
                            "Type": "5′UTR",
                            "Strand": "+",
                            "Locus": nxt["Locus"],
                            "Chromosome": chrom,
                            "product": nxt.get("product")
                        })

    # Combine and clean
    combined = pd.DataFrame(adjusted + new_features)
    combined = combined.dropna(subset=["Start", "End"]).copy()
    combined["Start"] = combined["Start"].astype(int)
    combined["End"] = combined["End"].astype(int)

    # Bound within contig
    combined["Start"] = combined.apply(lambda r: max(1, r["Start"]), axis=1)
    combined["End"] = combined.apply(lambda r: min(r["End"], contig_lengths.get(r["Chromosome"], r["End"])), axis=1)

    # Ensure no overlapping UTRs of the same type on the same strand within a contig
    combined = combined.sort_values(["Chromosome", "Strand", "Start", "End"]).reset_index(drop=True)
    cleaned = []
    last_by_key = {}  # (chrom,strand,type) -> end
    for _, row in combined.iterrows():
        if row["Type"] in ("3′UTR", "5′UTR"):
            key = (row["Chromosome"], row["Strand"], row["Type"])
            last_end = last_by_key.get(key, -1)
            if row["Start"] <= last_end:
                # shift start to avoid overlap; drop if invalid
                new_start = last_end + 1
                if new_start <= row["End"]:
                    row["Start"] = new_start
                else:
                    continue
            last_by_key[key] = row["End"]
        else:
            # reset “last_end” for non-UTR features on that strand to prevent unwanted carryover
            pass
        cleaned.append(row)
    combined = pd.DataFrame(cleaned)

    # Fallback: if any UTR got created adjacent to ncRNA/tRNA/rRNA due to earlier logic, rename back
    def maybe_restore_type(r):
        base_gene = r["Gene"].replace("3′", "").replace("5′", "")
        # find original type
        orig = df[(df["Chromosome"] == r["Chromosome"]) & (df["Gene"] == base_gene)]
        if not orig.empty and orig.iloc[0]["Type"] in UTR_TYPES_BLOCKED and r["Type"] in ("3′UTR", "5′UTR"):
            return orig.iloc[0]["Type"]
        return r["Type"]

    combined["Type"] = combined.apply(maybe_restore_type, axis=1)

    # Final tidy columns/order
    return combined[["Gene", "Start", "End", "Type", "Strand", "Locus", "Chromosome", "product"]].sort_values(
        ["Chromosome", "Start", "End"]
    ).reset_index(drop=True)


# --------------------------- rRNA masking ----------------------------------- #

RRN_GENE_RE = re.compile(r"^rrn([A-Za-z]+)[-_]?(16S|23S|5S)$", re.IGNORECASE)

def infer_rrna_label(row: pd.Series, rrna_types: List[str]) -> Optional[Tuple[str, str]]:
    """
    Return (label_key, type) to group duplicates.
    - label_key is the most specific identifier available:
        rrnX-16S if gene matches, else product-derived '16S'/'23S'/'5S', else None.
    """
    gene = (row.get("Gene") or "").strip()
    prod = (row.get("product") or "").strip()
    # Full rrnX-16S style
    m = RRN_GENE_RE.match(gene)
    if m:
        return (f"rrn{m.group(1).upper()}-{m.group(2).upper()}", m.group(2).upper())
    # Fall back to product substring
    for t in rrna_types:
        if t.lower() in gene.lower() or t.lower() in prod.lower():
            return (t.upper(), t.upper())
    return None


def mask_rrna_duplicates(
    fasta_path: str,
    df: pd.DataFrame,
    out_fasta: str,
    strategy: str,
    rrna_types: List[str],
):
    if not out_fasta or strategy == "none":
        return

    rrna = df[df["Type"] == "rRNA"].copy()
    if rrna.empty:
        logging.info("No rRNA features found; skipping FASTA masking.")
        return

    rrna["label"] = rrna.apply(lambda r: infer_rrna_label(r, rrna_types), axis=1)
    rrna = rrna[rrna["label"].notna()].copy()
    if rrna.empty:
        logging.info("No recognizable rRNA labels found; skipping masking.")
        return

    # Build mask intervals per contig
    mask_intervals: Dict[str, List[Tuple[int, int]]] = defaultdict(list)

    if strategy == "deduplicate-by-gene":
        # For the same rrnA-16S etc., keep the first by coordinate per contig; mask the rest
        rrna["label_key"] = rrna["label"].apply(lambda x: x[0])
        for (chrom, label_key), group in rrna.groupby(["Chromosome", "label_key"], sort=False):
            group = group.sort_values("Start")
            keep_first = True
            for _, row in group.iterrows():
                if keep_first:
                    keep_first = False
                    continue
                mask_intervals[chrom].append((row["Start"] - 1, row["End"]))  # 0-based slicing half-open
    elif strategy == "keep-first-per-type":
        # For each type (16S/23S/5S) per contig, keep the first occurrence; mask the remaining
        rrna["type_simple"] = rrna["label"].apply(lambda x: x[1])
        for (chrom, rtype), group in rrna.groupby(["Chromosome", "type_simple"], sort=False):
            group = group.sort_values("Start")
            keep_first = True
            for _, row in group.iterrows():
                if keep_first:
                    keep_first = False
                    continue
                mask_intervals[chrom].append((row["Start"] - 1, row["End"]))
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    # Apply masks
    records: List[SeqRecord] = []
    for rec in SeqIO.parse(fasta_path, "fasta"):
        seq = list(str(rec.seq).upper())
        for s, e in mask_intervals.get(rec.id, []):
            s = max(0, s)
            e = min(e, len(seq))
            if s < e:
                # Replace with 'A' to neutralize mapping; choose 'N' if preferred
                for i in range(s, e):
                    seq[i] = "A"
        newrec = SeqRecord(seq="".join(seq), id=rec.id, description=rec.description)
        records.append(newrec)

    SeqIO.write(records, out_fasta, "fasta")
    logging.info("Masked FASTA written: %s (strategy=%s)", out_fasta, strategy)


# --------------------------------- Main ------------------------------------- #

def main():
    args = get_args()
    setup_logging(args.verbose)

    rrna_types = [t.strip() for t in args.rrna_types.split(",") if t.strip()]
    logging.info("Loading FASTA to determine contig lengths…")
    contig_lengths = contig_lengths_from_fasta(args.fasta)

    logging.info("Parsing GFF3 and expanding features…")
    base_df = load_gff3(args.gff3, args.expand_ncrna, args.expand_trna, contig_lengths)

    logging.info("Annotating UTRs and adjusting small gaps (gap_cutoff=%d, utr_max=%d)…",
                 args.gap_cutoff, args.utr_max)
    final_df = annotate_utrs_and_adjust_genes(base_df, args.gap_cutoff, args.utr_max, contig_lengths)

    # Write CSV
    out_cols = ["Gene", "Start", "End", "Type", "Strand", "Locus", "Chromosome"]
    final_df[out_cols].sort_values(["Chromosome", "Start", "End"]).to_csv(args.out_annotations, index=False)
    logging.info("Annotations CSV written: %s", args.out_annotations)

    # Mask rRNA duplicates
    if args.out_masked_fasta:
        logging.info("Masking rRNA duplicates in FASTA (strategy=%s)…", args.rrna_mask_strategy)
        mask_rrna_duplicates(
            fasta_path=args.fasta,
            df=base_df,  # use base_df; UTRs not needed for masking
            out_fasta=args.out_masked_fasta,
            strategy=args.rrna_mask_strategy,
            rrna_types=rrna_types,
        )

    logging.info("Done.")


if __name__ == "__main__":
    main()
