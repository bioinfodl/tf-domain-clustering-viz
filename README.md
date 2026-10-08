# TF domain clustering viewer

An interactive [Streamlit](https://streamlit.io) app to explore how the domains of human
transcription factors (TFs), predicted or annotated by several sources, can be merged into a single
set of non-redundant domains per protein.

For every protein, four sources describe domains, often with slightly different boundaries:

| Source | What it is |
|---|---|
| **merizo**, **chainsaw**, **unidoc-ndr** | Three structure-based domain parsers of AlphaFold models, as provided by [TED](https://ted.cathdb.info) (The Encyclopedia of Domains) |
| **uniprot** | Domain-like features annotated in [UniProtKB](https://www.uniprot.org) (`Domain`, `Zinc finger`, `DNA binding`) |

The app shows how a clustering rule (similarity metric + threshold) groups these domains, which
representative is kept in each cluster, and how many final domains each setting gives over the
whole dataset.

> This repository is a self-contained, read-only export of the visualisation. The code that builds
> the dataset (fetching from UniProt, TED and AlphaFold DB) lives elsewhere; see
> [Updating the data](#updating-the-data).

## Run it

```bash
pip install -r requirements.txt
streamlit run app.py
```

Python 3.11 or later. The app only needs [`data/tf_domains.csv`](data/tf_domains.csv).

## What the page shows

1. **Understanding the similarity metrics** – a worked example of IoU, Overlap, Dice and Coverage on
   two segments.
2. **Step-by-step view (one protein)** – choose a protein, a metric and a threshold, then go through
   four steps:
   1. all domains of the protein, one row per source;
   2. clustering: two domains are linked when their similarity is at least the threshold, and a
      cluster is a connected component (A~B and B~C put A, B and C together even if A and C are not
      similar);
   3. choice of the representative of each cluster;
   4. the final domains.

   Domains of a same source that overlap are drawn on separate lanes. A *Manual merge* box merges
   several domains of one source by hand. A banner reports the clusters of the protein that hold
   several domains of the same source.
3. **Total number of domains by threshold (all proteins)** – grouped bars for the four metrics at every
   threshold from 0.50 to 1.00, with the number of raw domains as a dashed line. Click the legend to
   hide or show a metric.
4. **Clusters with several domains of the same source (all proteins)** – for a chosen metric, how many
   clusters hold at least two domains of one source, per threshold and per source.

The sidebar holds the filters that apply to *every* plot:

- **Include unidoc-ndr** (off by default);
- **UniProt features**: which of `Domain`, `Zinc finger`, `DNA binding` to keep (default: `Domain`).
  Zinc fingers are annotated one finger at a time, so enabling them adds thousands of short domains.

The metric and threshold are *not* in the sidebar: each section has its own controls.

## Similarity metrics

For two segments A and B (positions are 1-based and inclusive, `|.|` is a length in residues):

| Metric | Formula | Comment |
|---|---|---|
| **IoU (Jaccard)** | `\|A∩B\| / \|A∪B\|` | The strictest one. |
| **Overlap (Szymkiewicz–Simpson)** | `\|A∩B\| / min(\|A\|, \|B\|)` | Equals 1 when one segment lies inside the other, so it chains a lot. |
| **Dice** | `\|A∩B\| / ((\|A\| + \|B\|) / 2)` | Same as `2\|A∩B\| / (\|A\| + \|B\|)`. |
| **Coverage** | `\|A∩B\| / max(\|A\|, \|B\|)` | Share of the larger segment that is covered. |

### Clustering is transitive

Because clusters are connected components, two domains of the same source can end up together
without being similar to each other:

- through a domain of another source that is close to both (for example two UniProt features that
  both match one TED domain);
- by overlapping each other (UniProt features of different types often do).

With IoU and no overlap between A and B, a threshold above 0.5 prevents the first case. It does not
when A and B overlap, and no threshold prevents it with the Overlap metric. The last two sections of
the page quantify this.

## Choice of the representative of a cluster

- A cluster with one domain: that domain.
- Otherwise the first TED method present in the order **merizo > chainsaw > unidoc-ndr**.
- If the cluster also contains UniProt domains, the representative is the union (smallest start,
  largest end) of that TED domain and of all the UniProt domains of the cluster; it is shown on a
  separate *merged* row.
- A cluster with only UniProt domains: the union of these domains.

## Data

[`data/tf_domains.csv`](data/tf_domains.csv): one row per domain.

| Column | Meaning |
|---|---|
| `uniprot_id` | UniProt accession of the protein |
| `source` | `uniprot`, `merizo`, `chainsaw`, `unidoc-ndr` or `summary` (TED consensus; not used by the app) |
| `domain_idx` | 0-based index of the domain within its source |
| `start`, `end` | first and last residue, 1-based, inclusive |
| `description` | UniProt label (empty for TED) |
| `feature_type` | UniProt feature type: `Domain`, `Zinc finger` or `DNA binding` (empty for TED) |
| `score` | TED method-level score (empty for UniProt) |
| `mean_plddt` | mean pLDDT over the domain, from the AlphaFold DB confidence file |
| `mean_pae` | mean intra-domain PAE, from the AlphaFold DB PAE file |

Current content: 26,502 rows for 1,651 proteins (UniProt 9,494, unidoc-ndr 5,870, merizo 4,359,
chainsaw 3,801, summary 2,978). The proteins are the reviewed human UniProt entries matching a list
of TF gene names.

### Things to know about the data

- **TED domains that are not contiguous** (for example `253-289_298-437`) are reduced to their
  bounding box `(min start, max end)`. Overlapping intervals of a same TED method are then merged, so
  the file never contains overlapping domains of one TED method. In the raw TED output we checked, no two
  domains of a method share a residue; the merging only concerns discontinuous domains whose box
  contains another domain.
- **UniProt feature types.** UniProt annotates them in different sections of an entry, with no
  hierarchy between them, so the same module can appear as several types:
  [`Domain`](https://www.uniprot.org/help/domain) (a fold, mostly from PROSITE, Pfam and SMART),
  [`DNA binding`](https://www.uniprot.org/help/dna_bind) (DNA-binding domains and motifs such as
  helix-turn-helix; the basic region of bHLH and bZIP proteins) and
  [`Zinc finger`](https://www.uniprot.org/help/zn_fing) (one feature per finger; includes types such
  as RING that do not bind DNA). About 8% of the `DNA binding` features lie inside a `Domain`.
- Some proteins have no UniProt domain, and a few are absent from TED; they only appear with the
  sources that describe them.

## Updating the data

The dataset is produced by the `TFilament` pipeline of the main toolkit repository. To refresh this
viewer after a new run:

```bash
scripts/sync_from_toolkit.sh /path/to/toolkit
git add -A && git commit -m "Update data" && git push
```

The script copies the app and `tf_domains.csv` from the toolkit repository. A push on the deployed
branch redeploys the app.

## Deploying on Streamlit Community Cloud

1. Push this repository to GitHub (public).
2. On [share.streamlit.io](https://share.streamlit.io): *Create app*, then pick this repository and
   its main branch.
3. Main file path: `app.py`. In *Advanced settings*, choose Python 3.11.
4. *Deploy*. The first visit computes the global plots, which can take a while; they are cached afterwards.

## Sources and terms of use

The data come from public resources; check the terms of use of each before reusing them:
[UniProt](https://www.uniprot.org/help/license), [TED](https://ted.cathdb.info) and the
[AlphaFold Protein Structure Database](https://alphafold.ebi.ac.uk).
The three parsers are Chainsaw, Merizo and UniDoc-NDR; cite their authors and TED if you use the
domains in a publication.

No license is set for the code in this repository yet; add one before others reuse it.
