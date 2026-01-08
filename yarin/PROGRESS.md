# Akkadian LLM Dataset Unification - Progress Document

## Task 0: Project Context Understanding

### Current Project State (from v_1/README.md)

**Goal:** Train a PyTorch model for Akkadian text (next-token prediction / MLM) and analyze using SAEs.

### What Already Exists:

#### Data Sources Currently Available:
1. **eBL (Electronic Babylonian Library)**: ~28,000 fragments in `v_1/data/raw/extracted/full_corpus_dir/`
   - Each fragment = 1 CSV file (e.g., `EBL_3NT.907.267.csv`)
   - Public collection

2. **Archibab**: 1,541 texts in single file `v_1/data/raw/archibab.csv`
   - Private collection

#### Common CSV Schema (10 columns):
| Column | Description |
|--------|-------------|
| `fragment_id` | Unique identifier (e.g., "1848,0720.121") |
| `fragment_line_num` | Line number in fragment |
| `index_in_line` | Word position in line |
| `word_language` | Language (AKKADIAN, SUMERIAN, etc.) |
| `value` | Original transliteration text |
| `clean_value` | Cleaned version |
| `lemma` | Base form of word |
| `domain` | Text genre (e.g., "CANONICAL > Divination") |
| `place_discovery` | Archaeological site |
| `place_composition` | Original composition location |

#### Key Statistics (from EDA):
- Total Fragments: 28,194
- Total Words: ~1M+
- Akkadian Words: ~900k+
- Unique Akkadian Tokens: ~40k-60k
- Language Distribution: ~90% Akkadian, ~8% Sumerian, <1% Emesal
- Median fragment length: ~20-30 words
- 95th percentile: ~150-200 words

#### Important Notes:
- Words with breaks/incomplete fragments were REMOVED during preprocessing
- No temporal/period metadata available (genres.json has genre types only)
- place_discovery and place_composition are sparse

### What Needs to Be Added:
Two additional data sources need to be integrated:
1. **ORACC** (Open Richly Annotated Cuneiform Corpus): https://oracc.museum.upenn.edu/
2. **SEAL** (Sources of Early Akkadian Literature): https://seal.huji.ac.il/

### Project Philosophy:
- Build organically (no empty folders)
- Notebooks first, then scripts
- Clean code with minimal dependencies

---

## Task 1: Gemini Deep Research Summary Analysis

### Source: Akk/gemmini_dr_summary.md

### Key Insights:

#### 1. ORACC Architecture
- **Data Format**: Hierarchical JSON using CDL (Cuneiform Description Language)
- **Access**: Project ZIP files via Open Data API
- **URL Pattern**: `http://oracc.museum.upenn.edu/json/{project}.zip`
- **License**: Creative Commons Attribution Share-Alike

**CDL Node Types:**
| Node | Description | Action |
|------|-------------|--------|
| `c` (Chunk) | Structural container (tablet, face, column) | Recurse |
| `d` (Discontinuity) | Breaks, missing surfaces | Insert `<MISSING>` token |
| `l` (Line) | Text line container | Extract line number |
| `f` (Form) | The actual word/token | Extract data |

**Field Mapping (ORACC JSON -> Our Schema):**
- `form` -> `value`
- `norm` -> `clean_value`
- `cf` (Citation Form) -> `lemma`
- `lang` -> `word_language`
- `label` in `l` node -> `fragment_line_num`
- P-number from filename -> `fragment_id`

**Priority Projects for Akkadian:**
| Project | Content | Dialect |
|---------|---------|---------|
| saao/saa01-21 | Neo-Assyrian Letters & Admin | Neo-Assyrian |
| rinap/rinap1-5 | Royal Inscriptions | Standard Babylonian |
| riao | Early Assyrian Inscriptions | Old Assyrian |
| cams/gkab | Geographic/Literary | Various |

#### 2. SEAL Architecture
- **Format**: Web-based catalogue, NOT bulk JSON
- **Requires**: Two-stage scraping approach

**Stage 1 - Metadata CSV:**
- Download from Collections page
- Contains: SEAL no., CDLI no., Genre, Provenance

**Stage 2 - Content Scraping:**
- URL pattern: `https://seal.huji.ac.il/node/{SEAL_NO}`
- Parse HTML to extract transliteration
- Split lines by `<br>` or `<tr>` elements

**Field Mapping (SEAL -> Our Schema):**
- CDLI no. or "SEAL_{no}" -> `fragment_id`
- Regex from "1. text" -> `fragment_line_num`
- Split by whitespace -> tokens with `index_in_line`
- Default "AKKADIAN" -> `word_language`
- Raw token -> `value`
- Cleaned (remove brackets) -> `clean_value`
- **EMPTY** -> `lemma` (SEAL doesn't provide lemmatization)

#### 3. Standardization Rules

**Unicode Normalization:**
- Convert ALL ASCII transliterations to UTF-8 Unicode
- ORACC uses: sz, t, (ASCII)
- eBL/SEAL use: s, t (Unicode)
- Rule: Standardize to Unicode

**Subscript Mapping:**
```
0 -> ₀, 1 -> ₁, 2 -> ₂, ... x -> ₓ
```

**Clean Value Generation:**
- Remove: `[ ]` (breaks), `?` (uncertainty), `*` (collation), `#` (damage)
- Example: `[i]-par#-ra-as` -> `iparras`

#### 4. Target Directory Structure
```
data/processed/
├── ebl/
├── archibab/
├── seal/
├── oracc/
└── unified/
```

#### 5. Deduplication Strategy
- Use `fragment_id` (P-number) as primary key
- Priority: eBL > ORACC > Archibab > SEAL (richest metadata first)

#### 6. LLM Dataset Preparation
- Group by `fragment_id`
- Sort by `fragment_line_num` and `index_in_line`
- Join `clean_value` fields to form text
- Apply subword tokenizer (WordPiece)
- Slice into fixed-length windows (512 tokens)

---

## Summary for Review - Tasks 0 & 1

### Task 0 Summary:
You have a working project structure with eBL (~28k fragments) and Archibab (1.5k texts) already processed into a consistent 10-column CSV schema. The data is ~90% Akkadian with vocabulary of 40-60k unique tokens.

### Task 1 Summary:
The Gemini research provides a complete blueprint:

1. **ORACC**: Download project ZIPs, parse recursive CDL JSON, extract `f` nodes for tokens
2. **SEAL**: Two-stage process - download metadata CSV, then scrape individual text pages
3. **Standardization**: Unicode normalization, clean value generation (remove editorial marks)
4. **Schema**: Both can be mapped to your existing 10-column format
5. **Challenges**:
   - ORACC requires recursive JSON parsing
   - SEAL requires web scraping (polite delays needed)
   - SEAL has no lemmatization data

**Ready for Task 2?** I'll investigate the actual download mechanisms for ORACC and SEAL.

---

## Task 2: How to Download from SEAL and ORACC

### ORACC Download Method (VERIFIED from Akk/preprocessing/scraping.py)

The Akk repository contains **working code** for downloading ORACC data. Here's the proven approach:

#### Step 1: Get List of All Projects
```python
import requests
res = requests.get('http://oracc.museum.upenn.edu/projects.json')
projects_list = json.loads(res.content)['public']
# Returns list like: ['saao', 'rinap', 'riao', 'cams', ...]
```

#### Step 2: Download Each Project as ZIP
```python
import zipfile
import io

for project_name in projects_list:
    project_json_url = f'http://oracc.org/{project_name}/json'
    r = requests.get(project_json_url, stream=True)
    with zipfile.ZipFile(io.BytesIO(r.content)) as zip_ref:
        zip_ref.extractall(output_dir)
```

#### ORACC URL Patterns:
| Purpose | URL Pattern |
|---------|-------------|
| Project List | `http://oracc.museum.upenn.edu/projects.json` |
| Project ZIP | `http://oracc.org/{project_name}/json` |
| Text HTML | `http://oracc.iaas.upenn.edu/{project_name}/{text_id}/html` |

#### Downloaded Structure:
```
jsons/
├── {project_name}/
│   ├── catalogue.json      # Metadata for all texts
│   └── corpusjson/         # Individual text files
│       ├── P123456.json
│       ├── P123457.json
│       └── ...
```

#### Key JSON Structure (from scraping.py analysis):
```python
# Access the CDL structure:
cur_json['cdl'][0]['cdl'][-1]['cdl']  # Gets to sentence level

# Node types:
# - 'c' node: Sentence/chunk container
# - 'l' node: Word/lemma node
# - Contains 'f' with: form, lang, norm, cf (citation form)

# Check if word:
d.get('node') == 'l'  # True for word nodes
d['f']['form']        # The transliteration
d['f']['lang']        # Language code ('akk', 'sux', etc.)
```

#### Filtering Akkadian:
The Akk repo filters out texts that are >50% Sumerian:
```python
if text_lang_counter['sux'] + text_lang_counter['sux-x-emesal'] > 0.5 * sum(text_lang_counter.values()):
    continue  # Skip Sumerian-dominant texts
```

---

### SEAL Download Method

SEAL does **NOT** have a bulk download API. Based on web research:

#### Option 1: Metadata CSV + Scraping (Recommended)
1. **Get Metadata CSV**: Navigate to https://seal.huji.ac.il/collections and use "Download CSV"
   - Contains: SEAL no., CDLI no., Genre, Provenance

2. **Scrape Individual Texts**:
   - URL Pattern: `https://seal.huji.ac.il/node/{SEAL_NO}`
   - Parse HTML to extract transliteration from text blocks

#### Option 2: Check CDLI Cross-Reference
- Many SEAL texts have CDLI P-numbers
- These might also exist in ORACC
- Could reduce scraping needs

#### SEAL HTML Structure (from Gemini research):
- Text container: `div.field-name-body` or `table.text-display`
- Lines separated by `<br>` or `<tr>` elements
- Line format: "1. i-nu-ma i-lu..." (number + tokens)

#### SEAL Scraping Code Outline:
```python
import requests
from bs4 import BeautifulSoup
import time

def scrape_seal_text(seal_no):
    url = f"https://seal.huji.ac.il/node/{seal_no}"
    time.sleep(1)  # Polite delay
    res = requests.get(url)
    soup = BeautifulSoup(res.text, "html.parser")

    # Find text container
    text_div = soup.find('div', class_='field-name-body')
    # Parse lines and tokens...

    return parsed_text
```

#### Important Notes for SEAL:
- **License**: CC BY-NC-ND (non-commercial, no derivatives)
- **No lemmatization** available from SEAL
- **Requires polite scraping** (1 second delay between requests)
- Contains ~900 literary compositions

---

### Comparison Summary

| Aspect | ORACC | SEAL |
|--------|-------|------|
| **Access Method** | Bulk ZIP download | Web scraping |
| **Data Format** | JSON (CDL structure) | HTML |
| **API Available** | Yes (projects.json) | No |
| **Lemmatization** | Yes (cf field) | No |
| **Normalization** | Yes (norm field) | No (must generate) |
| **Volume** | Large (many projects) | ~900 texts |
| **License** | CC BY-SA | CC BY-NC-ND |

### Recommended Download Priority:
1. **ORACC First**: Bulk download, structured data, includes lemmatization
2. **SEAL Second**: Requires more work, but adds unique literary texts

### Key Projects to Download from ORACC:
Based on Akkadian content priority:
- `saao` (saa01-saa21): Neo-Assyrian letters & admin
- `rinap` (rinap1-5): Royal inscriptions
- `riao`: Early Assyrian inscriptions
- `cams/gkab`: Geographic/literary texts

---

## Task 3: Fields, Metadata, and File Naming Conventions

### Current Data Schema Comparison

#### eBL Files (Electronic Babylonian Library)
**File Naming**: `EBL_{fragment_id}.csv`
- Example: `EBL_1848,0720.121.csv`
- Prefix: `EBL_`
- Fragment ID uses comma notation (museum accession numbers)

**Columns** (11 columns, with unnamed index):
| Column | Type | Example | Notes |
|--------|------|---------|-------|
| (index) | int | 0 | Unnamed row index |
| `fragment_id` | str | "1848,0720.121" | Museum number |
| `fragment_line_num` | int | 8 | Line on tablet |
| `index_in_line` | int | 1 | Word position |
| `word_language` | str | "AKKADIAN" | UPPERCASE |
| `value` | str | "BE-ma" | Raw transliteration |
| `clean_value` | str | "BE-ma" | Often same as value |
| `lemma` | list | ['šumma I'] | Python list as string |
| `domain` | list | ['CANONICAL ➝ Divination ➝ Celestial'] | Hierarchical |
| `place_discovery` | str | "" | Often empty |
| `place_composition` | str | "" | Often empty |

#### Archibab Files
**File Naming**: Single file `archibab.csv`
- No prefix, contains all 1,541 texts

**Columns** (10 columns, no index):
| Column | Type | Example | Notes |
|--------|------|---------|-------|
| `fragment_id` | str | "ARM 10 33" | Publication reference |
| `fragment_line_num` | int | 1 | Line number |
| `index_in_line` | int | 0 | Word position |
| `word_language` | str | "akk" | lowercase |
| `domain` | str | "lettre politique" | French, flat |
| `place_discovery` | str | "Mari" | Often populated |
| `place_composition` | str | "Ilan-ṣura" | Often populated |
| `value` | str | "a-na" | Raw transliteration |
| `clean_value` | str | "" | Often empty! |
| `lemma` | str | "ana" | Plain string |

---

### Key Differences to Harmonize

| Aspect | eBL | Archibab | ORACC (expected) | SEAL (expected) |
|--------|-----|----------|------------------|-----------------|
| **Prefix** | `EBL_` | None | `ORACC_` | `SEAL_` |
| **Language Case** | UPPERCASE | lowercase | lowercase | Default AKKADIAN |
| **Lemma Format** | List as string | Plain string | Plain string (cf) | Empty |
| **Domain Format** | Hierarchical list | French string | Project name | Genre from metadata |
| **clean_value** | Populated | Often empty | From `norm` field | Generate from `value` |
| **Fragment ID** | Museum number | Publication ref | P-number | SEAL/CDLI number |

---

### Proposed Unified Schema

For `data/processed/{source}/` files:

| Column | Type | Description | Standardization |
|--------|------|-------------|-----------------|
| `fragment_id` | str | Unique ID | Keep source-specific format |
| `source` | str | Data source | "ebl", "archibab", "oracc", "seal" |
| `fragment_line_num` | int | Line number | Keep as-is |
| `index_in_line` | int | Word position | Keep as-is |
| `word_language` | str | Language code | UPPERCASE standardized |
| `value` | str | Raw transliteration | Keep original |
| `clean_value` | str | Cleaned text | Generate if empty |
| `lemma` | str | Base form | Plain string, empty if N/A |
| `domain` | str | Genre/category | English, standardized |
| `place_discovery` | str | Find location | Keep or empty |
| `place_composition` | str | Origin location | Keep or empty |

---

### File Naming Convention Proposal

```
data/processed/
├── ebl/
│   └── EBL_{fragment_id}.csv      # Keep existing format
├── archibab/
│   └── ARCHIBAB_{fragment_id}.csv # Add prefix, split by fragment
├── oracc/
│   └── ORACC_{P-number}.csv       # P-number as ID
└── seal/
    └── SEAL_{seal_no}.csv         # SEAL number as ID
```

---

### Language Code Standardization

From ORACC language codes (Akk/data/lists/languages.json):

| ORACC Code | Standardized | Description |
|------------|--------------|-------------|
| `akk` | AKKADIAN | Generic Akkadian |
| `akk-x-oldbab` | OLD_BABYLONIAN | Old Babylonian |
| `akk-x-midbab` | MIDDLE_BABYLONIAN | Middle Babylonian |
| `akk-x-neobab` | NEO_BABYLONIAN | Neo-Babylonian |
| `akk-x-ltebab` | LATE_BABYLONIAN | Late Babylonian |
| `akk-x-stdbab` | STANDARD_BABYLONIAN | Standard/Literary |
| `akk-x-oldass` | OLD_ASSYRIAN | Old Assyrian |
| `akk-x-midass` | MIDDLE_ASSYRIAN | Middle Assyrian |
| `akk-x-neoass` | NEO_ASSYRIAN | Neo-Assyrian |
| `sux` | SUMERIAN | Sumerian |
| `sux-x-emesal` | EMESAL | Emesal dialect |

---

### Domain Standardization

Need to map different domain formats:

**eBL Domains** (hierarchical):
- `CANONICAL ➝ Divination ➝ Celestial`
- `CANONICAL ➝ Technical ➝ Medicine`

**Archibab Domains** (French):
- `lettre politique` → "Political Letter"
- `texte administratif` → "Administrative"
- `texte juridique` → "Legal"

**ORACC Domains** (from catalogue.json):
- Uses `genre` field in catalogue
- Examples: "Royal Inscription", "Letter", "Administrative"

**SEAL Domains** (from metadata):
- Genre categories: "Hymns", "Prayers", "Epics", "Incantations"

---

### Key Observations

1. **eBL clean_value is often same as value** - need actual cleaning
2. **Archibab clean_value is often EMPTY** - must generate
3. **Lemma formats differ** - eBL uses list strings, Archibab uses plain strings
4. **Fragment IDs are source-specific** - can't unify, keep with source prefix
5. **ORACC has richest metadata** - includes `norm`, `cf`, catalogue data
6. **SEAL is poorest** - no lemma, no norm, must generate clean_value

---

## Task 4: Research Papers Analysis

### Paper 1: "Filling the Gaps in Ancient Akkadian Texts" (EMNLP 2021)
**Authors**: Lazar et al.
**Source**: Akk repository code is from this paper

#### Key Findings for Our Project:

**Dataset Used:**
- ORACC corpus: ~10,000 texts, **1 million words**, 2.3 million signs
- This is the same source we plan to use

**Preprocessing Steps (Critical for our pipeline):**
1. Remove ALL editorial annotations:
   - Uncertainty marks (`?`)
   - Superscripts (determinatives like `{d}`, `{m}`)
   - Subscripts (sign indices like `₂`, `₃`)
2. Replace `x` characters (missing signs) with `[MASK]` tokens during inference
3. Focus on isolating ONLY the original transcribed text

**Model Insights:**
- Multilingual BERT finetuned on Akkadian beats monolingual model trained from scratch
- **Zero-shot mBERT > Monolingual BERT** (even without Akkadian training!)
- Related Semitic languages (Hebrew, Arabic) in pretraining help
- Adding English translations provided NO additional benefit

**Performance:**
- 89% Hit@5 for single token prediction
- Performance degrades sharply for multi-token gaps

**Relevance:**
- For next-token prediction, the preprocessing approach is directly applicable
- Confirms multilingual pretraining is valuable even for ancient languages

---

### Paper 2: "EvaCun 2025 Shared Task" (ALP 2025 @ NAACL)
**Authors**: Gordin, Sahala, Spencer, Klein
**This paper uses OUR EXACT DATA (eBL + Archibab)!**

#### Dataset Statistics (CRITICAL):

**Token Prediction Dataset:**
| Metric | Value |
|--------|-------|
| Total Fragments | 28,472 |
| Unique Values | 118,550 |
| Train Fragments | 22,777 |
| Test Fragments | 5,695 |
| Akkadian Words | 970,237 |
| Sumerian Words | 130,596 |
| Emesal Words | 33,237 |

**Lemmatization Dataset:**
| Metric | Value |
|--------|-------|
| Total Fragments | 10,214 |
| Akkadian Words | 377,000 |

**Genre Distribution (Token Prediction):**
- Canonical: 11,332
- Unclassified: 10,994
- Archival: 2,940
- Other: 2,321
- Monumental: 344

#### Preprocessing Applied:
1. **Removed fragmentary markers**: `...`, `[`, `]`, `x`, `X`, `?`
2. **Removed numbers**
3. **Cleaned editorial marks**: `<`, `#`
4. **Masked 20% of data** for token prediction task
5. **Kept only first lemma** when multiple interpretations exist

#### Key Insights:
- Token prediction is HARD: best model achieved only **21% accuracy**
- OOV (out-of-vocabulary) words nearly impossible to predict (3% accuracy)
- Function words easier than content words
- BabyLemmatizer baseline: 14% accuracy on token prediction

#### Evaluation Approach:
- Removed Roman numerals (homonym indicators) from eBL data
- Applied harmonization for lemma variants

---

### Paper 3: eBL Data Paper (JOHD 2024)
**Authors**: Cobanoglu et al.
**THIS DESCRIBES OUR eBL DATA SOURCE**

#### Dataset Details:
- **~25,000 fragments** with transliterations
- **350,000+ lines** of text
- Constantly growing (public API available)
- Data license: CC BY-NC-SA 4.0

#### Data Access:
- **Zenodo**: https://doi.org/10.5281/zenodo.10018951
- **GitHub**: https://github.com/ElectronicBabylonianLiterature/transliterated-fragments
- **Public API**: Available for latest data

#### Data Format:
- Single JSON file with fragment objects
- Each fragment has: id, description, collection, museum info, genre, script type
- Transliteration stored as "atf" property (ATF format)
- Can be parsed into JSON tree using eBL-ATF parser

#### Quality:
- Each transliteration reviewed by expert
- Edit history tracked
- Produced by Assyriologists (professional quality)

#### Collections Included:
- British Museum
- Penn Museum
- Yale Babylonian Collection
- Hilprecht Collection
- Vorderasiatisches Museum

---

### Key Takeaways for Our Pipeline

#### Preprocessing Decisions:

**Based on EMNLP 2021 paper:**
1. Remove editorial marks: `?`, `*`, `#`, `<`, `>`
2. Handle determinatives: `{d}`, `{m}`, `{ki}` - keep or remove depending on goal
3. Handle subscripts: `₂`, `₃`, etc. - normalize or remove
4. Handle brackets: `[...]` for missing text

**Based on EvaCun 2025:**
1. Remove fragmentary words entirely: those with `...`, `[`, `]`, `x`, `X`, `?`
2. Remove numbers
3. For LLM training: mask tokens (20% standard)

#### Data Quality Notes:
- eBL data is professional quality, reviewed
- Archibab has good metadata (place info, domain)
- ORACC has richest annotation (norm, cf fields)
- SEAL lacks lemmatization - must live without it

#### Expected Performance:
- Token prediction: ~20% accuracy is current SOTA
- Much better for function words than content words
- OOV words are nearly impossible

#### Model Architecture Suggestions:
- Consider multilingual pretraining approach
- BERT-style MLM appropriate for gap-filling
- GPT-style appropriate for next-token prediction
- Data augmentation (15x repetition with different masks) helps

---

## Task 5: Akk Repository Preprocessing Code Analysis

### Source: Akk/preprocessing/main_preprocess.py

This code provides a complete preprocessing pipeline we can adapt. Here's the detailed analysis:

### Certainty Levels (Enum)

The Akk repo tracks certainty levels for each word:

```python
class Certainty(Enum):
    SURE = 0                    # Complete, certain text
    FORGOTTEN_SIGN = 1          # Has `<` or `>` markers
    FIXED_BY_EDITOR = 2         # Has `*` (editorial corrections)
    HAS_DOUBTS = 3              # Has `?` (uncertain readings)
    BLURRED = 4                 # In `⸢...⸣` (damaged but readable)
    MISSING_BUT_COMPLETED = 5   # In `[...]` (restored by editor)
    MISSING = 6                 # Has `x` or `...` (truly missing)
```

### Special Characters Handling

**Superscripts (Determinatives):**
```python
SUPERSCRIPTS_TO_UNICODE_CHARS = {
    "{1}": "\U0001F600",  # Male name
    "{m}": "\U0001F600",  # Male name
    "{d}": "\U0001F601",  # Divine name
    "{f}": "\U0001F602",  # Female name
    "{MÍ}": "\U0001F602", # Female
    "{ki}": "\U0001F603", # Place name
    "{kur}": "\U0001F604", # Country/land
    "{giš}": "\U0001F605", # Wood/tree
    "{uru}": "\U0001F606", # City
}
```

**Subscripts:**
```python
SUBSCRIPTS_LIST = ["₀", "₁", "₂", "₃", "₄", "₅", "₆", "₇", "₈", "₉"]
```

### Key Preprocessing Functions

#### 1. `remove_curly_brackets(word, remove_superscripts=True)`
- Replaces superscripts with Unicode characters OR removes them
- Non-superscript curly content: `{DUG}` → `DUG.` (capitalize + dot)

#### 2. `remove_subscripts(word)`
- Removes all subscripts: `u₂` → `u`
- Uses regex: `[₀₁₂₃₄₅₆₇₈₉]` → ``

#### 3. `remove_squared_brackets(word, remove_missing)`
- Removes `[...]` (full squared brackets - restorations)
- Removes `⸢...⸣` (upper squared brackets - damaged)
- If `remove_missing=True` and word has `x` or `...`, returns empty string

#### 4. `preprocess_text_akkadian()` - Main Preprocessing
```python
def preprocess_text_akkadian(text, remove_hyphens, freq_dist,
                              remove_missing, remove_subs, remove_supers):
    for word in text.split():
        # 1. Replace + with -
        word = re.sub(r'\+', '-', word)

        # 2. Remove redundant parts
        word = _remove_redundant_parts(word)

        # 3. Determine certainty level
        certainty_level = certainty_hierarchy(word, ...)

        # 4. Replace 'x' with special missing character
        word = word.replace('x', MISSING_SIGN_CHAR)

        # 5. Remove intentional holes
        word = word.replace('o', '')

        # 6. Remove editorial marks
        word = re.sub('[?*<>]', '', word)

        # 7. Handle brackets
        word = remove_squared_brackets(word, remove_missing)

        # 8. Remove subscripts
        word = remove_subscripts(word)

        # 9. Handle pseudo-words for rare tokens
        if freq_dist and freq_dist[word] < 3:
            word = decide_pse_word(word)  # Replace with NAME/GOD/PLACE

        # 10. Remove superscripts (determinatives)
        word = remove_curly_brackets(word, remove_supers)

        # 11. Optionally remove hyphens
        if remove_hyphens:
            word = word.replace('-', '')
```

### Default Settings Used in Paper

From the code, the default preprocessing settings are:
```python
preprocess_akk_file(
    remove_missing=False,     # Keep words with missing parts
    remove_hyphens=False,     # Keep hyphens between syllables
    pseudo_words=False,       # Don't replace rare words
    remove_subs=True,         # Remove subscripts
    remove_supers=True,       # Remove superscripts
)
```

### Pseudo-Words System (pse_words.py)

Replaces rare proper nouns with generic placeholders:
- Personal names → `NAME`
- Divine names → `GOD`
- Place names → `PLACE`

This helps with vocabulary sparsity.

### Output Format

Each preprocessed entry contains:
```python
{
    "id_text": "P123456",
    "project_name": "saao",
    "provenience": "Nineveh",
    "genre": "Royal Inscription",
    "period": "Neo-Assyrian",
    "language": "akk",
    "url": "http://oracc.iaas.upenn.edu/saao/P123456/html",
    "raw_text": "original text",
    "preprocessed_words": [
        {
            "original": "i-par-ra-as",
            "preprocessed": "iparras",
            "certainty": "SURE"
        }
    ],
    "bert_input": "clean text for model"
}
```

### Train/Test Split

- 80/20 split
- Fixed random seed (SEED = 2) for reproducibility

---

### What We Can Reuse/Adapt

1. **Certainty tracking** - useful for filtering/weighting training data
2. **Superscript handling** - determinative removal logic
3. **Subscript removal** - standardize sign readings
4. **Bracket handling** - clean editorial marks
5. **Pseudo-word system** - handle rare names
6. **Output format** - JSONL with word-level annotations

### Recommended Preprocessing Pipeline for Our Project

Based on Akk code + EvaCun paper, here's the recommended approach:

```python
def clean_akkadian_word(word):
    """Standardized cleaning for all sources."""

    # 1. Remove editorial marks
    word = re.sub(r'[?*<>#]', '', word)

    # 2. Remove brackets
    word = re.sub(r'[\[\]⸢⸣]', '', word)

    # 3. Remove subscripts (normalize sign readings)
    word = re.sub(r'[₀₁₂₃₄₅₆₇₈₉ₓ]', '', word)

    # 4. Handle determinatives
    # Option A: Remove completely
    word = re.sub(r'\{[^}]+\}', '', word)
    # Option B: Keep as uppercase markers
    # word = re.sub(r'\{([^}]+)\}', lambda m: m.group(1).upper() + '.', word)

    # 5. Remove + signs (word boundaries)
    word = word.replace('+', '')

    # 6. Handle intentional holes
    word = word.replace('o', '')

    return word.strip()

def should_include_word(word):
    """Filter out fragmentary words."""
    # Remove words with missing parts
    if 'x' in word.lower():
        return False
    if '...' in word:
        return False
    if word.startswith('[') or word.endswith(']'):
        return False
    return True
```

---

## Task 6: Plan for 4 Processed Data Folders

### Directory Structure

```
v_1/data/
├── raw/                          # Existing raw data (READ-ONLY)
│   ├── extracted/
│   │   ├── full_corpus_dir/      # ~28k eBL CSV files
│   │   └── filtered_json_files/  # eBL JSON source
│   ├── archibab.csv              # Archibab data
│   └── genres.json               # eBL genre metadata
│
├── downloaded/                   # NEW: Downloaded from web
│   ├── oracc/                    # ORACC project ZIPs
│   │   ├── saao.zip
│   │   ├── rinap.zip
│   │   └── ...
│   └── seal/                     # SEAL scraped data
│       ├── metadata.csv          # From Collections download
│       └── texts/                # Scraped HTML/text files
│
├── processed/                    # NEW: Cleaned data by source
│   ├── ebl/
│   │   └── ebl_corpus.parquet    # Single file, all fragments
│   ├── archibab/
│   │   └── archibab_corpus.parquet
│   ├── oracc/
│   │   └── oracc_corpus.parquet
│   ├── seal/
│   │   └── seal_corpus.parquet
│   └── unified/                  # Combined dataset
│       ├── train.parquet
│       ├── val.parquet
│       ├── test.parquet
│       └── metadata.json         # Dataset statistics
│
└── vocab/                        # Tokenizer files
    ├── vocab.txt
    └── tokenizer_config.json
```

### Unified Schema for Processed Files

Each processed parquet file will have these columns:

| Column | Type | Description | Required |
|--------|------|-------------|----------|
| `source` | str | "ebl", "archibab", "oracc", "seal" | Yes |
| `fragment_id` | str | Source-specific unique ID | Yes |
| `line_num` | int | Line number on tablet | Yes |
| `word_idx` | int | Word position in line | Yes |
| `language` | str | AKKADIAN, SUMERIAN, etc. | Yes |
| `value_raw` | str | Original transliteration | Yes |
| `value_clean` | str | Cleaned for model input | Yes |
| `lemma` | str | Base form (empty if N/A) | No |
| `domain` | str | Genre category (English) | No |
| `certainty` | str | SURE, BLURRED, etc. | No |

### Processing Pipeline for Each Source

#### 1. eBL Processing
```
Input:  v_1/data/raw/extracted/full_corpus_dir/*.csv
Output: v_1/data/processed/ebl/ebl_corpus.parquet

Steps:
1. Iterate all CSV files
2. For each file:
   - Extract fragment_id from filename
   - Load CSV
   - Filter: keep only AKKADIAN words (or include SUMERIAN?)
   - Filter: remove fragmentary words (those with [, ], x, ?)
   - Clean value column → value_clean
   - Add source="ebl"
   - Add certainty based on original value
3. Concatenate all fragments
4. Save as parquet
```

#### 2. Archibab Processing
```
Input:  v_1/data/raw/archibab.csv
Output: v_1/data/processed/archibab/archibab_corpus.parquet

Steps:
1. Load single CSV
2. Standardize language: "akk" → "AKKADIAN"
3. Translate domains: French → English
4. Generate value_clean from value (currently empty)
5. Add source="archibab"
6. Filter fragmentary words
7. Save as parquet
```

#### 3. ORACC Processing
```
Input:  v_1/data/downloaded/oracc/*.zip
Output: v_1/data/processed/oracc/oracc_corpus.parquet

Steps:
1. For each project ZIP:
   - Extract corpusjson/*.json files
   - Load catalogue.json for metadata
2. For each text JSON:
   - Recursive traverse CDL tree
   - Extract f nodes (words)
   - Map: form→value_raw, norm→value_clean, cf→lemma, lang→language
   - Get line_num from parent l node
   - Filter Akkadian-dominant texts (>50% Akkadian)
3. Join with catalogue metadata (genre, provenience)
4. Add source="oracc"
5. Concatenate all texts
6. Save as parquet
```

#### 4. SEAL Processing
```
Input:  v_1/data/downloaded/seal/metadata.csv + texts/*.html
Output: v_1/data/processed/seal/seal_corpus.parquet

Steps:
1. Load metadata CSV (SEAL no, genre, provenance)
2. For each text:
   - Scrape HTML from https://seal.huji.ac.il/node/{SEAL_NO}
   - Parse transliteration block
   - Split into lines and tokens
   - Generate value_clean from value_raw
3. Join with metadata
4. Add source="seal", language="AKKADIAN"
5. Concatenate all texts
6. Save as parquet
```

### Unified Dataset Creation

```
Input:  v_1/data/processed/{ebl,archibab,oracc,seal}/*.parquet
Output: v_1/data/processed/unified/{train,val,test}.parquet

Steps:
1. Load all source parquets
2. Concatenate into single DataFrame
3. Deduplication:
   - If same fragment_id appears in multiple sources, keep richest version
   - Priority: eBL > ORACC > Archibab > SEAL
4. Split by fragment (not by word!) to prevent leakage:
   - Train: 80%
   - Val: 10%
   - Test: 10%
5. Save splits as parquet
6. Generate metadata.json with statistics
```

---

## Task 7: Dataset Class Design for LLM Training

### Design Goals

1. **Fragment-Level Organization**: Each fragment is a training unit
2. **Text Reconstruction**: Join words back into readable text
3. **Flexible Tokenization**: Support word-level or subword
4. **HuggingFace Compatible**: Easy integration with transformers

### Parquet to Text Reconstruction

```python
def reconstruct_text(fragment_df: pd.DataFrame) -> str:
    """Convert tabular data back to text."""
    # Sort by line and position
    fragment_df = fragment_df.sort_values(['line_num', 'word_idx'])

    lines = []
    current_line = None
    current_words = []

    for _, row in fragment_df.iterrows():
        if row['line_num'] != current_line:
            if current_words:
                lines.append(' '.join(current_words))
            current_line = row['line_num']
            current_words = []
        current_words.append(row['value_clean'])

    if current_words:
        lines.append(' '.join(current_words))

    return ' '.join(lines)  # or '\n'.join(lines) for line-aware
```

### PyTorch Dataset Class

```python
import torch
from torch.utils.data import Dataset
import pandas as pd
from transformers import PreTrainedTokenizer

class AkkadianDataset(Dataset):
    """Dataset for Akkadian next-token prediction."""

    def __init__(
        self,
        parquet_path: str,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
        use_clean_value: bool = True
    ):
        self.df = pd.read_parquet(parquet_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.value_col = 'value_clean' if use_clean_value else 'value_raw'

        # Group by fragment
        self.fragments = self.df.groupby('fragment_id')
        self.fragment_ids = list(self.fragments.groups.keys())

    def __len__(self):
        return len(self.fragment_ids)

    def __getitem__(self, idx):
        fragment_id = self.fragment_ids[idx]
        fragment_df = self.fragments.get_group(fragment_id)

        # Reconstruct text
        text = self._reconstruct_text(fragment_df)

        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': encoding['input_ids'].squeeze(),  # For causal LM
            'fragment_id': fragment_id
        }

    def _reconstruct_text(self, fragment_df):
        fragment_df = fragment_df.sort_values(['line_num', 'word_idx'])
        words = fragment_df[self.value_col].tolist()
        return ' '.join(words)
```

### HuggingFace Datasets Integration

```python
from datasets import load_dataset, Dataset as HFDataset

def load_akkadian_dataset(data_dir: str):
    """Load dataset using HuggingFace datasets library."""

    # Load from parquet files
    dataset = load_dataset(
        'parquet',
        data_files={
            'train': f'{data_dir}/train.parquet',
            'validation': f'{data_dir}/val.parquet',
            'test': f'{data_dir}/test.parquet'
        }
    )

    return dataset

def prepare_for_training(examples, tokenizer, max_length=512):
    """Map function for tokenization."""

    # Group examples by fragment and reconstruct text
    # This is called in batches

    texts = []
    for fragment_id in examples['fragment_id']:
        # Get all rows for this fragment
        mask = [fid == fragment_id for fid in examples['fragment_id']]
        words = [examples['value_clean'][i] for i, m in enumerate(mask) if m]
        texts.append(' '.join(words))

    # Tokenize
    return tokenizer(
        texts,
        truncation=True,
        max_length=max_length,
        padding='max_length'
    )
```

### Alternative: Pre-Reconstructed Text Format

For simpler loading, we can also save reconstructed texts:

```python
# Create text-level dataset (one row per fragment)
def create_text_dataset(parquet_path, output_path):
    df = pd.read_parquet(parquet_path)

    texts = []
    for fragment_id, group in df.groupby('fragment_id'):
        group = group.sort_values(['line_num', 'word_idx'])
        text = ' '.join(group['value_clean'].tolist())

        texts.append({
            'fragment_id': fragment_id,
            'source': group['source'].iloc[0],
            'domain': group['domain'].iloc[0] if 'domain' in group else '',
            'text': text,
            'num_words': len(group)
        })

    text_df = pd.DataFrame(texts)
    text_df.to_parquet(output_path)
```

This creates a simpler format:
```
fragment_id | source | domain | text | num_words
------------|--------|--------|------|----------
P123456     | oracc  | Royal  | a-na be-li... | 45
```

### Recommended Approach

1. **Store processed data** in word-level parquet (preserves all metadata)
2. **Create text-level view** for training (faster loading)
3. **Use HuggingFace datasets** for training loop
4. **Train custom tokenizer** on the corpus first

### Tokenizer Training

```python
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

def train_akkadian_tokenizer(text_file, vocab_size=8000):
    """Train BPE tokenizer on Akkadian corpus."""

    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    )

    tokenizer.train([text_file], trainer)

    return tokenizer
```

---

## Summary: Implementation Roadmap

### Phase 1: Data Download (Manual + Scripted)
1. Download ORACC project ZIPs using existing Akk code
2. Manually download SEAL metadata CSV
3. Write SEAL scraping script (with polite delays)

### Phase 2: Data Processing
1. Create processing scripts for each source
2. Generate parquet files with unified schema
3. Create unified dataset with train/val/test splits

### Phase 3: Tokenization
1. Extract all text from unified dataset
2. Train BPE tokenizer on combined corpus
3. Save tokenizer for model training

### Phase 4: Dataset Class
1. Implement PyTorch Dataset class
2. Test with small batch
3. Integrate with training loop

### Estimated Data Volume
| Source | Fragments | Words (est.) |
|--------|-----------|--------------|
| eBL | ~28,000 | ~1,000,000 |
| Archibab | ~1,500 | ~50,000 |
| ORACC | ~10,000 | ~1,000,000 |
| SEAL | ~900 | ~50,000 |
| **Total** | **~40,000** | **~2,100,000** |

This is still a low-resource setting but significantly larger than previous work!

---

## CRITICAL: Tokenization Analysis & Decision

### Current State of Data

**Important Discovery**: The existing data (eBL and Archibab) is **NOT at the syllable level** - it's at the **WORD level with syllable markers**:

**Archibab examples:**
- `a-na` (syllables joined by hyphens)
- `be-lí` (syllables with hyphens)
- `ka-ka-bi` (multi-syllable word)

**eBL examples:**
- `BE-ma` (syllables with hyphens)
- `ŠEŠ-šu₂` (mix of logograms and syllables)
- `U₄.15.KAM₂` (logograms with dots)

### What the Papers Used

#### 1. EMNLP 2021 (Akk Repository)
**Tokenizer**: `BertWordPieceTokenizer`
- **Pre-tokenizer**: `Whitespace()` - splits on whitespace first
- **Algorithm**: WordPiece (like BERT)
- **Vocab size**: 2,000 for mini BERT, standard for mBERT
- **Min frequency**: 2

**Key Code:**
```python
tokenizer = BertWordPieceTokenizer(lowercase=False)
tokenizer.pre_tokenizer = Whitespace()
tokenizer.train(
    files=[bert_input_file],
    vocab_size=vocab_size,
    min_frequency=2,
)
```

**Process:**
1. Whitespace splits text into tokens: `"a-na be-lí"` → `["a-na", "be-lí"]`
2. WordPiece learns subword units from these tokens
3. Final vocabulary includes: whole words, syllables, character sequences

#### 2. EvaCun 2025 (BabyLemmatizer)
**Tokenizer**: "logo-syllabic tokenizer"
- Treats **logograms** as indivisible (e.g., `ŠEŠ`, `AN.GE₆`)
- Treats **syllabograms** as divisible phoneme sequences
- Uses transliterated signs as minimal units

**Quote from paper:**
> "Both models use the default logo-phonemic tokenization that treats logograms and determinatives as indivisible symbols, and syllabograms and phonetic complements as divisible phoneme sequences."

### The Problem

We have **TWO different representations** in our data:

1. **Syllabic transliteration** (Archibab, most Akkadian):
   - `a-na` = syllables joined by hyphens
   - Each hyphen-separated part is a syllable
   - This is the "phonetic spelling"

2. **Logographic + syllabic** (eBL, some ORACC):
   - `ŠEŠ-šu₂` = logogram + syllabic suffix
   - `AN.GE₆` = multiple logograms
   - Mixed representation

### What We Should Do

#### Option A: **Sign-level Tokenization** (Recommended)
Split on hyphens AND dots to get individual signs:

```python
def tokenize_akkadian_sign_level(text):
    """Split into individual cuneiform signs."""
    # Replace dots and hyphens with spaces
    text = text.replace('.', ' ').replace('-', ' ')
    return text.split()

# Examples:
"a-na be-lí" → ["a", "na", "be", "lí"]
"AN.GE₆" → ["AN", "GE₆"]
"ŠEŠ-šu₂" → ["ŠEŠ", "šu₂"]
```

**Then apply WordPiece on top** for rare signs/sequences.

#### Option B: **Word-level with Subword Tokenization** (What Akk used)
Keep as-is, let WordPiece learn syllable patterns:

```python
# Whitespace split first
"a-na be-lí ka-ka-bi" → ["a-na", "be-lí", "ka-ka-bi"]

# WordPiece learns:
# - Common words: "a-na" (whole)
# - Syllables: "ka", "bi"
# - Mixed: "be-lí" or ["be", "-lí"]
```

#### Option C: **Normalize to Sign-Level First**
Most linguistically accurate:

```python
def normalize_to_signs(word):
    """Convert to individual signs, preserve structure."""
    # Keep original for metadata
    original = word

    # Split on delimiters
    signs = word.replace('-', ' ').replace('.', ' ').split()

    return ' '.join(signs)

# Full text:
"a-na be-lí" → "a na be lí"
"ŠEŠ-šu₂ GU₇" → "ŠEŠ šu₂ GU₇"
```

### Recommendation

**Use Option A (Sign-level) + WordPiece:**

1. **Preprocessing**: Split all words into individual signs
   - `a-na` → `a na`
   - `ŠEŠ-šu₂` → `ŠEŠ šu₂`
   - `AN.GE₆` → `AN GE₆`

2. **Store both versions**:
   - `value_raw`: Original (`a-na`)
   - `value_signs`: Sign-level (`a na`)
   - `value_clean`: Cleaned signs for model

3. **Train BPE/WordPiece tokenizer** on sign-level text
   - Vocabulary will include individual signs
   - Can learn common sign combinations
   - Better generalization to unseen words

### Why This Matters

**Current data:**
- 2.1M **words** (hyphenated units like `a-na`)
- But actually **~5-6M signs** when split by hyphens
- Each "word" averages 2-3 signs

**Impact on model:**
- More training tokens = better learning
- Sign-level = more consistent units across sources
- Matches how cuneiform actually works (sign-by-sign)

### Updated Processing Pipeline

```python
def process_akkadian_value(value_raw):
    """
    Convert raw transliteration to sign-level representation.

    Args:
        value_raw: Original word (e.g., "a-na", "ŠEŠ-šu₂")

    Returns:
        dict with:
            - value_raw: Original
            - value_signs: Space-separated signs
            - value_clean: Cleaned for model
    """
    # 1. Remove editorial marks
    clean = re.sub(r'[?*<>#\[\]⸢⸣]', '', value_raw)

    # 2. Remove subscripts (optional - normalize sign readings)
    clean = re.sub(r'[₀₁₂₃₄₅₆₇₈₉ₓ]', '', clean)

    # 3. Remove determinatives
    clean = re.sub(r'\{[^}]+\}', '', clean)

    # 4. Split into signs (on hyphens and dots)
    signs = clean.replace('-', ' ').replace('.', ' ').replace('+', ' ')

    # 5. Clean up whitespace
    signs = ' '.join(signs.split())

    return {
        'value_raw': value_raw,
        'value_signs': signs,
        'value_clean': signs.lower()  # or keep case?
    }

# Examples:
process_akkadian_value("a-na")
# → {'value_raw': 'a-na', 'value_signs': 'a na', 'value_clean': 'a na'}

process_akkadian_value("ŠEŠ-šu₂")
# → {'value_raw': 'ŠEŠ-šu₂', 'value_signs': 'ŠEŠ šu', 'value_clean': 'šeš šu'}
```

### For ORACC and SEAL

**ORACC**: Already provides sign-level in CDL structure
- Extract `form` field from `f` nodes
- May need same splitting if hyphens present

**SEAL**: Need to apply same splitting after scraping
- HTML will have hyphenated words
- Apply same sign-splitting logic

### Final Unified Schema Update

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `value_raw` | str | Original transliteration | `"a-na"` |
| `value_signs` | str | Space-separated signs | `"a na"` |
| `value_clean` | str | Cleaned signs for model | `"a na"` |

### Training Flow

```
1. Process all sources → parquet with value_signs column
2. Reconstruct text from value_signs (space-joined)
3. Train WordPiece tokenizer on sign-level corpus
4. Tokenizer learns:
   - Individual signs: "a", "na", "ŠEŠ"
   - Common combinations: "a na" (might stay as one)
   - Character sequences for rare signs
5. Train LLM on tokenized sign sequences
```

This gives us **~5-6M training tokens** instead of 2.1M words!

---

## Answer: EvaCun 2025 Tokenization Explained

### What the Paper Says (Direct Quotes)

**From Section 6.2.1 (Lemmatizer Model):**
> "Both models use the default **logo-phonemic tokenization** that treats **logograms and determinatives as indivisible symbols**, and **syllabograms and phonetic complements as divisible phoneme sequences**. This setting **collapses homonymous syllabic signs such as ša and ša₂ together** but keeps logograms such as DU and DU₃ separate, since their meanings and readings are generally unrelated."

**From Section 6.2.2 (Token Predictor Model):**
> "We segment the input using BabyLemmatizer's **logo-syllabic tokenizer** using **transliterated signs as minimal units**, and generate the output sequence similarly."

### What This Actually Means

**Their tokenization splits on hyphens and dots to get individual signs:**

1. **Syllabograms** (phonetic signs like `a`, `na`, `be`) → **SPLIT on hyphens**
   - `a-na` → `["a", "na"]`
   - `be-lí` → `["be", "lí"]`

2. **Logograms** (word signs like `ŠEŠ`, `AN`) → **Keep as single units**
   - `ŠEŠ-šu₂` → `["ŠEŠ", "šu"]` (logogram + syllable)
   - `AN.GE₆` → `["AN", "GE₆"]` (SPLIT on dots too!)

3. **Subscripts** (like `₂`, `₃`) → **REMOVED/NORMALIZED**
   - Collapses `ša` and `ša₂` to the same token
   - This is the "collapses homonymous syllabic signs" part

### Your Data is ALREADY at the Right Level!

**Critical realization:** Your eBL and Archibab data is **NOT "already tokenized"** in the NLP sense - it's at the **WORD level** where each row is one cuneiform word (which may contain multiple signs joined by hyphens/dots).

**Current structure:**
```
Row 1: value = "a-na"       # One WORD (2 signs)
Row 2: value = "be-lí"      # One WORD (2 signs)
Row 3: value = "ŠEŠ-šu₂"    # One WORD (logogram + syllable)
```

**What EvaCun did:**
Split each `value` field into individual signs by treating hyphens and dots as delimiters.

### Can You Use This for All 4 Datasets?

**YES! ✓** This tokenization works perfectly for all your sources:

| Source | Current Format | Split on Hyphens/Dots | Result |
|--------|----------------|----------------------|--------|
| **Archibab** | `a-na` | `a-na` → `a na` | ✓ Works |
| **eBL** | `BE-ma`, `AN.GE₆` | `BE-ma` → `BE ma`<br>`AN.GE₆` → `AN GE₆` | ✓ Works |
| **ORACC** | Similar to eBL | Same splitting | ✓ Works |
| **SEAL** | Will be like Archibab | Same splitting | ✓ Works |

### Exact Implementation for Your Data

```python
def evaCun_tokenization(value_raw):
    """
    Apply EvaCun 2025 logo-syllabic tokenization.

    Based on paper (Section 6.2.2):
    - Treats logograms as indivisible
    - Treats syllabograms as divisible
    - Uses transliterated signs as minimal units
    - Collapses subscripted variants (ša₂ → ša)
    """

    # 1. Remove editorial marks (not part of the text)
    clean = re.sub(r'[?*<>#\[\]⸢⸣]', '', value_raw)

    # 2. Remove determinatives (semantic classifiers in {})
    clean = re.sub(r'\{[^}]+\}', '', clean)

    # 3. NORMALIZE subscripts (collapse ša₂ → ša)
    # This is the "collapses homonymous syllabic signs" part
    clean = re.sub(r'[₀₁₂₃₄₅₆₇₈₉ₓ]', '', clean)

    # 4. SPLIT on hyphens and dots to get individual signs
    # This is the "transliterated signs as minimal units" part
    signs = clean.replace('-', ' ').replace('.', ' ').replace('+', ' ')

    # 5. Clean up whitespace
    signs = ' '.join(signs.split())

    return signs

# Examples matching your data:
evaCun_tokenization("a-na")
# → "a na"

evaCun_tokenization("be-lí")
# → "be lí"

evaCun_tokenization("BE-ma")
# → "BE ma"

evaCun_tokenization("ŠEŠ-šu₂")
# → "ŠEŠ šu"  (subscript removed, split on hyphen)

evaCun_tokenization("AN.GE₆")
# → "AN GE"  (subscript removed, split on dot)
```

### Why This Works for All Your Datasets

1. **Archibab**: `a-na` (syllabic) → `a na` ✓
2. **eBL**: `BE-ma` (syllabic), `AN.GE₆` (logographic) → `BE ma`, `AN GE` ✓
3. **ORACC**: Uses same conventions as eBL → Same splitting ✓
4. **SEAL**: Will scrape hyphenated words → Same splitting ✓

### Updated Schema with EvaCun Tokenization

| Column | Description | Example |
|--------|-------------|---------|
| `value_raw` | Original from CSV | `"ŠEŠ-šu₂"` |
| `value_signs` | **EvaCun tokenized** | `"ŠEŠ šu"` |
| `value_clean` | Lowercase normalized | `"šeš šu"` |

### Training Flow with EvaCun Tokenization

```
1. Process each source with evaCun_tokenization()
   - Archibab: "a-na" → "a na"
   - eBL: "BE-ma" → "BE ma"
   - ORACC: Similar
   - SEAL: Similar

2. Store in parquet with value_signs column

3. Reconstruct text by joining value_signs:
   Fragment text = "a na be lí šeš šu ..."

4. Train WordPiece/BPE on this sign-level text
   - Learns individual signs: "a", "na", "ŠEŠ", "šu"
   - Can merge common sequences: "a na" might become one token

5. Train LLM on tokenized sequences
```

### Key Difference from What You Thought

**You thought:** eBL and Archibab are "already broken into tokens"

**Reality:** They're at the **word level**, where each row = one cuneiform word. The `value` field contains multiple signs joined by hyphens/dots.

**What EvaCun did:** Split those hyphenated/dotted words into individual signs.

### Why Not Just Use WordPiece on Hyphenated Words?

**You could**, but sign-level is better because:

1. **More training data**: 2.1M words → 5-6M signs (2.5x more!)
2. **Consistent across sources**: All use same sign-level units
3. **Matches paper methodology**: Directly comparable to EvaCun 2025
4. **Linguistically correct**: Cuneiform is written sign-by-sign
5. **Better generalization**: Model learns sign patterns, can handle new word combinations

### Final Recommendation

**Use EvaCun's logo-syllabic tokenization for ALL sources:**

```python
# For all sources (eBL, Archibab, ORACC, SEAL):
for idx, row in df.iterrows():
    value_raw = row['value']
    value_signs = evaCun_tokenization(value_raw)
    # Store both in processed data
```

This gives you:
- **Direct comparability** with EvaCun 2025 paper
- **Unified tokenization** across all 4 sources
- **~5-6M training tokens** instead of 2.1M words
- **Same methodology** as current SOTA baseline

---

## CRITICAL QUESTION: Should We Also Split on Whitespace?

### The Question

Your question: **"what is the tradeoff of break it also on the white space?"**

This asks whether we should **preserve word boundaries** or **flatten everything to pure sign sequences**.

### Three Options

Let me show you the exact difference using real Archibab data (lines 1-3 of ARM 10 33):

```
Row 1: fragment_line_num=1, index_in_line=0, value="a-na"
Row 2: fragment_line_num=1, index_in_line=1, value="be-lí"
Row 3: fragment_line_num=1, index_in_line=2, value="ù"
Row 4: fragment_line_num=1, index_in_line=3, value="ka-ka-bi"
```

After `evaCun_tokenization()` on each row:
```
"a-na" → "a na"
"be-lí" → "be lí"
"ù" → "ù"
"ka-ka-bi" → "ka ka bi"
```

#### Option A: Preserve Word Boundaries (Current EvaCun Approach)

**How it works:**
- Each row = one word
- Split hyphens/dots WITHIN each word
- Join rows with spaces to reconstruct lines

**Reconstruction of line 1:**
```python
line_text = " ".join([row['value_signs'] for row in line_1_rows])
# Result: "a na be lí ù ka ka bi"
```

**What the model sees:**
```
Sequence: "a na be lí ù ka ka bi"
```

**Problem:** The space between `na` and `be` looks IDENTICAL to the space between `a` and `na`, but they mean different things:
- `a na` = two signs in ONE word (ana = "to")
- `na be` = last sign of one word + first sign of next word (ana bêlum = "to the lord")

**The model cannot distinguish word boundaries from sign boundaries!**

#### Option B: Use Special Word Boundary Marker

**How it works:**
- Same as Option A, but use `|` or `<WORD>` to mark word boundaries

**Reconstruction:**
```python
line_text = " | ".join([row['value_signs'] for row in line_1_rows])
# Result: "a na | be lí | ù | ka ka bi"
```

**What the model sees:**
```
Sequence: "a na | be lí | ù | ka ka bi"
```

**Now the model knows:**
- Space = sign boundary within a word
- `|` = word boundary

#### Option C: Flatten Completely (Remove All Word Structure)

**How it works:**
- Concatenate ALL signs from entire fragment
- No word boundary information at all

**Reconstruction:**
```python
all_signs = []
for row in fragment_rows:
    all_signs.extend(row['value_signs'].split())
fragment_text = " ".join(all_signs)
# Result: "a na be lí ù ka ka bi qí bí ma um ma ..."
```

**What the model sees:**
```
Sequence: "a na be lí ù ka ka bi qí bí ma um ma ..."
```

**The model has NO idea where words end.**

### The Tradeoffs

#### Linguistic Considerations

**Does Akkadian word segmentation matter?**

YES, word boundaries are linguistically meaningful:

1. **Grammatical structure**: Akkadian is a **fusional language** where morphemes attach to word stems
   - Example: `be-lí-ia` = "my lord" (bêl = lord, -ī = my)
   - The word boundary matters because `-ia` is a possessive suffix that attaches to a WORD

2. **Morphological patterns**: Akkadian verbs have complex internal structure
   - Example: `i-ta-ar-ra-an-ni` = "he will return to me"
   - This is ONE word with infixes and suffixes, not separate signs

3. **Logograms vs syllables**: Some logograms are ENTIRE words
   - Example: `DUMU-MUNUS-ka-a-ma` (row 9 in Archibab)
   - `DUMU-MUNUS` = logogram for "daughter", `-ka` = "your", `-ma` = emphatic particle
   - These attach to a WORD, not random signs

**Conclusion:** Word boundaries provide important grammatical signal.

#### Model Learning Implications

| Aspect | Option A: No Marker | Option B: With Marker | Option C: Flatten |
|--------|---------------------|----------------------|-------------------|
| **Vocabulary size** | Smallest (~5-6k signs) | Medium (~5-6k signs + 1 marker) | Smallest (~5-6k signs) |
| **Training data** | ~5-6M signs | ~5-6M signs + ~2.1M markers | ~5-6M signs |
| **What model learns** | Sign sequences ignoring word structure | Sign sequences + word boundaries | Pure sign sequences |
| **Grammatical patterns** | Hard to learn (confusing signal) | Easier (word structure preserved) | Very hard (no word info) |
| **Next-sign prediction** | Ambiguous context window | Clear context window | Pure sequential |
| **Matches EvaCun 2025?** | Likely YES (they use CSV rows) | NO (they don't mention markers) | NO (they preserve rows) |

#### What Did EvaCun 2025 Actually Do?

Let me check what the paper ACTUALLY says (Section 6.2.2):

**Direct quote:**
> "We segment the input using BabyLemmatizer's logo-syllabic tokenizer using **transliterated signs as minimal units**, and generate the output sequence similarly."

**Critical detail from Section 6.2.1:**
> "Both models use the default logo-phonemic tokenization that treats **logograms and determinatives as indivisible symbols**, and **syllabograms and phonetic complements as divisible phoneme sequences**. This setting **collapses homonymous syllabic signs such as ša and ša₂ together** but keeps logograms such as DU and DU₃ separate."

**Key from 6.2.2:**
> "The token prediction model... relies purely on **sign-to-sign relations**."

**What this tells us:**

1. They use **sign-level tokenization** (not word-level)
2. They split syllabograms: `a-na` → splits into individual signs
3. They keep logograms whole: `ŠEŠ` stays as one unit
4. They normalize subscripts: `ša₂` → `ša` (for syllables only)
5. The model predicts based on "**sign-to-sign relations**"

**What they DON'T say:**
- They don't mention preserving word boundaries
- They don't mention any special word boundary markers
- They don't describe how they join signs from different words

**Most likely interpretation:** They used **Option A or Option C** - the paper doesn't specify whether they preserve the original word boundaries from the CSV structure when creating training sequences.

**The ambiguity in the paper:** When they say "symmetric window of three words" for context, but tokenize at sign level - this suggests they might track word boundaries internally but we can't tell from the paper text how this is represented in the actual training data.

**CRITICAL INSIGHT from "symmetric window of three words":**

The paper says (Section 6.2.2):
> "predict transliteration for each masked **word** based on its surrounding context with a **symmetric window of three words**"

This means they ARE tracking word boundaries! Otherwise they couldn't define a "window of three words."

**Two possibilities:**

1. **They preserve word-level structure in data** (likely Option A):
   - The CSV has one row per word
   - They tokenize each word to signs internally
   - They keep track of which signs belong to which word
   - Context window counts WORDS, not signs
   - Example: To predict word 4, they look at words 1-3 and 5-7

2. **They use word boundary markers** (like Option B):
   - Less likely, since not mentioned

**Most probable answer: They use Option A** - preserve the word-level CSV structure, tokenize each word internally to signs, but maintain word boundaries in the data structure so they can define "three word" context windows.

### My Recommendation

Based on the paper, EvaCun 2025 **definitely tracks word boundaries** (otherwise they couldn't use "three word" context windows), but the paper doesn't explicitly say HOW they represent this.

**You have three choices:**

**Option A: Match EvaCun exactly** (preserve word-level CSV rows)
- ✅ Directly comparable to their results
- ✅ Natural: each CSV row is one word
- ✅ Can track word context windows
- ❌ When converting to text for LLM training, word boundaries become ambiguous
- **Use this if**: You want exact replication of EvaCun methodology

**Option B: Improve on EvaCun** (add explicit word boundary markers)
- ✅ Preserves linguistic structure without ambiguity
- ✅ Better for pure next-sign prediction (model knows word vs sign boundaries)
- ✅ Minimal overhead (one extra token)
- ❌ Not directly comparable to EvaCun results
- **Use this if**: You want to potentially improve on their approach

**Option C: Flatten completely** (remove word boundaries)
- ❌ Loses grammatical information
- ❌ Doesn't match EvaCun at all
- ❌ Can't use word-level context windows
- **Don't use this** - it's worse than both other options

**Implementation:**

```python
def reconstruct_fragment_text(fragment_df):
    """
    Reconstruct fragment text from processed rows.
    Each row = one word's signs.
    """
    lines = fragment_df.groupby('fragment_line_num')

    fragment_text = []
    for line_num, line_rows in lines:
        # Join signs within each word with space
        # Join words with " | "
        line_text = " | ".join(line_rows['value_signs'].tolist())
        fragment_text.append(line_text)

    # Join lines with newline or special <LINE> token
    return "\n".join(fragment_text)

# Example output:
# Line 1: "a na | be lí | ù | ka ka bi"
# Line 2: "qí bí ma"
# Line 3: "um ma | ki ru ú ma"
```

### Final Answer to Your Question

**You asked: "what is the tradeoff of break it also on the white space?"**

Based on the EvaCun 2025 paper, here's what we know:

**What they DO:**
1. Tokenize at sign level (split hyphens/dots within words)
2. Track word boundaries (they use "three word" context windows)
3. Rely on "sign-to-sign relations" for prediction

**What they DON'T specify:**
- How word boundaries are represented in the actual training sequences
- Whether they use special markers or just track this in metadata

**Your three options:**

**Option A - Keep word structure like EvaCun (most likely what they did):**
- Each CSV row = one word
- Tokenize internally: `"a-na"` → `["a", "na"]`
- Track which signs belong to which word in your data structure
- For training: join signs with spaces, but maintain word metadata
- **Tradeoff**: Preserves word info in structure, but if you flatten to pure text for LLM training, boundaries become ambiguous

**Option B - Add explicit word boundary marker (improvement on EvaCun):**
- Same as Option A, but add `|` token between words: `"a na | be lí"`
- **Tradeoff**: Makes boundaries explicit, adds one token to vocab, not directly comparable to EvaCun

**Option C - Flatten completely (ignore word boundaries):**
- Treat entire fragment as one sign sequence
- **Tradeoff**: Simplest, but loses grammatical structure that Akkadian needs

**My recommendation**: Start with **Option A** (match EvaCun) since you want to compare to their baseline. Your data structure will naturally preserve word boundaries since each CSV row is one word. Later, you can experiment with **Option B** to see if explicit markers improve next-token prediction.

**The key insight**: Don't flatten completely (Option C). Akkadian word structure matters grammatically, and EvaCun clearly tracks it even while tokenizing at sign level.

---

## IMPLEMENTATION PHASE

### Task 8: Data Download and Processing Scripts

#### Status: IN PROGRESS

**Goal**: Download all 4 data sources and create processing scripts to convert them to unified parquet format.

#### Subtask 8.1: Download ORACC Project ZIPs

**Status**: ✅ SCRIPT CREATED, RUNNING

**Location**: `yarin/scripts/01_download_oracc.py`

**What it does**:
1. Fetches list of all public ORACC projects from `http://oracc.museum.upenn.edu/projects.json`
2. Downloads each project as a ZIP file from `http://oracc.org/{project_name}/json`
3. Extracts ZIPs to `v_1/data/downloaded/oracc/`
4. Reports success/failure statistics

**Output directory structure**:
```
v_1/data/downloaded/oracc/
├── {project1}/
│   ├── catalogue.json
│   └── corpusjson/
│       ├── P123456.json
│       ├── P123457.json
│       └── ...
├── {project2}/
│   └── ...
```

**Running**: Download in progress (background process)

---

#### Subtask 8.2: Download SEAL Metadata CSV

**Status**: PENDING

**Action required**: Manual download from https://seal.huji.ac.il/collections

**Steps**:
1. Navigate to https://seal.huji.ac.il/collections
2. Click "Download CSV" button
3. Save to `v_1/data/downloaded/seal/metadata.csv`

---

#### Subtask 8.3: Process eBL Data

**Status**: ✅ SCRIPT CREATED

**Location**: `yarin/scripts/02_process_ebl.py`

**Input**: `v_1/data/raw/extracted/full_corpus_dir/*.csv` (~28,000 files)

**Output**: `v_1/data/processed/ebl/ebl_corpus.parquet`

**What the script does**:
1. Reads all CSV files in full_corpus_dir
2. For each file:
   - Extracts fragment_id from filename (removes "EBL_" prefix)
   - Filters out fragmentary words (containing x, [...], etc.)
   - Keeps only AKKADIAN language words
   - Determines certainty level based on editorial marks
3. Combines all fragments into single parquet
4. Adds source="ebl" column

**Implemented schema**:
| Column | Type | Description |
|--------|------|-------------|
| source | str | "ebl" |
| fragment_id | str | Fragment identifier |
| line_num | int | Line number |
| word_idx | int | Word position in line |
| language | str | "AKKADIAN", "SUMERIAN", etc. |
| value_raw | str | Original transliteration |
| value_clean | str | Clean value from CSV |
| lemma | str | Base form |
| domain | str | Genre |
| place_discovery | str | Archaeological site |
| place_composition | str | Original location |
| certainty | str | SURE, BLURRED, MISSING, etc. |

**NOTE**: `value_signs` column (sign-level tokenization) is NOT YET IMPLEMENTED - waiting for tokenization decision from paper author.

**To run** (after installing dependencies):
```bash
# Test on 10 files
python scripts/02_process_ebl.py --limit 10

# Full processing
python scripts/02_process_ebl.py
```

---

#### Subtask 8.4: Dependencies and Setup

**Status**: ✅ REQUIREMENTS FILE CREATED

**Location**: `yarin/requirements.txt`

**Dependencies**:
- pandas, pyarrow: Data processing and parquet support
- requests, urllib3, beautifulsoup4: Web scraping
- tqdm: Progress bars

**To install**:
```bash
pip3 install -r requirements.txt
```

---

## ✅ DATA PREPARATION COMPLETE (December 28, 2025)

### All Decisions Finalized

| Decision | Choice | Status |
|----------|--------|--------|
| **Tokenization** | Sign-level (split `value_signs` on spaces) | ✅ |
| **Word Boundaries** | Implicit (Option A - no explicit marker) | ✅ |
| **Objective** | MLM (Masked Language Modeling) | ✅ |
| **Model Architecture** | Simplified Aeneas Twin | ✅ |
| **SAE Layers** | Every 4th layer: [0, 4, 8, 12, 16] | ✅ |

### Data Pipeline Complete

- ✅ ORACC downloaded and processed (1.4M tokens)
- ✅ eBL processed (1M tokens)
- ✅ Archibab processed (65k tokens)
- ✅ Unified dataset created: `v_1/data/processed/unified/`
- ✅ Train/Val/Test splits (80/10/10, no leakage verified)

### Unified Dataset Location

```
v_1/data/processed/unified/
├── train.parquet      (1,960,636 words, 32,343 texts)
├── val.parquet        (253,798 words, 4,042 texts)
├── test.parquet       (235,660 words, 4,044 texts)
├── unified_corpus.parquet
├── dataset_stats.txt
└── eda_summary.txt
```

---

## 🚀 PHASE 1: Baseline Model Implementation (NEXT)

### Phase 1A: Dataset Pipeline
- [ ] **1A.1** Fragment text builder (join `value_signs` per fragment, split on spaces)
- [ ] **1A.2** Sign vocabulary (~16,740 + `[PAD] [UNK] [CLS] [SEP] [MASK]`)
- [ ] **1A.3** PyTorch MLM Dataset + Collator (15-20% masking, 80/10/10 strategy)
- [ ] **1A.4** Fixed eval subset (~500 fragments for pre/post analysis)

### Phase 1B: Model Implementation ("Simplified Aeneas Twin")
- [ ] **1B.1** Implement model: 16 layers, d_model=384, d_kv=32, RoPE, 2-layer MLP head
- [ ] **1B.2** Add hidden states extraction for layers [0, 4, 8, 12, 16]

### Phase 1C: Training + Checkpointing
- [ ] **1C.1** Training script (`baseline_init.pt`, `baseline_best.pt`, `baseline_last.pt`)
- [ ] **1C.2** Extract embeddings + hidden states pre/post training

---

## Model Architecture Reference

**"Simplified Aeneas Twin"** — See `yarin/justification/justification_aeneas_twin_architecture.md`

| Parameter | Value |
|-----------|-------|
| `d_model` | 384 |
| `d_ff` | 1,536 |
| `d_kv` | 32 (per head) |
| `num_heads` | 8 |
| `num_layers` | 16 |
| `vocab_size` | ~16,750 |
| `max_seq_len` | 768 |
| Positional | RoPE |
| Norm | Pre-Norm (RMSNorm) |
| Head | 2-layer MLP |

---

## Documentation Index

| File | Purpose |
|------|---------|
| `yarin/Tasks.md` | Full implementation checklist |
| `yarin/STATUS_SUMMARY.md` | Current status overview |
| `yarin/justification/justification_mlm.md` | MLM decision rationale |
| `yarin/justification/justification_sign_level_tokenization.md` | Tokenization rationale |
| `yarin/justification/justification_aeneas_twin_architecture.md` | Model architecture |
| `yarin/justification/data_source_summary.md` | Dataset composition |
