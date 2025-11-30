# V1 - Akkadian Text Research

Clean implementation for exploring and working with Akkadian text data.

## What You Have Now

### Data Structure
```
v_1/
├── data/downloaded/
│   ├── full_corpus_dir/     # 28,194 CSV files (one per fragment)
│   └── filtered_json_files/  # 30 JSON files (raw source data)
│
├── notebooks/
│   └── 01_data_exploration.ipynb  # EDA notebook
│
├── requirements.txt          # Python dependencies
└── setup_jupyter.sh         # Setup script
```

### The Data

All data is in CSV format with these columns:
- `fragment_id`: Unique identifier (e.g., "1848,0720.121")
- `fragment_line_num`: Line number in the fragment
- `index_in_line`: Word position in the line
- `word_language`: Language (mostly AKKADIAN)
- `value`: Original text
- `clean_value`: Cleaned text
- `lemma`: Base form of word
- `domain`: Text genre (e.g., "CANONICAL ➝ Divination ➝ Celestial")
- `place_discovery`: Where found
- `place_composition`: Where originally written

## Getting Started

### 1. Setup Environment
```bash
cd v_1
./setup_jupyter.sh
```

### 2. Activate Environment
```bash
source venv/bin/activate
```

### 3. Start Jupyter
```bash
jupyter notebook
```

### 4. Open the EDA Notebook
Open `notebooks/01_data_exploration.ipynb` and run the cells to explore your data.

## What the EDA Notebook Does

1. Counts all available fragments (28,194)
2. Loads and inspects a single fragment
3. Shows the fragment structure (columns, data types)
4. Reconstructs text from the tabular format
5. Loads 100 fragments for statistics
6. Analyzes language distribution
7. Shows top domains/genres
8. Finds most common Akkadian words

## Next Steps

After running the EDA notebook, you'll understand:
- How many fragments you have
- What languages are present
- What domains/genres exist
- How the data is structured
- What the vocabulary looks like

Then we can decide together what to build next.
