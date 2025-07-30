# OCR Testing Record & Report

## 🧪 1. Test Setup and Hardware Configuration

| Item | Value |
|------|-------|
| Model | olmOCR |
| Inference Backend | Modal |
| Input Type | PDF or PNG |
| GPU Types | A100 (40GiB), H100 (80GiB) |
| Key Optimized Parameters | `batch_size`, `max_num_seqs` |
| Timing Source | Modal internal execution time (in seconds) |

---

## 📁 2. Pre-Optimization (A100)

### 📌 A100 - `batch_size=3`, `max_num_seqs=3`

| File | Pages | Time (s) |
|------|-------|----------|
| Heyjude.pdf | 1 | 2 |
| Resume.pdf | 1 | 13 |
| BingjiaYang_Resume_7.25 | 3 | 28 |
| Charts.png | 1 | 3 |
| Heyjude.png | 1 | 2 |
| Resume.png | 1 | 13 |

🧐 Image file performance is stable, and optimization for multi-page PDFs is even more valuable.

---

## 🧪 3. Multi-page PDF Testing

### 📌 A100 - `batch_size=3`, `max_num_seqs=3`

| Pages | Time (s) |
|-------|----------|
| 1 | 11 |
| 3 | 13 |
| 6 | 26 |
| 9 | 39 |
| 12 | 51 |

---

### 📌 H100 - `batch_size=3`, `max_num_seqs=3`

| Pages | Time (s) |
|-------|----------|
| 1 | 7 |
| 3 | 8 |
| 6 | 16 |
| 9 | 24 |
| 12 | 31 |

✅ ~40% speedup over A100 under same config.

---

### 📌 H100 - `batch_size=6`, `max_num_seqs=6`

| Pages | Time (s) |
|-------|----------|
| 1 | 7 |
| 3 | 8 |
| 6 | 10 |
| 9 | 17 |
| 12 | 19 |

✅ Clear batching benefit. 6 pages ≈ 1 page time.

---

### 📌 H100 - `batch_size=12`, `max_num_seqs=12`

| Pages | Time (s) |
|-------|----------|
| 6 | 10 |
| 12 | 14 |
| 24 | 28 |
| 48 | 57 |

✅ Higher batch → better throughput. Sublinear growth.

---

### 📌 H100 - `batch_size=48`, `max_num_seqs=48`

| Pages | Time (s) |
|-------|----------|
| 6 | 12 |
| 12 | 15 |
| 24 | 19 |
| 48 | 41 |
| 96 | 66 |

✅ Best for large documents (24+ pages). Overhead increases for smaller tasks.

---

## ✅ 4. Conclusions and Recommendations

### 🧠 Key Insights

- **H100 outperforms A100** by ~35–45% in execution time under same settings.
- **Batching effective from `batch_size ≥ 6`**. Beyond `12`, marginal gains flatten.
- `batch_size=48` shows highest throughput but only beneficial for long documents.

---

### 🛠 Recommended Configurations

| Pages | Config | GPU | Notes |
|-------|--------|-----|-------|
| 1–3 | `batch_size=3`, `max_num_seqs=3` | A100/H100 | No need for batching |
| 4–12 | `batch_size=12`, `max_num_seqs=12` | H100 preferred | Best trade-off |
| >12 | `batch_size=48`, `max_num_seqs=48` | H100 | Max throughput |

---

## 💰 Billing Strategy Tips

- Modal bills based on app **runtime**, even without active calls.
- 🔒 Always stop apps after testing.