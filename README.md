# 🚀 Project Name
Official implementation of ReFine:


## 🌟 Main Contributions

:one: We identify two key challenges of LLM in tabular data generation in low-data regimes: (i) distributional drift of the synthetic data; and (ii) localized redundancy in the synthetic data. 

:two: To address the two challenges, we propose ReFine1, a framework that constructs association rules to guide tabular data generation, and applies proxy-based distribution estimation with dual-granularity curation to correct localized redundancy.

:three: Experimental results demonstrate that ReFine consistently outperforms strong baselines, achieving up to 0.44 absolute gain in R2 for regression and 10% relative improvement in F1 for classification. Comprehensive ablations further highlight the respective contributions of Rules-Guided Generation and Dual-Granularity Filtering components.

---

## 📂 Run the Code

### Step 1
Generate rules.

```bash
python rf_llm.py
```

### Step 2
Merge rules generated in Step 1.

```bash
python consistency.py
```

### Step 3
Use merged rule in Step 2 to guide LLM to generate data.

```bash
python llm_generation.py
```


## 📊 Results
![Results](image.png)


## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Citations

