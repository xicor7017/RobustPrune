## 5 Experiments

Below we outline a comprehensive experimental section to validate Frequency‑Based Pruning (FBP) and Activation‑Variability Pruning (AVP) against a broad suite of baselines and datasets. We cover:
1. A controlled synthetic benchmark  
2. Vision tasks (CelebA & Waterbirds)  
3. An NLP task (MNLI)  
4. Toxicity detection (FDCL18)  
5. Ablations & new dataset proposals  

---

### 5.1 Experimental Setup

#### 5.1.1 Synthetic 2D Gaussian Benchmark  
- **Data generation**  
  - Five classes in ℝ²: four centered uniformly on the unit circle + one at the origin, shared isotropic variance σ².  
  - **Biased labels:** class _i_ → label _j_ deterministically (structured mislabeling).  
  - **Unbiased noisy labels:** random flip to any other class with probability ϵ.  
- **Model & pruning**  
  - MLP with two 32‑unit hidden layers (Tanh), trained until zero training error.  
  - **AVP schedule:** prune the two lowest‑variance weights per epoch with p=0.5.  
- **Baselines**  
  - Random‑Prune, magnitude‑based pruning, PruSC (Lê et al.), Silent‑Majority (You et al.), DFR (Kirichenko et al.).  
- **Metrics**  
  - **Generalization accuracy** on clean test set  
  - **Decision boundary alignment** (visual inspection + boundary‑error rate)  
  - **Spurious‑ratio** β̂ (Def. 3.10) before/after pruning  

#### 5.1.2 Vision Benchmarks: CelebA & Waterbirds (Main comparison paper: https://arxiv.org/pdf/2501.14182v1)

- **Datasets**  
  - **CelebA**: classify hair color (blonde/dark); gender as spurious attribute. Four group splits (hair×gender) for “clean” evaluation; over‑represent dark‑hair males for “imperfect” split.  
  - **Waterbirds**: bird species classification with correlated backgrounds; four bird×background groups.  
- **Models & pruning**  
  - ResNet‑18 & ResNet‑50 pretrained on ImageNet, fine‑tuned with Adam (lr=1e‑4, wd=1e‑5, bs=128) for 50 epochs.  
  - **AVP** applied from epoch 10 to final epoch; prune 2 candidates/epoch chosen by activation variability κ<sub>ij</sub> (Def. 3.11).  
- **Baselines**  
  - ERM, GroupDRO, GC‑DRO, Random‑Prune, PruSC, DaC (Hosseininoohdani et al.), ExMap (Chakraborty et al.), Silent‑Majority.  
- **Metrics**  
  - **Worst‑group accuracy** (min across 4 groups) and **average accuracy** on test splits  
  - **Sparsity vs. robustness curves**: trade‑off between pruned fraction and worst‑group performance  

#### 5.1.3 NLP Benchmark: MNLI  
- **Dataset**  
  - MNLI with negation words (“no”, “nothing”) spuriously correlated with the “contradiction” label.  
  - **Clean:** 3 negation‑levels × 3 labels → 9 groups; **Imperfect:** collapse to 3 groups mixing negation.  
- **Model & pruning**  
  - RoBERTa‑base, fine‑tuned (lr=2e‑5, bs=32) for 5 epochs.  
  - **AVP** on intermediate transformer layers: prune lowest‑variance heads/neurons per epoch.  
- **Baselines**  
  - ERM, GroupDRO, GC‑DRO, SELF (LaBonte et al.), Explanation‑Based Finetuning (Ludan et al.), CER (Kumar et al.).  
- **Metrics**  
  - Worst‑group & average accuracy on held‑out splits  
  - **Δβ̂**: reduction in learned spurious ratio (Def. 3.10)  

#### 5.1.4 Toxicity Detection: FDCL18  
- **Dataset**  
  - 100 K tweets labeled {hateful, spam, abusive, normal}, dialect as spurious attribute.  
  - **Clean:** 4 dialects × 4 labels → 16 groups; **Imperfect:** 4 groups by dialect only.  
- **Model & pruning**  
  - RoBERTa‑base, fine‑tuned (lr=2e‑5, bs=32) for 5 epochs.  
  - **AVP** on final classifier layer: prune low‑variance weights.  
- **Baselines**  
  - ERM, GroupDRO, Soft‑Label Integration (Li et al.), DPR (Han et al.).  
- **Metrics**  
  - Worst‑group and average F1‑score  
  - **Bias drop:** AUC improvement on minority‑dialect groups  

---

### 5.2 Implementation Details  
- **Optimization:** Adam with linear warmup (10% steps) and cosine decay.  
- **Pruning hyperparameters:** candidates = 2/epoch, prune‑prob = 0.5; ablate pruning rate {1, 2, 5} and start epoch {5, 10, 20}.  
- **Repetitions:** 5 random seeds; report mean ± std.  

---

### 5.3 Evaluation Metrics  
1. **Average accuracy** (or F1/F1‑macro)  
2. **Worst‑group accuracy**: minimum over pre‑defined groups  
3. **Spurious ratio** β̂ (Def. 3.10): measures residual spurious dependence  
4. **Sparsity vs. robustness curves**: trade‑off between pruned fraction and worst‑group performance  
5. **Latent alignment**: cosine similarity between learned latents **ẑ** and ground‑truth **z** (synthetic only)  

---

### 5.4 Baselines & Comparative Methods  

| Category                              | Methods                                                                                   |
|---------------------------------------|-------------------------------------------------------------------------------------------|
| **ERM & Robust Training**             | ERM; Resampling; GroupDRO; GC‑DRO; DPR; DIVDIS                                            |
| **Pruning‑based**                     | Random‑Prune; Magnitude‑Prune; FBP (ours); AVP (ours); PruSC                              |
| **Data Augmentation / Counterfactual**| DaC; DISC; LBC; Explanation‑Based Finetuning; Concept‑Level Augmentation                   |
| **Representation‑based**              | DFR; SELF; Silent‑Majority; ExMap; SFB; LC; CER                                           |
| **Post‑hoc Editing**                  | Single‑Weight Editing; Logit Correction                                                   |

---

### 5.5 Ablation Studies  
1. **Pruning signal ablation:** compare AVP vs. magnitude vs. random  
2. **Pruning schedule:** one‑shot vs. per‑epoch vs. adaptive (e.g. prune more early)  
3. **Layer‑wise pruning:** only classifier head vs. only feature extractor vs. all layers  
4. **Perturbation diversity:** vary synthetic Δ for 2D Gaussian to test Theorem 3.6 invariance  
5. **Interaction with dataset difficulty:** run on SpuCoMNIST, SpuCoAnimals, Spawrious to assess scalability  

---

### 5.6 New Dataset Proposals  
1. **SpuCo-Graph:** synthetic graphs where node color spuriously correlates with label—test GNN pruning  
2. **SpuCo-TimeSeries:** time‑series with seasonality spuriously linked to event labels—test RNN/Transformer pruning  
3. **SpuCo-Audio:** speech commands with background noise spuriously predictive—evaluate audio CNN pruning  
4. **SpuCo-Tabular:** tabular data where one feature group is spurious—test decision‑tree and MLP pruning  
5. **SpuCo-MultiSpur:** mixtures of multiple spurious attributes in vision (e.g. color, texture, context) at controlled strengths  

---

**Summary:**  
This experimental design rigorously evaluates FBP & AVP across controlled synthetic and real‑world tasks, benchmarks against a wide spectrum of baselines, and probes the key components—pruning signal, schedule, and scope—ensuring both robustness and latent‑identification improvements necessary for ICLR 2026 viability.  
