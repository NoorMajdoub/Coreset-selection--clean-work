#you pass everything normalised
#this is oritented to single label
def get_samples(corpus, C_set, size_samples,seed=42):
    candidates = list(set(range(len(corpus))) - C_set)
    if len(candidates) <= size_samples:
        return candidates
    rng = random.Random(seed)
    return rng.sample(candidates, size_samples)


class WeightedGreedyCoreset:
    
    def __init__(self, corpus, labels, metric, n_neighbors,hardness_scores):
        self.corpus = np.array(corpus, dtype=np.float32)
        self.labels = np.array(labels)
        self.n = len(corpus)
        self.metric = metric
        self.n_neighbors = n_neighbors

        # Compute inverse frequency weights per sample
        # normalize sample_weights to [0,1] before storing
        w = self._compute_weights()
        self.sample_weights = (w - w.min()) / (w.max() - w.min() + 1e-10)
        
        # hardness already [0,1]
        h = hardness_scores
        self.hardness = (h - h.min()) / (h.max() - h.min() + 1e-10)
               



    def _dist(self, a, b):
                if self.metric == "euclidean":
                    return np.linalg.norm(a - b)
                elif self.metric == "cosine":
                    return 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    def _compute_all_dists(self, point):
        if self.metric == 'euclidean':
            return np.linalg.norm(self.corpus - point, axis=1)
        elif self.metric == 'cosine':
            point_norm = point / (np.linalg.norm(point) + 1e-10)
            return 1 - self.corpus @ point_norm
    def select(self, corset_size, sample_size,weights,seed=42):
            print(sample_size, self.n_neighbors, self.metric)
            indices = self._selectfast(corset_size, sample_size,weights,seed)

            labels = self.labels[indices]
            rarity = self.sample_weights[indices]
            hardness = self.hardness[indices]
        
            return {
                "indices": indices,
                "labels": labels,
                "rarity": rarity,
                "hardness": hardness
                   }
    def _compute_weights(self):
        #Inverse frequency weight per sample: rare class → high weight.
        labels = self.labels
        if labels.ndim == 2:
            print("multilabels_weights")
            # Weight = mean inverse frequency across positive labels
            class_counts = labels.sum(axis=0) + 1e-6  # (C,)
            class_weights = 1.0 / class_counts         # (C,)
            class_weights /= class_weights.sum()       # normalize

            # Per-sample weight = sum of weights of its positive classes
            sample_weights = (labels * class_weights).sum(axis=1)  # (N,)
        else:
            # Single-label
            print("singlelabels_weights")
            unique, counts = np.unique(labels, return_counts=True)
            freq = dict(zip(unique, counts))
            sample_weights = np.array([1.0 / freq[l] for l in labels], dtype=np.float32)

        w_min, w_max = sample_weights.min(), sample_weights.max()

        return sample_weights
    
    
    def _selectfast(self, corset_size, sample_size,weights,seed):
            w_cov,w_hard,w_rare=weights
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
            corpus_t = torch.tensor(self.corpus, device=device) 
            hardness_t = torch.tensor(self.hardness, device=device)
            rarity_t = torch.tensor(self.sample_weights, device=device)
            C = []
            set_c = set()
            min_dists = torch.full((self.n,), float('inf'), device=device)
        
            if corset_size > 0:
                mean = corpus_t.mean(dim=0)
                mean = mean / (mean.norm() + 1e-10)
                if self.metric == 'cosine':
                    first_dists = 1 - corpus_t @ mean
                else:
                    first_dists = (corpus_t - mean).norm(dim=1)
                first_idx = int(first_dists.argmax())
                C.append(first_idx)
                set_c.add(first_idx)
                if self.metric == 'cosine':
                    min_dists = 1 - corpus_t @ corpus_t[first_idx]
                else:
                    min_dists = (corpus_t - corpus_t[first_idx]).norm(dim=1)
        
            in_coreset = torch.zeros(self.n, dtype=torch.bool, device=device)
            in_coreset[first_idx] = True
            while len(C) < corset_size:
                    candidates = get_samples(self.corpus, set_c, size_samples=sample_size,seed=seed)
                    cand_idx = torch.tensor(candidates, device=device)
                    cand_vecs = corpus_t[cand_idx]
                
                    if self.metric == 'cosine':
                        dist_matrix = 1 - cand_vecs @ corpus_t.T
                    else:
                        dist_matrix = torch.cdist(cand_vecs, corpus_t)
                
                    improvement = torch.clamp(min_dists.unsqueeze(0) - dist_matrix, min=0)
                    improvement[:, in_coreset] = 0
                
                    coverage = improvement.sum(dim=1)                    # (num_candidates,)
                    coverage = coverage / (coverage.max() + 1e-10)  # normalize to [0,1]

                    utility = w_cov * coverage + w_hard * hardness_t[cand_idx] + w_rare * rarity_t[cand_idx]

                    best_local = int(utility.argmax())
                    best_t = candidates[best_local]
                
                    C.append(best_t)
                    set_c.add(best_t)
                    in_coreset[best_t] = True
                    min_dists = torch.minimum(min_dists, dist_matrix[best_local])
                
                    if len(C) % 100 == 0:
                        print(f"Selected {len(C)}/{corset_size}")
                    """if len(C)  == 61:
                            print(f"Selected {len(C)}/{corset_size}")
                            print(f"  rarity:   {rarity_t[best_t].item():.4f}")
                            print(f"  hardness: {hardness_t[best_t].item():.4f}")
                            print(f"  coverage: {coverage[best_local].item():.4f}")
                            print(f"  utility:  {utility[best_local].item():.4f}") """
            return C
def label_hardness(
    embeddings: np.ndarray,
    labels: np.ndarray,        # expected: [N, L] binary int/float, e.g. ChestMNIST shape [N, 14]
    k: int = 15,
    w_impurity: float = 0.6,
    w_margin: float = 0.4,
    margin_agg: str = "soft_min",
    soft_min_temp: float = 5.0,
    n_jobs: int = -1,single_label=False  #default is multilabels say
) -> np.ndarray:
    """
    Label hardness function
    """
    # --- Input validation & normalisation ---
    labels = np.asarray(labels, dtype=np.float32)  #just makes float

    if labels.ndim == 2 and labels.shape[1] == 1:    #this is for processing single label
    
        print("Single label dataset")
        n_classes = int(labels.max()) + 1
        labels = np.eye(n_classes, dtype=np.float32)[labels.squeeze().astype(int)]  #this is to make hot encoding of single label
    elif labels.ndim == 1:  #hope we dont get this case
        print("Wrong Case")
        n_classes = int(labels.max()) + 1
        labels = np.eye(n_classes, dtype=np.float32)[labels.astype(int)]
    else: #already [N, L] multihot — use as-is 
        print("Multilabel dataset")

    assert labels.ndim == 2, f"labels must be 2D after processing, got shape {labels.shape}"
    assert set(np.unique(labels)).issubset({0.0, 1.0})

    N, L = labels.shape  #N ,14 or N 11
    print(f"[{N} , {L}]")
        
    #Metric 1:KNN neighbors purity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10  # we apply normalisation to the embeddings
    emb = embeddings / norms   
    nbrs = NearestNeighbors(n_neighbors=k + 1, metric="cosine", n_jobs=n_jobs)
    nbrs.fit(emb)
    _, indices = nbrs.kneighbors(emb)
    nbr_idx  = indices[:, 1:]        #shape of [N,K]  those are you k closest neighbors  
    nbr_labels = labels[nbr_idx]       # now you got their labels   
    print(nbr_idx.shape, nbr_labels.shape)
    label_disagreement = (nbr_labels != labels[:, None, :]).astype(np.float32)
    impurity = label_disagreement.mean(axis=2).mean(axis=1)  
    # 1= everyone disagree with you --- 0 everyone agree with you
    # Metric  2:per-label centroid margin
    #but it has to be positive
    per_label_margins = np.zeros((N, L), dtype=np.float32)
    for l in range(L):  #per label
        pos_mask = labels[:, l] == 1
        neg_mask = labels[:, l] == 0
        # skip degenerate labels (all positive or all negative)
        if pos_mask.sum() == 0 or neg_mask.sum() == 0:
            per_label_margins[:, l] = 0.0   # neutral — no information
            continue
        pos_centroid = emb[pos_mask].mean(axis=0)
        neg_centroid = emb[neg_mask].mean(axis=0)
        #since this problem is only with the positive samples "pure samples" for multilabels data
        if single_label==False:
            label_counts = labels.sum(axis=1) 
            weights = 1.0 / (label_counts + 1.0) 
            pos_weights = weights[pos_mask]
            pos_weights /= pos_weights.sum()  # normalize
            pos_centroid = (emb[pos_mask] * pos_weights[:, None]).sum(axis=0)
            neg_weights = weights[neg_mask] 
            neg_weights /= neg_weights.sum()
            neg_centroid = (emb[neg_mask] * neg_weights[:, None]).sum(axis=0)
        pos_centroid /= np.linalg.norm(pos_centroid) + 1e-10
        neg_centroid /= np.linalg.norm(neg_centroid) + 1e-10
        sim_pos = emb @ pos_centroid                     
        sim_neg = emb @ neg_centroid                    
        raw_margin = sim_pos - sim_neg                    
        correct_side = labels[:, l] * 2 - 1               
        per_label_margins[:, l] = correct_side * raw_margin
    τ = soft_min_temp
    agg_margin = -τ * np.log(
            np.exp(-per_label_margins / τ).mean(axis=1) + 1e-10
        )                                                  
    # impurity_norm = np.clip(impurity * 2.0, 0.0, 1.0)
    i_min, i_max = impurity.min(), impurity.max()
    impurity_norm = np.clip(   #we do simple normalisation , is more logial for 
        (impurity - i_min) / (i_max - i_min + 1e-10), 
        0.0, 1.0
    )
    # normalize margin based on actual data range, not assumed [-2,2]
    m_min, m_max = agg_margin.min(), agg_margin.max()
    m_range = m_max - m_min
    # if range is tiny, margin has no signal — let impurity dominate
    margin_confidence = np.clip(m_range / 2.0, 0.0, 1.0)  # 0 = no signal, 1 = full signal
    margin_hard = 1.0 - np.clip((agg_margin - m_min) / (m_range + 1e-10), 0.0, 1.0)
    #sanity check
    """print("=== IMPURITY ===")
    print(f"  min={impurity.min():.3f}  max={impurity.max():.3f}  "
          f"mean={impurity.mean():.3f}  std={impurity.std():.3f}")
    print("=== AGG MARGIN ===")
    print(f"  min={agg_margin.min():.3f}  max={agg_margin.max():.3f}  "
          f"mean={agg_margin.mean():.3f}  std={agg_margin.std():.3f}")"""
    # blend: when margin has no signal, fall back entirely to impurity
    hardness = (
        w_impurity * impurity_norm +
        w_margin * margin_confidence * margin_hard +
        w_margin * (1.0 - margin_confidence) * impurity_norm  # redirect margin weight to impurity
    ).astype(np.float32)
    return np.clip(hardness, 0.0, 1.0)
