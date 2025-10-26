# QML Feature Map Study

Systematic comparison of quantum data encoding strategies and their impact on variational quantum classifier performance. Investigates how the choice of feature map affects classification accuracy, quantum kernel alignment, and circuit expressibility.

Built as part of independent study work on quantum-classical hybrid optimization.

## Background

In quantum machine learning, the **feature map** — the circuit that encodes classical data into quantum states — is arguably the most important design decision. Unlike classical neural networks where the input layer is straightforward, the quantum feature map fundamentally determines what the model can and cannot learn.

### Why Feature Maps Matter

A quantum feature map $U_\Phi(\mathbf{x})$ maps classical data $\mathbf{x}$ into a quantum state $|\phi(\mathbf{x})\rangle = U_\Phi(\mathbf{x})|0\rangle^{\otimes n}$. The geometry of these quantum states in Hilbert space determines:

- **Decision boundaries** — what class separations the model can represent
- **Kernel function** — the similarity measure $K(x_i, x_j) = |\langle\phi(x_i)|\phi(x_j)\rangle|^2$
- **Expressibility** — how much of the Hilbert space the model can explore
- **Trainability** — whether gradient-based optimization can find good parameters

### Feature Maps Compared

| Feature Map | Encoding | Entanglement | Qubits Needed |
|------------|----------|-------------|---------------|
| Angle Encoding | $R_Y(x_i)$ per qubit | None | n = features |
| ZZFeatureMap | $R_Z(x_i) + R_{ZZ}(x_ix_j)$ | Pairwise ZZ | n = features |
| IQP-style | Diagonal + Hadamard layers | Diagonal ZZ | n = features |
| Amplitude Encoding | Encode in state amplitudes | Implicit | $\log_2$(features) |

## Project Structure

```
qml-feature-map-study/
├── src/
│   ├── feature_maps.py          # All feature map implementations
│   ├── kernel_analysis.py       # Kernel computation and alignment
│   ├── expressibility.py        # Expressibility and effective dimension
│   ├── classification.py        # VQC training with different maps
│   ├── data_utils.py            # Dataset loading and preprocessing
│   └── plotting.py              # Visualization functions
├── notebooks/
│   └── feature_map_comparison.ipynb
├── results/
├── requirements.txt
├── README.md
└── LICENSE
```

## Installation

```bash
git clone https://github.com/jrouhana/qml-feature-map-study.git
cd qml-feature-map-study
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

### Quick Start

```python
from src.feature_maps import get_all_feature_maps
from src.classification import compare_feature_maps_classification
from src.data_utils import load_moons

X_train, X_test, y_train, y_test = load_moons(n_samples=200)
feature_maps = get_all_feature_maps(n_qubits=2)

results = compare_feature_maps_classification(
    feature_maps, X_train, y_train, X_test, y_test
)
for name, acc in results.items():
    print(f"{name}: {acc:.4f}")
```

### Jupyter Notebook

```bash
jupyter notebook notebooks/feature_map_comparison.ipynb
```

## Results

### Classification Accuracy

| Feature Map | make_moons | make_circles | Iris 2D |
|------------|-----------|-------------|---------|
| Angle Encoding | 0.82 | 0.78 | 0.88 |
| ZZFeatureMap | 0.93 | 0.91 | 0.95 |
| IQP-style | 0.90 | 0.89 | 0.93 |
| Amplitude Encoding | 0.86 | 0.84 | 0.90 |

*Accuracies averaged over 3 random seeds.*

### Key Findings

- **ZZFeatureMap** consistently outperforms other encodings on these datasets — the pairwise entangling interactions help capture nonlinear structure
- **Angle encoding** is the simplest but least expressive, since it creates product states with no entanglement
- **IQP-style** performs well and has theoretical guarantees for classical hardness of simulation
- **Kernel target alignment** correlates strongly with classification accuracy — it's a useful metric for feature map selection without full training
- Expressibility alone doesn't predict performance; alignment with the specific dataset matters more

## References

1. Havlicek, V., et al. (2019). "Supervised learning with quantum-enhanced feature spaces." *Nature*, 567, 209-212.
2. Schuld, M. (2021). "Supervised quantum machine learning models are kernel methods." [arXiv:2101.11020](https://arxiv.org/abs/2101.11020)
3. Sim, S., Johnson, P. D., & Aspuru-Guzik, A. (2019). "Expressibility and entangling capability of parameterized quantum circuits." [arXiv:1905.10876](https://arxiv.org/abs/1905.10876)
4. Abbas, A., et al. (2021). "The power of quantum neural networks." *Nature Computational Science*, 1, 403-409.

## License

MIT License — see [LICENSE](LICENSE) for details.
