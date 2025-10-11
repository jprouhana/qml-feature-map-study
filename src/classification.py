"""
VQC training with different feature maps for comparison.
"""

from qiskit.circuit.library import RealAmplitudes
from qiskit_algorithms.optimizers import COBYLA
from qiskit_machine_learning.algorithms import VQC
from sklearn.metrics import accuracy_score


def train_vqc_with_feature_map(feature_map, X_train, y_train,
                                ansatz_reps=3, maxiter=100):
    """
    Train a VQC using the given feature map and RealAmplitudes ansatz.
    """
    n_qubits = feature_map.num_qubits
    ansatz = RealAmplitudes(num_qubits=n_qubits, reps=ansatz_reps,
                            entanglement='full')
    optimizer = COBYLA(maxiter=maxiter)

    obj_values = []
    def callback(weights, obj_value):
        obj_values.append(obj_value)

    vqc = VQC(
        feature_map=feature_map,
        ansatz=ansatz,
        optimizer=optimizer,
        callback=callback,
    )

    vqc.fit(X_train, y_train)

    return vqc, {'objective_values': obj_values}


def evaluate_model(model, X_test, y_test):
    """Evaluate a trained VQC and return accuracy + predictions."""
    predictions = model.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    return acc, predictions


def compare_feature_maps_classification(feature_maps_dict, X_train, y_train,
                                         X_test, y_test):
    """
    Train a VQC with each feature map and compare test accuracies.
    Returns dict mapping feature map name -> accuracy.
    """
    results = {}
    for name, fm in feature_maps_dict.items():
        print(f"  Training with {name}...")
        model, info = train_vqc_with_feature_map(fm, X_train, y_train)
        acc, _ = evaluate_model(model, X_test, y_test)
        results[name] = acc
        print(f"    Accuracy: {acc:.4f}")
    return results
