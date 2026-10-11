import math

import numpy as np

from scripts import explain_local_model_shap
from scripts.evaluate_shared_feature_federation import FEATURES
from src.federated.simulated_fedavg import stratified_split


def test_shap_explanations_are_aligned_additive_and_reproducible(monkeypatch):
    rng = np.random.default_rng(2026)
    raw_features = rng.lognormal(mean=2.0, sigma=1.1, size=(500, len(FEATURES)))
    x = np.log1p(raw_features)
    score = (x[:, 0] - 2.1) + 0.35 * (x[:, 1] - 2.1) - 0.25 * (x[:, 2] - 2.1)
    y = (score + rng.normal(0, 1.25, len(x)) > 0).astype(np.int64)
    monkeypatch.setattr(
        explain_local_model_shap,
        "load_rows",
        lambda _: (x, y, {"rows": len(y)}),
    )

    original_train = explain_local_model_shap.train_parameters
    training_inputs = []

    def record_training(x_train, y_train, parameters):
        training_inputs.append(x_train.copy())
        return original_train(x_train, y_train, parameters)

    monkeypatch.setattr(explain_local_model_shap, "train_parameters", record_training)

    original_explainer = explain_local_model_shap.shap.LinearExplainer
    background_inputs = []

    def record_background(model, masker):
        background_inputs.append(np.asarray(masker.data).copy())
        return original_explainer(model, masker)

    monkeypatch.setattr(explain_local_model_shap.shap, "LinearExplainer", record_background)

    first = explain_local_model_shap.explain()
    second = explain_local_model_shap.explain()
    train_idx, eval_idx = stratified_split(y, seed=explain_local_model_shap.SEED)

    assert first == second
    assert first["features"] == list(FEATURES)
    assert {example["case"] for example in first["examples"]} == {
        "correct_benign",
        "correct_attack",
        "false_positive",
        "false_negative",
    }
    assert len(first["global_importance_mean_abs_shap_logit"]) == len(FEATURES)
    assert {
        item["feature"] for item in first["global_importance_mean_abs_shap_logit"]
    } == set(FEATURES)

    training_rows = {tuple(row) for row in x[train_idx]}
    evaluation_rows = {tuple(row) for row in x[eval_idx]}
    for fitted_input, background in zip(training_inputs, background_inputs, strict=True):
        assert np.array_equal(fitted_input, x[train_idx])
        background_rows = {tuple(row) for row in background}
        assert background_rows <= training_rows
        assert background_rows.isdisjoint(evaluation_rows)

    for example in first["examples"]:
        row = int(eval_idx[example["evaluation_row_index"]])
        contributions = example["feature_contributions"]
        assert [item["feature"] for item in contributions] == list(FEATURES)
        assert example["actual_class"] == int(y[row])
        assert math.isclose(
            example["base_value_logit"] + sum(
                item["shap_logit_contribution"] for item in contributions
            ),
            example["logit"],
            rel_tol=0.0,
            abs_tol=1e-8,
        )
        assert math.isclose(
            example["attack_probability"],
            1.0 / (1.0 + math.exp(-example["logit"])),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        assert example["predicted_class"] == int(example["attack_probability"] >= 0.5)
        assert [
            abs(item["shap_logit_contribution"])
            for item in example["top_contributors"]
        ] == sorted(
            (abs(item["shap_logit_contribution"]) for item in contributions),
            reverse=True,
        )
