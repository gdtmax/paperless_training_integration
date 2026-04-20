import os
import yaml
import mlflow
from mlflow.tracking import MlflowClient


# ========================
# 从 MLflow 读取最新 run 的 metrics
# ========================
def get_latest_metrics(client, experiment_name, run_name_prefix):
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"Experiment {experiment_name} not found")

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"]
    )

    for run in runs:
        if run.data.tags.get("mlflow.runName", "").startswith(run_name_prefix):
            return run.data.metrics

    raise ValueError(f"No run found for {run_name_prefix}")


# ========================
# 核心：执行 gate 判断
# ========================
def check_gate(model_type, metrics, gate_config):
    required = gate_config.get("required_metrics", [])
    thresholds = gate_config.get("thresholds", {})
    fail_on_missing = gate_config.get("fail_on_missing_metrics", True)

    # 1. 检查 required metrics
    for m in required:
        if m not in metrics:
            if fail_on_missing:
                return False, f"Missing metric: {m}"
            else:
                continue

    # 2. 检查 thresholds
    for key, value in thresholds.items():

        # max_loss
        if key == "max_loss":
            if metrics.get("loss", float("inf")) > value:
                return False, f"Loss too high: {metrics.get('loss')}"

        # min_accuracy
        elif key == "min_accuracy":
            if metrics.get("accuracy", 0) < value:
                return False, f"Accuracy too low: {metrics.get('accuracy')}"

        # min_recall_at_k
        elif key == "min_recall_at_k":
            if metrics.get("recall_at_k", 0) < value:
                return False, f"Recall too low: {metrics.get('recall_at_k')}"

        # embedding_dim_check
        elif key == "expected_embedding_dim":
            if metrics.get("embedding_dim_check", 0) != 1:
                return False, "Embedding dim check failed"

    return True, "PASS"


# ========================
# 查找 baseline training run,用于 register_model
# ========================
def find_baseline_training_run(client, model_type):
    """Return (run_id, run) for the `{model_type}_baseline` training
    run, or (None, None) if there isn't one yet.

    Note: eval.py loads the fixed checkpoint at
    outputs/checkpoints/{model_type}_baseline.pt, so the quality gate's
    PASS/FAIL decision is always about the `baseline` run specifically.
    We register the SAME run's logged model artifact here to keep
    evaluation and registration referring to the same weights.
    """
    experiment = client.get_experiment_by_name("training")
    if experiment is None:
        return None, None

    run_name = f"{model_type}_baseline"
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"tags.mlflow.runName = '{run_name}'",
        order_by=["start_time DESC"],
        max_results=1,
    )
    if runs:
        return runs[0].info.run_id, runs[0]
    return None, None


# ========================
# 主函数
# ========================
def main():
    # Tracking URI env-overridable so the same gate logic runs against
    # a shared MLflow server (e.g. http://mlflow:5000 inside compose).
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
    mlflow.set_experiment("quality_gate")

    client = MlflowClient()

    with open("configs/quality_gate.yaml", "r") as f:
        config = yaml.safe_load(f)

    results = {}

    for model_type, model_cfg in config["models"].items():
        print(f"\n===== Checking {model_type} =====")

        gate_cfg = model_cfg["gate"]

        if not gate_cfg.get("enabled", True):
            print("Gate disabled")
            continue

        # 从 evaluation experiment 取结果
        metrics = get_latest_metrics(
            client,
            experiment_name="evaluation",
            run_name_prefix=f"eval_{model_type}"
        )

        passed, reason = check_gate(model_type, metrics, gate_cfg)

        # `status` is the gate-metric verdict only. `registration_status`
        # is orthogonal — it tracks whether promotion to the Model
        # Registry actually happened. Keeping the two separate avoids
        # the internally-inconsistent "PASS but not registered" state.
        results[model_type] = {
            "status": "PASS" if passed else "FAIL",
            "reason": reason,
            "registration_status": "NOT_APPLICABLE",   # gate didn't pass
            "registered_version": None,
            "register_error": None,
        }

        # On PASS, promote to the Model Registry so downstream deploy
        # scripts (e.g. paperless-ml/scripts/deploy_model.sh) can find
        # the latest gated model via `models:/paperless-<type>`. This is
        # the canonical rubric pattern — "saved models are registered
        # ONLY if they pass the quality gate."
        if passed:
            run_id, _ = find_baseline_training_run(client, model_type)
            if run_id is None:
                msg = f"no `{model_type}_baseline` training run found"
                results[model_type]["registration_status"] = "SKIPPED"
                results[model_type]["register_error"] = msg
                print(f"  skipping register_model: {msg}")
            else:
                model_uri = f"runs:/{run_id}/model"
                try:
                    mv = mlflow.register_model(
                        model_uri=model_uri,
                        name=f"paperless-{model_type}",
                    )
                    results[model_type]["registration_status"] = "REGISTERED"
                    results[model_type]["registered_version"] = mv.version
                    print(f"  registered paperless-{model_type} v{mv.version} from {model_uri}")
                except Exception as e:
                    results[model_type]["registration_status"] = "FAILED"
                    results[model_type]["register_error"] = str(e)
                    print(f"  register_model failed for {model_type}: {e}")

        print(f"{model_type}: {results[model_type]}")

        # log 到 MLflow
        with mlflow.start_run(run_name=f"gate_{model_type}"):
            mlflow.log_param("model_type", model_type)
            mlflow.log_param("status", results[model_type]["status"])
            mlflow.log_param("reason", results[model_type]["reason"])
            mlflow.log_param("registration_status", results[model_type]["registration_status"])
            if results[model_type]["registered_version"] is not None:
                mlflow.log_param("registered_version", results[model_type]["registered_version"])
            # Exception strings can exceed MLflow's param length limit
            # (~500 chars). Tags tolerate ~5000, so use set_tag here.
            if results[model_type]["register_error"] is not None:
                mlflow.set_tag("register_error", results[model_type]["register_error"])

    print("\n===== FINAL RESULT =====")
    for k, v in results.items():
        print(k, ":", v)


if __name__ == "__main__":
    main()