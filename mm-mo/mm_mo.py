import random
import time
import subprocess
import numpy as np
import yaml
import logging
import sys
import os
import torch
import json

from merge_local import run_mergekit_yaml
from evaluate_model_opencompass import opencompass_eval_mcq, opencompass_eval_nomcq
from evaluate_model_fitness import cal_delta_l1_norm
from utils.utils import cleanup_merged_models, get_info_from_json, save_info_to_json

from botorch.models.gp_regression import SingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP
from botorch import fit_gpytorch_mll
from botorch.acquisition.multi_objective.monte_carlo import (
    qExpectedHypervolumeImprovement,
    qNoisyExpectedHypervolumeImprovement,
)
from botorch.optim.optimize import optimize_acqf, optimize_acqf_list
from botorch.sampling.normal import SobolQMCNormalSampler
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
from botorch.utils.transforms import unnormalize, normalize
from botorch.utils.multi_objective.box_decompositions.dominated import (
    DominatedPartitioning,
)
from botorch.utils.multi_objective.box_decompositions.non_dominated import (
    FastNondominatedPartitioning,
)

logger = logging.getLogger(__name__)

device = None
bounds = None
current_dir = None
merged_path = None

# Real Evaluation = INIT_SIZE + N_BATCH * NUM_OF_OFFSPRINGS
INIT_SIZE = 10
NUM_OF_OBJECTIVES = 3
N_BATCH = 4
BATCH_SIZE = 10
NUM_OF_OFFSPRINGS = 5

MC_SAMPLES = 128
NUM_RESTARTS = 10
RAW_SAMPLES = 512


get_dare_ties_config = (
    lambda w: f"""
models:
  - model: Qwen/Qwen1.5-7B
    # No parameters necessary for base model
  - model: Qwen/Qwen1.5-7B-Chat
    parameters:
      density: {w[0]}
      weight: {w[1]}
  - model: abacusai/Liberated-Qwen1.5-7B
    parameters:
      density: {w[2]}
      weight: {w[3]}
  - model: YeungNLP/firefly-qwen1.5-en-7b
    parameters:
      density: {w[4]}
      weight: {w[5]}
merge_method: dare_ties
base_model: Qwen/Qwen1.5-7B
parameters:
  int8_mask: true
dtype: bfloat16
random_seed: 0
    """
)


def setup_logger():
    logging.basicConfig(level=logging.INFO)
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    os.makedirs("./save_logs/mm-mo-opencompass", exist_ok=True)
    fh = logging.FileHandler(f"./save_logs/mm-mo-opencompass/{str(time.time())}.log")
    fh.setLevel(logging.INFO)

    ch = logging.StreamHandler()
    ch.setLevel(logging.WARNING)

    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    if not any(isinstance(h, logging.FileHandler) for h in root_logger.handlers):
        root_logger.addHandler(fh)
    if not any(isinstance(h, logging.StreamHandler) for h in root_logger.handlers):
        root_logger.addHandler(ch)

    return root_logger


def get_evaluated_objectives():
    merge_info_path = os.path.join(current_dir, "merge_info", "merge_info.json")
    with open(merge_info_path, "r") as file:
        data = json.load(file)

    ceval_accuracy = np.array([entry["CEVAL Accuracy"] for entry in data])
    gsm8k_accuracy = np.array([entry["GSM8K Accuracy"] for entry in data])
    pareto_front = np.vstack((ceval_accuracy, gsm8k_accuracy)).T
    return pareto_front


def select_best_solutions(population, num_best=5):
    scores = get_evaluated_objectives()
    composite_scores = np.sum(scores, axis=1)
    best_indices = np.argsort(composite_scores)[::-1][:num_best]
    best_population = [population[i] for i in best_indices]
    return best_population


def adjust_solution(
    new_solution, best_solutions, bounds, mutation_factor=0.5, crossover_prob=0.5
):
    best_solutions = np.asarray(best_solutions)
    if best_solutions.ndim != 2:
        raise ValueError("best_solutions should be a 2D array")

    bounds_np = bounds.cpu().numpy()
    lower_bounds, upper_bounds = np.asarray(bounds_np[0]), np.asarray(bounds_np[1])

    adjusted_solutions = []
    for solution in new_solution:
        indices = np.random.choice(best_solutions.shape[0], 3, replace=False)
        a, b, c = best_solutions[indices]
        mutant = np.clip(a + mutation_factor * (b - c), lower_bounds, upper_bounds)

        crossover = np.random.rand(len(lower_bounds)) < crossover_prob
        if not np.any(crossover):
            crossover[np.random.randint(0, len(lower_bounds))] = True

        adjusted_solution = np.where(crossover, mutant, solution)
        adjusted_solutions.append(adjusted_solution)

    return np.array(adjusted_solutions)


def have_evaluated(unique_id):
    if os.path.exists("./merge_info/merge_info.json"):
        info_list = get_info_from_json()
        for info in info_list:
            if info["Model ID"] == unique_id:
                return info.get("Evaluated", False)
    return False


def save_config(config, file_path):
    with open(file_path, "w") as file:
        file.write(config)


def generate_random_config():
    """dare-ties configuration generator"""
    w = [random.uniform(0, 1) for _ in range(6)]
    w = [0.001 if val == 0 else 0.999 if val == 1 else val for val in w]
    config = get_dare_ties_config(w)
    unique_id = "-".join([str(np.round(x, 2)) for x in w])
    logger.info(f"unique_id: {unique_id}")

    name = f"{unique_id}"
    local_file_path = f"./config/{name}.yaml"
    save_config(config, local_file_path)
    return name


def evaluate_accuracy(model_name):
    model_path = os.path.join(merged_path, model_name)

    ceval_accuracy = opencompass_eval_mcq(model_name)
    gsm8k_accuracy = opencompass_eval_nomcq(model_name)

    assert ceval_accuracy is not None, "CEVAL Accuracy should not be None"
    assert gsm8k_accuracy is not None, "GSM8K Accuracy should not be None"

    delta_l1_norm = cal_delta_l1_norm(
        model_name_or_path=model_path, base_model="Qwen/Qwen1.5-7B"
    )
    delta_l1_norm = -delta_l1_norm
    assert delta_l1_norm is not None, "Delta L1-Norm should not be None"

    cleanup_merged_models(model_path)

    logger.info(f"CEVAL Accuracy: {ceval_accuracy}")
    logger.info(f"GSM8K Accuracy: {gsm8k_accuracy}")
    logger.info(f"Delta L1-Norm: {delta_l1_norm}")

    info = {
        "Model ID": model_name,
        "Dataset Name": "C-EVAL",
        "CEVAL Accuracy": ceval_accuracy,
        "GSM8K Accuracy": gsm8k_accuracy,
        "Delta L1-Norm": delta_l1_norm,
        "Evaluated": True,
    }

    os.makedirs("./merge_info", exist_ok=True)
    save_info_to_json(info, "./merge_info/merge_info.json")
    return ceval_accuracy, gsm8k_accuracy, delta_l1_norm


def evaluate_config(unique_id):
    evaluated_flag = have_evaluated(unique_id)
    if evaluated_flag:
        logger.info(f"Model {unique_id} has been evaluated.")
        info_list = get_info_from_json()
        for info in info_list:
            if info["Model ID"] == unique_id:
                return info["CEVAL Accuracy"], info["GSM8K Accuracy"], info["Delta L1-Norm"]

    logger.info("=================================")
    logger.info(f"         Begin Merging: {unique_id}")
    logger.info("=================================")

    _ = run_mergekit_yaml(
        f"./config/{unique_id}.yaml",
        f"./merged/{unique_id}",
        options=["--copy-tokenizer"],
    )

    logger.info("=================================")
    logger.info("         Merging Complete")
    logger.info("=================================")

    model_name = unique_id
    logger.info(f"Model name: {model_name}")

    evaluate_accuracy(model_name)

    logger.info("=================================")
    logger.info("         Evaluation Complete")
    logger.info("=================================")

    info_list = get_info_from_json()
    for info in info_list:
        if info["Model ID"] == unique_id:
            return info["CEVAL Accuracy"], info["GSM8K Accuracy"], info["Delta L1-Norm"]


def load_config_from_file(unique_id):
    file_path = f"./config/{unique_id}.yaml"
    try:
        with open(file_path, "r") as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        logger.info(f"File not found: {file_path}")
        return None
    except Exception as e:
        logger.info(f"An error occurred while loading the YAML file: {e}")
        return None


def parse_w_from_config(config_name):
    config = load_config_from_file(config_name)
    w = [
        config["models"][1]["parameters"]["density"],
        config["models"][1]["parameters"]["weight"],
        config["models"][2]["parameters"]["density"],
        config["models"][2]["parameters"]["weight"],
        config["models"][3]["parameters"]["density"],
        config["models"][3]["parameters"]["weight"],
    ]
    return w


def convert_all_population_to_X(population):
    return [parse_w_from_config(config_name) for config_name in population]


def conver_one_X_to_config_name(one_of_X):
    if isinstance(one_of_X, torch.Tensor):
        one_of_X = one_of_X.cpu().numpy()
    return "-".join([str(np.round(x, 2)) for x in one_of_X])


def convert_all_X_to_population(X):
    return [conver_one_X_to_config_name(one_of_X) for one_of_X in X]


def save_new_configs(new_x):
    for x in new_x:
        x_np = x.cpu().numpy()
        child_w = x_np.tolist()
        child_w = [0.001 if val == 0 else 0.999 if val == 1 else val for val in child_w]
        new_config = get_dare_ties_config(child_w)
        new_unique_id = "-".join([str(np.round(val, 2)) for val in child_w])
        name = f"{new_unique_id}.yaml"
        local_file_path = f"./config/{name}"
        save_config(new_config, local_file_path)
        logger.info(f"Saved new configuration to {local_file_path}")


def compute_objectives(unique_id):
    ceval_accuracy, gsm8k_accuracy, delta_l1_norm = evaluate_config(unique_id)
    return torch.tensor([ceval_accuracy, gsm8k_accuracy, delta_l1_norm], dtype=torch.float64).to(device)


def z_score_normalize(data):
    mean = data.mean(dim=0, keepdim=True)
    std = data.std(dim=0, keepdim=True)
    return (data - mean) / std


def fisher_information(model, x):
    with torch.no_grad():
        posterior = model.posterior(normalize(x, bounds))
        variances = posterior.variance

    fi = torch.zeros(x.shape[0]).to(device)
    for i in range(variances.shape[1]):
        var = variances[:, i]
        fi += 1 / var
    return fi


def select_next_configuration(model, candidate_x, batch_size):
    fi_values = []
    for x in candidate_x:
        x_reshaped = x.unsqueeze(0)
        fi_value = fisher_information(model, x_reshaped).sum().item()
        fi_values.append((fi_value, x))

    fi_values.sort(key=lambda x: x[0])
    best_configurations = [x[1] for x in fi_values[:batch_size]]
    return torch.stack(best_configurations)


def evolve_configs():

    def initialize_model(train_x, train_y):
        train_x_n = normalize(train_x, bounds)
        train_y_n = z_score_normalize(train_y)

        models = []
        for i in range(NUM_OF_OBJECTIVES):
            models.append(SingleTaskGP(train_x_n, train_y_n[..., i : i + 1]))
        model = ModelListGP(*models)
        mll = SumMarginalLogLikelihood(model.likelihood, model)
        return mll, model

    def optimize_qehvi_and_get_observation(model, train_x, train_obj, sampler, ref_point):
        with torch.no_grad():
            pred = model.posterior(normalize(train_x, bounds)).mean

        partitioning = FastNondominatedPartitioning(ref_point=ref_point, Y=pred)
        acq_func = qExpectedHypervolumeImprovement(
            model=model, ref_point=ref_point, partitioning=partitioning, sampler=sampler
        )

        candidates, _ = optimize_acqf(
            acq_function=acq_func,
            bounds=bounds,
            q=BATCH_SIZE,
            num_restarts=NUM_RESTARTS,
            raw_samples=RAW_SAMPLES,
            options={"batch_limit": 5, "maxiter": 200},
            sequential=True,
        )

        new_x = unnormalize(candidates.detach(), bounds=bounds)

        best_solutions_np = select_best_solutions(train_x.cpu().numpy())
        adjusted_new_x_np = adjust_solution(new_x.cpu().numpy(), best_solutions_np, bounds)
        new_x = torch.tensor(adjusted_new_x_np, dtype=torch.float64).to(device)

        new_x = select_next_configuration(model, new_x, NUM_OF_OFFSPRINGS)
        save_new_configs(new_x)

        new_y = torch.stack([compute_objectives(conver_one_X_to_config_name(x)) for x in new_x])
        return new_x, new_y

    hvs = []

    population = [generate_random_config() for _ in range(INIT_SIZE)]

    train_X = torch.tensor(convert_all_population_to_X(population), dtype=torch.float64).to(device)
    train_Y = torch.stack([compute_objectives(conver_one_X_to_config_name(x)) for x in train_X]).to(device)

    mll, model = initialize_model(train_X, train_Y)

    ref_point = train_Y.min(dim=0).values
    ref_point[ref_point > 0] *= 0.9
    ref_point[ref_point < 0] *= 1.1

    bd = DominatedPartitioning(ref_point=ref_point, Y=train_Y)
    hvs.append(bd.compute_hypervolume().item())
    logger.info(f"\nBatch  0: Hypervolume (qEHVI) = ({hvs[-1]:>4.6f}), ")

    for iteration in range(1, N_BATCH + 1):
        t0 = time.monotonic()
        fit_gpytorch_mll(mll)

        qehvi_sampler = SobolQMCNormalSampler(sample_shape=torch.Size([MC_SAMPLES]))
        new_X, new_Y = optimize_qehvi_and_get_observation(model, train_X, train_Y, qehvi_sampler, ref_point)

        train_X = torch.cat([train_X, new_X])
        train_Y = torch.cat([train_Y, new_Y])

        bd = DominatedPartitioning(ref_point=ref_point, Y=train_Y)
        hvs.append(bd.compute_hypervolume().item())

        mll, model = initialize_model(train_X, train_Y)

        t1 = time.monotonic()
        logger.info(
            f"\nBatch {iteration:>2}: Hypervolume (qEHVI) = "
            f"({hvs[-1]:>4.6f}), time = {t1-t0:>4.2f}."
        )

    final_population = convert_all_X_to_population(train_X)
    logger.info(f"The length of final population: {len(final_population)}")

    best_config = max(final_population, key=lambda config: evaluate_config(config)[0])
    return best_config


def get_maximum_ceval_accuracy_from_json():
    info_list = get_info_from_json()
    return max(info["CEVAL Accuracy"] for info in info_list)


def main():
    global logger, device, bounds, current_dir, merged_path

    time_start = time.time()

    logger = setup_logger()

    os.makedirs("./config", exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bounds = torch.stack([torch.zeros(6), torch.ones(6)]).to(device)

    current_dir = os.path.dirname(os.path.abspath(__file__))
    merged_path = os.path.join(current_dir, "merged")

    best_config = evolve_configs()
    logger.info(yaml.dump(best_config))

    best_config_ceval_accuracy, _, _ = evaluate_config(best_config)
    max_ceval_accuracy_in_json = get_maximum_ceval_accuracy_from_json()

    try:
        assert best_config_ceval_accuracy == max_ceval_accuracy_in_json, (
            "The best config does not have the maximum CEVAL accuracy"
        )
        logger.info(
            f"Assertion passed: The best config {best_config} has the maximum CEVAL accuracy "
            f"of {best_config_ceval_accuracy}"
        )
    except AssertionError as e:
        logger.error(e)
    finally:
        time_end = time.time()
        time_taken = time_end - time_start
        logger.info(f"Time taken: {time_taken} seconds")
        logger.info(f"Time taken: {time_taken / 60} minutes")


if __name__ == "__main__":
    main()