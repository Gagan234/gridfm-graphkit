import argparse
from datetime import datetime
from gridfm_graphkit.cli import main_cli, benchmark_cli


def main():
    parser = argparse.ArgumentParser(
        prog="gridfm_graphkit",
        description="gridfm-graphkit CLI",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    exp_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # ---- TRAIN SUBCOMMAND ----
    train_parser = subparsers.add_parser("train", help="Run training")
    train_parser.add_argument("--config", type=str, required=True)
    train_parser.add_argument("--exp_name", type=str, default=exp_name)
    train_parser.add_argument("--run_name", type=str, default="run")
    train_parser.add_argument("--log_dir", type=str, default="mlruns")
    train_parser.add_argument("--data_path", type=str, default="data")
    train_parser.add_argument(
        "--dataset_wrapper",
        type=str,
        default=None,
        help="Registered name of a dataset wrapper (see DATASET_WRAPPER_REGISTRY), e.g. SharedMemoryCacheDataset",
    )
    train_parser.add_argument(
        "--plugins",
        nargs="*",
        default=[],
        help="Python packages to import for plugin registration, e.g. gridfm_graphkit_ee",
    )
    train_parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="Override data.workers from the YAML config. Use 0 to debug worker crashes.",
    )
    train_parser.add_argument(
        "--dataset_wrapper_cache_dir",
        type=str,
        default=None,
        help="Directory for the dataset wrapper's disk cache.",
    )
    train_parser.add_argument(
        "--profiler",
        type=str,
        default=None,
        choices=["simple", "advanced", "pytorch"],
        help="Enable Lightning profiler.",
    )

    # ---- FINETUNE SUBCOMMAND ----
    finetune_parser = subparsers.add_parser("finetune", help="Run fine-tuning")
    finetune_parser.add_argument("--config", type=str, required=True)
    finetune_parser.add_argument("--model_path", type=str, required=True)
    finetune_parser.add_argument("--exp_name", type=str, default=exp_name)
    finetune_parser.add_argument("--run_name", type=str, default="run")
    finetune_parser.add_argument("--log_dir", type=str, default="mlruns")
    finetune_parser.add_argument("--data_path", type=str, default="data")
    finetune_parser.add_argument(
        "--dataset_wrapper",
        type=str,
        default=None,
        help="Registered name of a dataset wrapper.",
    )
    finetune_parser.add_argument(
        "--plugins",
        nargs="*",
        default=[],
        help="Python packages to import for plugin registration.",
    )
    finetune_parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="Override data.workers from the YAML config.",
    )
    finetune_parser.add_argument(
        "--dataset_wrapper_cache_dir",
        type=str,
        default=None,
        help="Directory for the dataset wrapper's disk cache.",
    )
    finetune_parser.add_argument(
        "--profiler",
        type=str,
        default=None,
        choices=["simple", "advanced", "pytorch"],
        help="Enable Lightning profiler.",
    )

    # ---- EVALUATE SUBCOMMAND ----
    evaluate_parser = subparsers.add_parser(
        "evaluate",
        help="Evaluate model performance",
    )
    evaluate_parser.add_argument("--model_path", type=str, default=None)
    evaluate_parser.add_argument(
        "--normalizer_stats",
        type=str,
        default=None,
        help="Path to normalizer_stats.pt from a training run.",
    )
    evaluate_parser.add_argument("--config", type=str, required=True)
    evaluate_parser.add_argument("--exp_name", type=str, default=exp_name)
    evaluate_parser.add_argument("--run_name", type=str, default="run")
    evaluate_parser.add_argument("--log_dir", type=str, default="mlruns")
    evaluate_parser.add_argument("--data_path", type=str, default="data")
    evaluate_parser.add_argument(
        "--dataset_wrapper",
        type=str,
        default=None,
        help="Registered name of a dataset wrapper.",
    )
    evaluate_parser.add_argument(
        "--plugins",
        nargs="*",
        default=[],
        help="Python packages to import for plugin registration.",
    )
    evaluate_parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="Override data.workers from the YAML config.",
    )
    evaluate_parser.add_argument(
        "--dataset_wrapper_cache_dir",
        type=str,
        default=None,
        help="Directory for the dataset wrapper's disk cache.",
    )
    evaluate_parser.add_argument(
        "--profiler",
        type=str,
        default=None,
        choices=["simple", "advanced", "pytorch"],
        help="Enable Lightning profiler.",
    )
    evaluate_parser.add_argument(
        "--compute_dc_ac_metrics",
        action="store_true",
    )
    evaluate_parser.add_argument(
        "--save_output",
        action="store_true",
    )

    # ---- PREDICT SUBCOMMAND ----
    predict_parser = subparsers.add_parser("predict", help="Run prediction")
    predict_parser.add_argument("--model_path", type=str, required=False)
    predict_parser.add_argument("--normalizer_stats", type=str, default=None)
    predict_parser.add_argument("--config", type=str, required=True)
    predict_parser.add_argument("--exp_name", type=str, default=exp_name)
    predict_parser.add_argument("--run_name", type=str, default="run")
    predict_parser.add_argument("--log_dir", type=str, default="mlruns")
    predict_parser.add_argument("--data_path", type=str, default="data")
    predict_parser.add_argument("--dataset_wrapper", type=str, default=None)
    predict_parser.add_argument("--plugins", nargs="*", default=[])
    predict_parser.add_argument("--num_workers", type=int, default=None)
    predict_parser.add_argument("--dataset_wrapper_cache_dir", type=str, default=None)
    predict_parser.add_argument("--output_path", type=str, default="data")
    predict_parser.add_argument(
        "--profiler",
        type=str,
        default=None,
        choices=["simple", "advanced", "pytorch"],
    )

    # ---- BENCHMARK SUBCOMMAND ----
    benchmark_parser = subparsers.add_parser(
        "benchmark",
        help="Benchmark train-dataloader iteration speed",
    )
    benchmark_parser.add_argument("--config", type=str, required=True)
    benchmark_parser.add_argument("--data_path", type=str, default="data")
    benchmark_parser.add_argument("--epochs", type=int, default=3)
    benchmark_parser.add_argument("--dataset_wrapper", type=str, default=None)
    benchmark_parser.add_argument("--dataset_wrapper_cache_dir", type=str, default=None)
    benchmark_parser.add_argument("--num_workers", type=int, default=None)
    benchmark_parser.add_argument("--plugins", nargs="*", default=[])

    args = parser.parse_args()

    if args.command == "benchmark":
        benchmark_cli(args)
    else:
        main_cli(args)


if __name__ == "__main__":
    main()