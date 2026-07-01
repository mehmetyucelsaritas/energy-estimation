# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
from bench_utils import fetech_tf_bench_results


def run_on_android(
    modelpath,
    adb,
    remote_dir="/data/local/tmp/nn_meter_models",
    benchmark_path="/data/local/tmp/benchmark_model",
    taskset="60",
    num_threads=1,
    warm_ups=30,
    num_runs=40,
):
    modelname = modelpath.split("/")[-1]
    remote_model = f"{remote_dir}/{modelname}"

    adb.push_files(modelpath, remote_dir + "/")

    taskset_cmd = f"taskset {taskset}" if taskset else ""
    command = (
        f"{taskset_cmd} {benchmark_path} "
        f"--num_threads={num_threads} "
        f"--warmup_runs={warm_ups} "
        f"--num_runs={num_runs} "
        f"--graph={remote_model}"
    )
    bench_str = adb.run_cmd(command)
    std_ms, avg_ms, footprint = fetech_tf_bench_results(bench_str)

    adb.run_cmd(f"rm -f {remote_model}")
    return std_ms, avg_ms, footprint
