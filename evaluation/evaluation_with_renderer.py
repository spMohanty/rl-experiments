#!/usr/bin/env python
import ray
from ray.rllib.evaluation.metrics import collect_episodes, summarize_episodes


def rogi_rl_eval_with_render(trainer, eval_workers):
    """
    Custom evaluation function that evaluates episodes in
    both human and ascii renderer.
    We only save the custom metrics in this evaluator.
    Arguments:
        trainer (Trainer): trainer class to evaluate.
        eval_workers (WorkerSet): evaluation workers.
    Returns:
        metrics (dict): evaluation metrics dict.
    """

    # We configured 2 eval workers in the training config.
    # This can be done in yaml configs as `evaluation_num_workers: 2`
    worker_1, worker_2 = eval_workers.remote_workers()

    # Set different env settings for each worker. Here we use a fixed config,
    # which also could have been computed in each worker by looking at
    # env_config.worker_index (printed in SimpleCorridor class above).

    # If we run 2 episodes with 2 eval workers
    # Then we save recordings with both human and ascii renderer
    # Also remember to set `monitor: True`
    # Episodes can be set in yaml configs as `evaluation_num_episodes: 2`
    worker_1.foreach_env.remote(lambda env: env.set_renderer("ascii"))
    worker_2.foreach_env.remote(lambda env: env.set_renderer("human"))

    # If we are evaluating more than 2 episodes we need to increase range(1)
    # For e.g. if we are evaluating 10 episodes with 2 workers use range(5)
    for i in range(1):
        print("Custom evaluation round", i)
        # Calling .sample() runs exactly one episode per worker due to how the
        # eval workers are configured.
        ray.get([w.sample.remote() for w in eval_workers.remote_workers()])

    # Collect the accumulated episodes on the workers, and then summarize the
    # episode stats into a metrics dict.
    episodes, _ = collect_episodes(
        remote_workers=eval_workers.remote_workers(), timeout_seconds=99999)
    # You can compute metrics from the episodes manually, or use the
    # convenient `summarize_episodes()` utility:
    metrics = summarize_episodes(episodes)
    # Note that the above two statements are the equivalent of:
    # metrics = collect_metrics(eval_workers.local_worker(),
    #                           eval_workers.remote_workers())

    # Removing all metrics except for the custom metrics from metrics dict.
    metrics = metrics["custom_metrics"]
    # print("Eval Metrics:", metrics)
    return metrics
