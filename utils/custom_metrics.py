#!/usr/bin/env python
import numpy as np
# Custom Callbacks


def on_episode_start(info):
    episode = info["episode"]
    # print("episode {} started".format(episode.episode_id))
    episode.user_data["ActionTypes"] = []
    episode.hist_data["ActionTypes"] = []
    episode.user_data["X"] = []
    episode.hist_data["X"] = []
    episode.user_data["Y"] = []
    episode.hist_data["Y"] = []


def on_episode_step(info):
    episode = info["episode"]
    actions = episode.last_action_for()
    episode.user_data["ActionTypes"].append(actions[0])
    episode.user_data["X"].append(int(actions[1]))
    episode.user_data["Y"].append(int(actions[2]))


def on_episode_end(info):
    """
    Gathers all numeric values in the
    episode info object, and
    adds them as custom metrics
    """
    episode = info['episode']
    info_object = episode.last_info_for()

    episode.hist_data["ActionTypes"] = episode.user_data["ActionTypes"]
    episode.hist_data["X"] = episode.user_data["X"]
    episode.hist_data["Y"] = episode.user_data["Y"]

    for _key in info_object.keys():
        try:
            float(info_object[_key])
            episode.custom_metrics[_key] = float(info_object[_key])
        except ValueError:
            """
            We will receive a ValueError in case of non-numeric keys
            """
            pass
    episode.custom_metrics["VaccinationStepRatio"] = np.mean(
        episode.hist_data["ActionTypes"])
