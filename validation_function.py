import torch
from torch import Tensor

import numpy as np
from collections import OrderedDict, defaultdict
from l5kit.evaluation.csv_utils import read_gt_csv, read_pred_csv, write_gt_csv, write_pred_csv
from torch.utils.data import DataLoader, Sampler
from l5kit.geometry import transform_points
from torch.utils.data.sampler import SubsetRandomSampler
from tqdm import tqdm
from l5kit.evaluation.metrics import neg_multi_log_likelihood, time_displace

def convert_agent_coordinates_to_world_offsets(
    agents_coords: np.ndarray,
    world_from_agents: np.ndarray,
    centroids: np.ndarray,
) -> np.ndarray:
    coords_offset = []
    for agent_coords, world_from_agent, centroid in zip(
        agents_coords, world_from_agents, centroids
    ):
        predition_offset = []
        for agent_coord in agent_coords:
            predition_offset.append(
                transform_points(agent_coord, world_from_agent) - centroid[:2]
            )
        predition_offset = np.stack(predition_offset)
        coords_offset.append(predition_offset)
    return np.stack(coords_offset)

def validation(eval_gt_path, model, eval_dataloader):


    # store information for evaluation
    future_coords_offsets_pd = []
    timestamps = []
    confidences_list = []
    agent_ids = []
    target_ls = []
    avails_ids = []
    progress_bar = tqdm(eval_dataloader)

    for data in progress_bar:

        # Forward pass
        preds, confidences = model(data)
        # convert agent coordinates into world offsets
        preds = convert_agent_coordinates_to_world_offsets(
            preds.detach().cpu().numpy(),
            data["world_from_agent"].numpy(),
            data["centroid"].numpy(),
        )

        future_coords_offsets_pd.append(preds)
        confidences_list.append(confidences.detach().cpu().numpy())
        timestamps.append(data["timestamp"].detach().numpy())
        agent_ids.append(data["track_id"].detach().numpy())

    # create submission to submit to Kaggle
    pred_path = 'submission.csv'
    write_pred_csv(pred_path, timestamps=np.concatenate(timestamps), track_ids=np.concatenate(agent_ids),
                   coords=np.concatenate(future_coords_offsets_pd), confs=np.concatenate(confidences_list))

    ground_truth_path = eval_gt_path
    ground_truth = OrderedDict()
    inference = OrderedDict()

    for el in read_gt_csv(ground_truth_path):
        ground_truth[el["track_id"] + el["timestamp"]] = el
    for el in read_pred_csv(pred_path):
        inference[el["track_id"] + el["timestamp"]] = el

    metrics = [neg_multi_log_likelihood] #, time_displace
    metrics_dict = defaultdict(list)

    for key, ground_truth_value in ground_truth.items():
        gt_coord = ground_truth_value["coord"]
        avail = ground_truth_value["avail"]

        # we subsampled the eval datset -> not every timestamp is available
        if key in inference:
            pred_coords = inference[key]["coords"]
            conf = inference[key]["conf"]
            for metric in metrics:
                metrics_dict[metric.__name__].append(metric(gt_coord, pred_coords, conf, avail))
    return metrics_dict