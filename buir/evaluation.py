import torch
import numpy as np
import copy


def evaluate(model, data_loader, train_mat, valid_mat, test_mat):

    metrics = {"P10": [], "P20": [], "P50": [], "R10": [], "R20": [], "R50": [], "N10": [], "N20": [], "N50": []}
    eval_results = {"valid": copy.deepcopy(metrics), "test": copy.deepcopy(metrics)}

    u_online, u_target, i_online, i_target = model.get_embedding()
    score_mat_ui = torch.matmul(u_online, i_target.transpose(0, 1))
    score_mat_iu = torch.matmul(u_target, i_online.transpose(0, 1))
    score_mat = score_mat_ui + score_mat_iu

    sorted_mat = torch.argsort(score_mat.cpu(), dim=1, descending=True)

    for test_user, test_item in test_mat:
        sorted_list = list(np.array(sorted_mat[test_user]))

        # Only test
        for mode in ["test"]:
            sorted_list_tmp = []
            if mode == "valid":
                gt_mat_set = set()
                for user, item in valid_mat:
                    if user == test_user:
                        gt_mat_set.add(item)

                already_seen_items = set()
                for user, item in train_mat:
                    if user == test_user:
                        already_seen_items.add(item)
            elif mode == "test":
                gt_mat_set = set()
                for user, item in test_mat:
                    if user == test_user:
                        gt_mat_set.add(item)

                already_seen_items = set()
                for user, item in train_mat:
                    if user == test_user:
                        already_seen_items.add(item)

            for item in sorted_list:
                if item not in already_seen_items:
                    sorted_list_tmp.append(item)
                if len(sorted_list_tmp) == 50:
                    break

            hit_10 = len(set(sorted_list_tmp[:10]) & set(gt_mat_set))
            hit_20 = len(set(sorted_list_tmp[:20]) & set(gt_mat_set))
            hit_50 = len(set(sorted_list_tmp[:50]) & set(gt_mat_set))

            eval_results[mode]["P10"].append(hit_10 / min(10, len(gt_mat_set)))
            eval_results[mode]["P20"].append(hit_20 / min(20, len(gt_mat_set)))
            eval_results[mode]["P50"].append(hit_50 / min(50, len(gt_mat_set)))

            eval_results[mode]["R10"].append(hit_10 / len(gt_mat_set))
            eval_results[mode]["R20"].append(hit_20 / len(gt_mat_set))
            eval_results[mode]["R50"].append(hit_50 / len(gt_mat_set))

            # ndcg
            denom = np.log2(np.arange(2, 10 + 2))
            dcg_10 = np.sum(np.in1d(sorted_list_tmp[:10], list(gt_mat_set)) / denom)
            idcg_10 = np.sum((1 / denom)[: min(len(list(gt_mat_set)), 10)])

            denom = np.log2(np.arange(2, 20 + 2))
            dcg_20 = np.sum(np.in1d(sorted_list_tmp[:20], list(gt_mat_set)) / denom)
            idcg_20 = np.sum((1 / denom)[: min(len(list(gt_mat_set)), 20)])

            denom = np.log2(np.arange(2, 50 + 2))
            dcg_50 = np.sum(np.in1d(sorted_list_tmp[:50], list(gt_mat_set)) / denom)
            idcg_50 = np.sum((1 / denom)[: min(len(list(gt_mat_set)), 50)])

            eval_results[mode]["N10"].append(dcg_10 / idcg_10)
            eval_results[mode]["N20"].append(dcg_20 / idcg_20)
            eval_results[mode]["N50"].append(dcg_50 / idcg_50)

    for mode in ["test"]:
        for topk in [10, 20, 50]:
            eval_results[mode]["P" + str(topk)] = round(np.asarray(eval_results[mode]["P" + str(topk)]).mean(), 4)
            eval_results[mode]["R" + str(topk)] = round(np.asarray(eval_results[mode]["R" + str(topk)]).mean(), 4)
            eval_results[mode]["N" + str(topk)] = round(np.asarray(eval_results[mode]["N" + str(topk)]).mean(), 4)

    return eval_results


def print_eval_results(logger, eval_results):
    for mode in ["test"]:
        for topk in [10, 20, 50]:
            p = eval_results[mode]["P" + str(topk)]
            r = eval_results[mode]["R" + str(topk)]
            n = eval_results[mode]["N" + str(topk)]
            logger.info("{:5s} P@{}: {:.4f}, R@{}: {:.4f}, N@{}: {:.4f}".format(mode.upper(), topk, p, topk, r, topk, n))


def plot_eval_results(plt, EXP_FOLDER, list_eval_results):
    eval_list_results = {"P10": [], "P20": [], "P50": [], "R10": [], "R20": [], "R50": [], "N10": [], "N20": [], "N50": []}
    for mode in ["test"]:
        for topk in [10, 20, 50]:
            p = [eval_result[mode]["P" + str(topk)] for eval_result in list_eval_results]
            r = [eval_result[mode]["R" + str(topk)] for eval_result in list_eval_results]
            n = [eval_result[mode]["N" + str(topk)] for eval_result in list_eval_results]
            eval_list_results["P" + str(topk)] = p
            eval_list_results["R" + str(topk)] = r
            eval_list_results["N" + str(topk)] = n

    for mode in ["test"]:
        for metric_type in ["P", "R", "N"]:
            plt.figure()
            for topk in [10, 20, 50]:
                metric = metric_type + str(topk)
                # TODO mode
                values = eval_list_results[metric]
                plt.plot(range(len(values)), values, label=f"{metric}")
            plt.xlabel("epochs")
            plt.ylabel("value")
            plt.title(f"plot - {metric_type}@X")
            plt.legend()
            plt.savefig(f"{EXP_FOLDER}/metric-plot-{metric_type}.png")
