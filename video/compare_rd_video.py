# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse
import json
import os
from typing import Dict
import warnings
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from src.metrics.bjontegaard_metric import calculate_bd_rate
from src.utils.common import create_folder


def parse_args():
    parser = argparse.ArgumentParser(description="RD comparison for video codec models")

    parser.add_argument(
        "--compare_between",
        type=str,
        default="class",
        choices=["class", "sequence"],
        help="compare the performance between different classes/sequences",
    )
    parser.add_argument(
        "--compare_frame_type",
        type=str,
        default="default",
        choices=["default", "all", "i", "p"],
        help="frame type to compare BD-Rate, default is i, p, and all",
    )
    parser.add_argument("--base_method", type=str, required=True, help="name of the anchor model")
    parser.add_argument(
        "--log_paths",
        type=str,
        required=True,
        nargs="+",
        help="list of model test result paths, model name followed by file path",
    )
    parser.add_argument(
        "--output_path", type=str, default="stdout", help="print the results to console or save to file; TXT or CSV"
    )
    parser.add_argument("--plot_path", type=str, default=None, help="path to save the plots")
    parser.add_argument(
        "--plot_scheme", type=str, default=None, choices=[None, "combined", "separate"], help="RD curve plot scheme"
    )
    parser.add_argument(
        "--distortion_metrics",
        type=str,
        nargs="+",
        default=["psnr"],
        choices=[
            "psnr",
            "psnr_rgb",
            "psnr_y",
            "psnr_u",
            "psnr_v",
            "msssim",
            "msssim_rgb",
            "msssim_y",
            "msssim_u",
            "msssim_v",
            "vif",
            "vif_scale_0",
            "vif_scale_1",
            "psnr_roi",
            "psnr_inv_roi",
            "lpips",
            "deqa_score_rgb",
        ],
        help="distortion metrics used to calculate BD-Rate and plot",
    )
    parser.add_argument("--output_plot_data", type=str, default=None, help="if given, save the plot data to csv file")
    parser.add_argument("--plot_rd_curve", type=int, default=1, choices=[0, 1], help="if 1, plot RD curves")
    parser.add_argument(
        "--auto_test", type=int, default=0, choices=[0, 1], help="configurations for auto testing on AML"
    )
    parser.add_argument(
        "--rate_exclude", type=str, required=False, nargs="+", default=None, help="list of rate points to be excluded"
    )

    args = parser.parse_args()

    return args


def _matplotlib_plt(data_dict, out_path, ds_name=None, distortion_metric="psnr"):
    plt.figure()
    for key in data_dict:
        plt.plot(data_dict[key]["bpp"], data_dict[key][distortion_metric], label=key)
        plt.scatter(data_dict[key]["bpp"], data_dict[key][distortion_metric])

    if ds_name is not None:
        plt.gca().set_title(ds_name)
    plt.grid(True)
    plt.legend(loc="lower right")
    plt.savefig(out_path)
    plt.close("all")


def matplotlib_plt(all_dataset_names, data_dict, out_prefix, distortion_metric, plot_scheme="combined"):
    subplot_dicts = []
    for ds_name in all_dataset_names:
        ds_data_dict = {}
        for key, value in data_dict.items():
            if ds_name in value:
                ds_data_dict[key] = value[ds_name]
        if plot_scheme == "separate":
            _matplotlib_plt(
                ds_data_dict,
                f"{out_prefix}_{distortion_metric}_{ds_name}.png",
                ds_name=ds_name,
                distortion_metric=distortion_metric,
            )
        else:
            subplot_dicts.append({"data_dict": ds_data_dict, "ds_name": ds_name})

    if plot_scheme == "combined":
        fig, axs = plt.subplots(1, len(subplot_dicts), figsize=(5 * len(subplot_dicts), 4))
        if len(subplot_dicts) == 1:
            axs = [axs]
        for ax, d in zip(axs, subplot_dicts):
            data_dict = d["data_dict"]
            ds_name = d["ds_name"]
            for k in data_dict:
                ax.plot(data_dict[k]["bpp"], data_dict[k][distortion_metric], label=k)
                ax.scatter(data_dict[k]["bpp"], data_dict[k][distortion_metric])

            ax.grid(True)
            ax.legend(loc="lower right")
            ax.set_title(f"{ds_name}, metric {distortion_metric}")
            ax.set_xlim(left=0.0)
            # ax.set_xlim([0.0, 0.005])
            # ax.set_ylim([31, 42])

        fig.tight_layout()
        fig.savefig(out_prefix + "_" + distortion_metric + ".png")


def rotate_results(ds_names, results, print_overall=False):
    new_ds_names = set()
    new_results = {}
    for ds in ds_names:
        new_results[ds] = {}
        for m in results:
            if ds in results[m]:
                new_results[ds][m] = results[m][ds]
            new_ds_names.add(m)
    if print_overall:
        new_results["* Overall"] = {}
        new_results["* Average"] = {}
        for m in results:
            if "* Overall" in results[m]:
                new_results["* Overall"][m] = results[m]["* Overall"]
                new_results["* Average"][m] = results[m]["* Average"]
    return list(new_ds_names), new_results


def avg_results(all_dataset_names, results):
    all_bd_rates = {}
    for method in sorted(list(results.keys())):
        for ds_name in all_dataset_names:
            if method not in all_bd_rates:
                all_bd_rates[method] = []
            if ds_name in results[method]:
                all_bd_rates[method].append(results[method][ds_name])
    avg_bd_rates = {}
    for key, value in all_bd_rates.items():
        if len(value) > 0:
            avg_bd_rates[key] = np.mean(value)
    return avg_bd_rates


def get_results(all_dataset_names, results, rotate=False, print_overall=False):
    if len(results.keys()) == 0:
        return

    if rotate:
        # after rotation, the meaning of all_dataset_names and results are exchanged
        all_dataset_names, results = rotate_results(all_dataset_names, results, print_overall)

    all_dataset_names = sorted([x for x in all_dataset_names if not x.startswith("*")])
    all_method_names = sorted([x for x in results.keys() if not x.startswith("*")])

    if print_overall and rotate:
        all_method_names += ["* Overall", "* Average"]
    elif print_overall:
        all_dataset_names += ["* Overall", "* Average"]

    output = {}
    for method in all_method_names:
        output[method] = {}
        for ds_name in all_dataset_names:
            value = results[method].get(ds_name)
            if value is not None:
                output[method][ds_name] = value

    return output


def print_results_different_metric(all_dataset_names, all_sequence_names, seq_results, results):
    for ds in sorted(all_dataset_names):
        print("-" * 4, ds, "-" * 4)
        print_seq_results = {}
        avg_bd_rates = avg_results(all_sequence_names[ds], seq_results)
        for m in sorted(list(results.keys())):
            if ds in results[m]:
                print_seq_results[m] = {}
                for seq in all_sequence_names[ds]:
                    if seq in seq_results[m]:
                        print_seq_results[m][seq] = seq_results[m][seq]
                    print_seq_results[m]["* Overall"] = results[m][ds]
                    print_seq_results[m]["* Average"] = avg_bd_rates[m]

        get_results(all_sequence_names[ds], print_seq_results, rotate=True, print_overall=True)
    print()


def mean_over_rate_point(rate_point, distortion_metric):
    i_frame_num = 0
    p_frame_num = 0
    i_frame_bpp = 0.0
    i_frame_dist = 0.0
    p_frame_bpp = 0.0
    p_frame_dist = 0.0
    all_frame_bpp = 0.0
    all_frame_dist = 0.0
    i_frame_dist_key = f"ave_i_frame_{distortion_metric}"
    p_frame_dist_key = f"ave_p_frame_{distortion_metric}"
    all_frame_dist_key = f"ave_all_frame_{distortion_metric}"
    for seq in rate_point:
        if i_frame_dist_key not in seq:
            seq[i_frame_dist_key] = 0
        if p_frame_dist_key not in seq:
            seq[p_frame_dist_key] = 0
        if all_frame_dist_key not in seq:
            seq[all_frame_dist_key] = 0

        i_frame_num += seq["i_frame_num"]
        p_frame_num += seq["p_frame_num"]

        i_frame_bpp += seq["ave_i_frame_bpp"] * seq["i_frame_num"]
        i_frame_dist += seq[i_frame_dist_key] * seq["i_frame_num"]

        p_frame_bpp += seq["ave_p_frame_bpp"] * seq["p_frame_num"]
        p_frame_dist += seq[p_frame_dist_key] * seq["p_frame_num"]

        all_frame_bpp += seq["ave_all_frame_bpp"] * (seq["p_frame_num"] + seq["i_frame_num"])
        all_frame_dist += seq[all_frame_dist_key] * (seq["p_frame_num"] + seq["i_frame_num"])

    out_res = {}
    out_res["i_frame_num"] = i_frame_num
    out_res["p_frame_num"] = p_frame_num

    all_frame_num = i_frame_num + p_frame_num
    i_frame_num = 1 if i_frame_num == 0 else i_frame_num
    p_frame_num = 1 if p_frame_num == 0 else p_frame_num

    out_res["ave_i_frame_bpp"] = i_frame_bpp / i_frame_num
    out_res[i_frame_dist_key] = i_frame_dist / i_frame_num
    out_res["ave_p_frame_bpp"] = p_frame_bpp / p_frame_num
    out_res[p_frame_dist_key] = p_frame_dist / p_frame_num
    out_res["ave_all_frame_bpp"] = all_frame_bpp / all_frame_num
    out_res[all_frame_dist_key] = all_frame_dist / all_frame_num
    return out_res


def mean_over_sequence(res, distortion_metric):
    new_res = {}  # model -> dataset -> [models]
    for m in res:
        new_res[m] = {}
        for d in res[m]:
            rate_points = {}
            for s in res[m][d]:
                for rate_point, value in res[m][d][s].items():
                    if rate_point in rate_points:
                        rate_points[rate_point].append(value)
                    else:
                        rate_points[rate_point] = [value]
            new_res[m][d] = {}
            for rate_point, value in rate_points.items():
                new_res[m][d][rate_point] = mean_over_rate_point(value, distortion_metric)
    return new_res


def flatten_test_results(res):
    new_res = {}
    for key_method in res:
        new_res[key_method] = {}
        for ds_name in res[key_method]:
            for seq in res[key_method][ds_name]:
                ds_seq_name = seq
                new_res[key_method][ds_seq_name] = res[key_method][ds_name][seq]
    return new_res


class RDTester:
    def __init__(
        self,
        args: argparse.Namespace,
    ):

        if (args.auto_test == 1) and (args.plot_rd_curve == 1):
            args.plot_rd_curve = 0
            warnings.warn("Plotting RD curves is disabled during auto test")

        base_method_name = args.base_method
        assert len(args.log_paths) % 2 == 0, (
            "log paths shoud include both the method name and the corresponding log path"
        )
        log_paths = {}
        for i in range(len(args.log_paths) // 2):
            log_paths[args.log_paths[2 * i]] = args.log_paths[2 * i + 1]
        assert base_method_name in log_paths, "log paths must include the base method"

        assert args.output_plot_data is None or args.output_plot_data[-4:] in [
            ".csv",
            ".CSV",
        ], "only CSV format supported for plot data output"

        if args.plot_scheme is None:
            args.plot_scheme = "combined" if args.compare_between == "class" else "separate"

        if args.plot_path is not None:
            create_folder(args.plot_path)

        self.base_method_name: str = base_method_name
        self.log_paths: Dict[str, str] = log_paths

        self.compare_frame_type = args.compare_frame_type
        self.compare_between = args.compare_between
        self.distortion_metrics = args.distortion_metrics
        self.rate_exclude = args.rate_exclude

        self.plot_rd_curve = args.plot_rd_curve
        self.output_plot_data = args.output_plot_data
        self.plot_scheme = args.plot_scheme
        self.plot_path = args.plot_path
        self.output_path = args.output_path
        self.auto_test = args.auto_test

        self.results = {}

        self.ds_seq_names = None

    def parse_logs(self):
        json_dict = {}  # model -> dataset -> seq -> [models]
        ds_seq_names = {}
        seq_consistency = True
        for key_method, json_file in self.log_paths.items():
            json_dict[key_method] = {}
            with open(json_file) as f:
                json_data = json.load(f)
                for ds_name in json_data:
                    json_dict[key_method][ds_name] = {}
                    if ds_name not in ds_seq_names:
                        ds_seq_names[ds_name] = set(json_data[ds_name].keys())
                    else:
                        if ds_seq_names[ds_name] != set(json_data[ds_name].keys()):
                            seq_consistency = False
                    for seq in json_data[ds_name]:
                        json_dict[key_method][ds_name][seq] = {}
                        rate_points = list(json_data[ds_name][seq].keys())
                        seq_bpps = [json_data[ds_name][seq][r_p]["ave_all_frame_bpp"] for r_p in rate_points]
                        rate_points = sorted(rate_points, key=lambda x: seq_bpps[rate_points.index(x)])
                        for r_p_idx, rate_point in enumerate(rate_points):
                            if self.rate_exclude is not None and rate_point in self.rate_exclude:
                                continue
                            json_dict[key_method][ds_name][seq][r_p_idx] = json_data[ds_name][seq][rate_point]
        if not seq_consistency:
            warnings.warn("inconsistency found in the sequences tested in each dataset")

        self.ds_seq_names = ds_seq_names

        return json_dict

    def retrieve_data(self, json_dict, frame_type, base_method_name, distortion_metric):
        max_num_bitrate = 0
        data_dict = {}
        for key_method in json_dict:
            data_dict[key_method] = {}
            for ds_name in json_dict[key_method]:
                data_dict[key_method][ds_name] = {}
                data_dict[key_method][ds_name]["bpp"] = []
                data_dict[key_method][ds_name][distortion_metric] = []
                for _, one_data in json_dict[key_method][ds_name].items():
                    data_dict[key_method][ds_name]["bpp"].append(one_data[f"ave_{frame_type}_frame_bpp"])
                    val = one_data[f"ave_{frame_type}_frame_{distortion_metric}"]
                    if distortion_metric == "lpips":
                        val = 1 - val
                    data_dict[key_method][ds_name][distortion_metric].append(val)
                max_num_bitrate = max(max_num_bitrate, len(data_dict[key_method][ds_name]["bpp"]))

        results = {}
        results[distortion_metric] = {}
        for key_method in json_dict:
            if key_method == base_method_name:
                continue
            results[distortion_metric][key_method] = {}
            for ds_name in json_dict[key_method]:
                assert ds_name in data_dict[base_method_name], f"{ds_name} not found in {base_method_name} dict"
                assert len(data_dict[key_method][ds_name]["bpp"]) >= 3, (
                    "At least 3 rate points are required for BD rate!"
                )
                assert data_dict[base_method_name][ds_name]["bpp"][0] >= 0, "Bitrate must be positive!"
                assert data_dict[key_method][ds_name][distortion_metric][0] is not None, "Distortion metric must exist!"
                assert data_dict[key_method][ds_name][distortion_metric][0] > 0, "Distortion metric must be positive!"

                results[distortion_metric][key_method][ds_name] = calculate_bd_rate(
                    data_dict[base_method_name][ds_name]["bpp"],
                    data_dict[base_method_name][ds_name][distortion_metric],
                    data_dict[key_method][ds_name]["bpp"],
                    data_dict[key_method][ds_name][distortion_metric],
                    piecewise=1,
                )

        return data_dict, results, max_num_bitrate

    def main(self):

        json_dict = self.parse_logs()
        assert self.ds_seq_names is not None
        all_dataset_names = sorted(list(self.ds_seq_names.keys()))

        for distortion_metric in self.distortion_metrics:
            self.results[distortion_metric] = {}

            cls_result_dict = mean_over_sequence(json_dict, distortion_metric)
            all_dataset_names = sorted(list(self.ds_seq_names.keys()))
            all_sequence_names = {}
            if self.compare_between == "sequence":
                seq_json_dict = flatten_test_results(json_dict)
                for key, value in self.ds_seq_names.items():
                    all_sequence_names[key] = []
                    for seq in value:
                        all_sequence_names[key].append(seq)

            if self.compare_frame_type == "default":
                frame_types = ["i", "p", "all"]
            else:
                frame_types = [self.compare_frame_type]

            for frame_type in frame_types:
                frame_data, results_list, max_num_bitrate = self.retrieve_data(
                    cls_result_dict, frame_type, self.base_method_name, distortion_metric
                )
                if self.compare_between == "sequence":
                    seq_frame_data, seq_results_list, seq_max_num_bitrate = self.retrieve_data(
                        seq_json_dict, frame_type, self.base_method_name, distortion_metric
                    )

                    print_results_different_metric(
                        all_dataset_names,
                        all_sequence_names,
                        seq_results_list[distortion_metric],
                        results_list[distortion_metric],
                    )
                    frame_data, results_list[distortion_metric], max_num_bitrate = (
                        seq_frame_data,
                        seq_results_list[distortion_metric],
                        seq_max_num_bitrate,
                    )
                else:
                    results = get_results(all_dataset_names, results_list[distortion_metric])
                    self.results[distortion_metric][frame_type] = results

                if self.plot_rd_curve:
                    self.plot_curve(frame_data, all_sequence_names, all_dataset_names, distortion_metric, frame_type)

                if self.output_plot_data:
                    self.plot_output(frame_data, distortion_metric, max_num_bitrate)

        if self.output_path == "stdout":
            print(self.results)
        else:
            write_mode = "a" if self.auto_test == 1 else "w"
            with open(self.output_path, write_mode) as f:
                json.dump(self.results, f)
                f.write("\n")

    def plot_curve(self, frame_data, all_sequence_names, all_dataset_names, distortion_metric, frame_type):
        matplotlib.use("Agg")
        plt.rcParams.update({"grid.color": "0.5", "grid.linewidth": 0.5, "savefig.dpi": 300})

        if self.compare_between == "sequence":
            names = []
            for _, value in all_sequence_names.items():
                names += value
        else:
            names = all_dataset_names
        if self.plot_scheme == "combined" and len(names) > 7:
            warnings.warn("plotting in combined mode with more than 7 datasets/sequences is not supported")
        else:
            matplotlib_plt(
                names,
                frame_data,
                os.path.join(self.plot_path, f"{frame_type}_frame"),
                distortion_metric,
                plot_scheme=self.plot_scheme,
            )

    def plot_output(self, frame_data, distortion_metric, max_num_bitrate):
        try:
            import pandas as pd  # pylint: disable=C0415
        except ImportError:
            raise ImportError("saving plot data to CSV requires Pandas library")
        all_dfs = {}
        models = sorted(list(frame_data.keys()))
        for m in models:
            for d in frame_data[m]:
                if d + "_all_frame" not in all_dfs:
                    all_dfs[d + "_all_frame"] = pd.DataFrame(columns=[d + "_all_frame"])  # type: ignore[reportArgumentType]
                    all_dfs[d + "_all_frame"][d + "_all_frame"] = [x + 1 for x in range(max_num_bitrate)]
                all_dfs[d + "_all_frame"][m + "_bpp"] = pd.Series(frame_data[m][d]["bpp"])
                all_dfs[d + "_all_frame"][m + "_" + distortion_metric] = pd.Series(frame_data[m][d][distortion_metric])

        for key, value in all_dfs.items():
            with open(self.output_plot_data, "a") as f:
                value.to_csv(f)


if __name__ == "__main__":
    args = parse_args()
    tester = RDTester(args)
    tester.main()
