# import warnings
# warnings.filterwarnings('ignore')
# import os
# import numpy as np
# from prettytable import PrettyTable
# from ultralytics import YOLO
# from ultralytics.utils.torch_utils import model_info
#
# def get_weight_size(path):
#     stats = os.stat(path)
#     return f'{stats.st_size / 1024 / 1024:.1f}'
#
# if __name__ == '__main__':
#     model_path = 'D:\\python_space\\ultralytics-yolo11-main\\runs\\train\\exp13\\weights\\best.pt'
#     model = YOLO(model_path) # 选择训练好的权重路径
#     result = model.val(data='D:\\python_space\\ultralytics-yolo11-main\\Dense_dataset_experiment\\dataset\\data_dense.yaml',
#                         split='test', # split可以选择train、val、test 根据自己的数据集情况来选择.
#                         imgsz=640,
#                         batch=8,
#                         # iou=0.7,
#                         # rect=False,
#                         # save_json=True, # if you need to cal coco metrice
#                         project='runs/val',
#                         name='exp',
#                         )
#
#     if model.task == 'detect': # 仅目标检测任务适用 需要改别的任务可以看：https://www.bilibili.com/video/BV1dBQDY6Ec5/
#         length = result.box.p.size
#         model_names = list(result.names.values())
#         preprocess_time_per_image = result.speed['preprocess']
#         inference_time_per_image = result.speed['inference']
#         postprocess_time_per_image = result.speed['postprocess']
#         all_time_per_image = preprocess_time_per_image + inference_time_per_image + postprocess_time_per_image
#
#         n_l, n_p, n_g, flops = model_info(model.model)
#
#         print('-'*20 + '论文上的数据以以下结果为准' + '-'*20)
#         print('-'*20 + '论文上的数据以以下结果为准' + '-'*20)
#         print('-'*20 + '论文上的数据以以下结果为准' + '-'*20)
#         print('-'*20 + '论文上的数据以以下结果为准' + '-'*20)
#         print('-'*20 + '论文上的数据以以下结果为准' + '-'*20)
#
#         model_info_table = PrettyTable()
#         model_info_table.title = "Model Info"
#         model_info_table.field_names = ["GFLOPs", "Parameters", "前处理时间/一张图", "推理时间/一张图", "后处理时间/一张图", "FPS(前处理+模型推理+后处理)", "FPS(推理)", "Model File Size"]
#         model_info_table.add_row([f'{flops:.1f}', f'{n_p:,}',
#                                   f'{preprocess_time_per_image / 1000:.6f}s', f'{inference_time_per_image / 1000:.6f}s',
#                                   f'{postprocess_time_per_image / 1000:.6f}s', f'{1000 / all_time_per_image:.2f}',
#                                   f'{1000 / inference_time_per_image:.2f}', f'{get_weight_size(model_path)}MB'])
#         print(model_info_table)
#
#         model_metrice_table = PrettyTable()
#         model_metrice_table.title = "Model Metrice"
#         model_metrice_table.field_names = ["Class Name", "Precision", "Recall", "F1-Score", "mAP50", "mAP75", "mAP50-95"]
#         for idx in range(length):
#             model_metrice_table.add_row([
#                                         model_names[idx],
#                                         f"{result.box.p[idx]:.4f}",
#                                         f"{result.box.r[idx]:.4f}",
#                                         f"{result.box.f1[idx]:.4f}",
#                                         f"{result.box.ap50[idx]:.4f}",
#                                         f"{result.box.all_ap[idx, 5]:.4f}", # 50 55 60 65 70 75 80 85 90 95
#                                         f"{result.box.ap[idx]:.4f}"
#                                     ])
#         model_metrice_table.add_row([
#                                     "all(平均数据)",
#                                     f"{result.results_dict['metrics/precision(B)']:.4f}",
#                                     f"{result.results_dict['metrics/recall(B)']:.4f}",
#                                     f"{np.mean(result.box.f1[:length]):.4f}",
#                                     f"{result.results_dict['metrics/mAP50(B)']:.4f}",
#                                     f"{np.mean(result.box.all_ap[:length, 5]):.4f}", # 50 55 60 65 70 75 80 85 90 95
#                                     f"{result.results_dict['metrics/mAP50-95(B)']:.4f}"
#                                 ])
#         print(model_metrice_table)
#
#         with open(result.save_dir / 'paper_data.txt', 'w+', errors="ignore", encoding="utf-8") as f:
#             f.write(str(model_info_table))
#             f.write('\n')
#             f.write(str(model_metrice_table))
#
#         print('-'*20, f'结果已保存至{result.save_dir}/paper_data.txt...', '-'*20)
#         print('-'*20, f'结果已保存至{result.save_dir}/paper_data.txt...', '-'*20)
#         print('-'*20, f'结果已保存至{result.save_dir}/paper_data.txt...', '-'*20)
#         print('-'*20, f'结果已保存至{result.save_dir}/paper_data.txt...', '-'*20)
#         print('-'*20, f'结果已保存至{result.save_dir}/paper_data.txt...', '-'*20)


#
#
# '''
# 密集和不密集数据集的测试
# '''
# import warnings
# warnings.filterwarnings('ignore')
#
# import os
# import numpy as np
# from prettytable import PrettyTable
# from ultralytics import YOLO
# from ultralytics.utils.torch_utils import model_info
#
#
# def get_weight_size(path):
#     stats = os.stat(path)
#     return f'{stats.st_size / 1024 / 1024:.1f}'
#
#
# def run_val(model, data_yaml, run_name):
#     result = model.val(
#         data=data_yaml,
#         split='test',
#         imgsz=640,
#         batch=8,
#         conf=0.25,
#         project=r'D:\python_space\ultralytics-yolo11-main-origin\runs\dense_eval2',
#         name=run_name,
#         save_json=False,
#         save=True,
#         plots=True,
#         show_conf=False,
#         save_txt=False,
#     )
#     return result
#
#
# def safe_div(a, b):
#     return a / b if b != 0 else 0.0
#
#
# def calc_det_acc_from_pr(p, r, n_gt):
#     """
#     自定义检测ACC:
#     ACC = TP / (TP + FP + FN)
#     """
#     tp = r * n_gt
#     fp = tp * (1.0 / p - 1.0) if p > 0 else 0.0
#     fn = n_gt - tp
#     return safe_div(tp, tp + fp + fn)
#
#
# def extract_metrics(result):
#     class_names = list(result.names.values())
#     num_classes = result.box.p.size
#
#     per_class = []
#     for i in range(num_classes):
#         p = float(result.box.p[i])
#         r = float(result.box.r[i])
#         f1 = float(result.box.f1[i])
#         map50 = float(result.box.ap50[i])
#         map75 = float(result.box.all_ap[i, 5])   # IoU=0.75
#         map5095 = float(result.box.ap[i])
#
#         per_class.append({
#             "Class": class_names[i],
#             "ACC": 0.0,   # 先占位，或者改成 "N/A"
#             "Precision": p,
#             "Recall": r,
#             "F1_score": f1,
#             "mAP50": map50,
#             "mAP75": map75,
#             "mAP50-95": map5095,
#         })
#
#     p_all = float(result.results_dict['metrics/precision(B)'])
#     r_all = float(result.results_dict['metrics/recall(B)'])
#     f1_all = safe_div(2 * p_all * r_all, p_all + r_all)
#
#     overall = {
#         "Class": "all",
#         "ACC": 0.0,
#         "Precision": p_all,
#         "Recall": r_all,
#         "F1_score": f1_all,
#         "mAP50": float(result.results_dict['metrics/mAP50(B)']),
#         "mAP75": float(np.mean(result.box.all_ap[:num_classes, 5])),
#         "mAP50-95": float(result.results_dict['metrics/mAP50-95(B)']),
#     }
#
#     return per_class, overall
#
# def summarize_dicts(dict_list, class_name):
#     metrics_keys = ["ACC", "Precision", "Recall", "F1_score", "mAP50", "mAP75", "mAP50-95"]
#
#     summary = {"Class": class_name}
#     for k in metrics_keys:
#         vals = np.array([d[k] for d in dict_list], dtype=float)
#         mean_v = np.mean(vals)
#         std_v = np.std(vals, ddof=1) if len(vals) > 1 else 0.0
#         summary[k] = f"{mean_v:.4f} ± {std_v:.4f}"
#     return summary
#
#
# def dump_model_info_table(model, model_path):
#     n_l, n_p, n_g, flops = model_info(model.model)
#
#     table = PrettyTable()
#     table.title = "Model Info"
#     table.field_names = ["GFLOPs", "Parameters", "Model Size"]
#     table.add_row([
#         f'{flops:.1f}',
#         f'{n_p:,}',
#         f'{get_weight_size(model_path)}MB'
#     ])
#     return table
#
#
# def dump_summary_table(summary_rows, title):
#     table = PrettyTable()
#     table.title = title
#     table.field_names = ["Class", "ACC", "Precision", "Recall", "F1_score", "mAP50", "mAP75", "mAP50-95"]
#
#     for row in summary_rows:
#         table.add_row([
#             row["Class"],
#             row["ACC"],
#             row["Precision"],
#             row["Recall"],
#             row["F1_score"],
#             row["mAP50"],
#             row["mAP75"],
#             row["mAP50-95"],
#         ])
#     return table
#
#
# def evaluate_three_models(model_paths, data_yaml, dataset_tag):
#     all_runs_per_class = []
#     all_runs_overall = []
#
#     for idx, model_path in enumerate(model_paths, start=1):
#         model = YOLO(model_path)
#         run_name = f"{dataset_tag}_run{idx}"
#
#         result = run_val(model, data_yaml, run_name)
#         per_class, overall = extract_metrics(result)
#
#         all_runs_per_class.append(per_class)
#         all_runs_overall.append(overall)
#
#     num_classes = len(all_runs_per_class[0])
#     summary_rows = []
#
#     for cls_idx in range(num_classes):
#         class_name = all_runs_per_class[0][cls_idx]["Class"]
#         cls_runs = [run[cls_idx] for run in all_runs_per_class]
#         summary_rows.append(summarize_dicts(cls_runs, class_name))
#
#     summary_rows.append(summarize_dicts(all_runs_overall, "all"))
#     return summary_rows
#
#
# if __name__ == '__main__':
#     model_paths = [
#         r'D:\python_space\ultralytics-yolo11-main-origin\runs\train\yolo11n-LEGM\exp_seed_0\weights\best.pt',
#         r'D:\python_space\ultralytics-yolo11-main-origin\runs\train\yolo11n-LEGM\exp_seed_42\weights\best.pt',
#         r'D:\python_space\ultralytics-yolo11-main-origin\runs\train\yolo11n-LEGM\exp_seed_2024\weights\best.pt',
#     ]
#
#     dense_yaml = r'D:\python_space\ultralytics-yolo11-main-origin\Dense_dataset_experiment\dataset\dense.yaml'
#     nondense_yaml = r'D:\python_space\ultralytics-yolo11-main-origin\Dense_dataset_experiment\dataset\nodense.yaml'
#
#     # 模型信息表
#     model_for_info = YOLO(model_paths[0])
#     info_table = dump_model_info_table(model_for_info, model_paths[0])
#     print(info_table)
#
#     # dense 测试集：3次结果取均值 ± 标准差
#     dense_summary = evaluate_three_models(model_paths, dense_yaml, 'dense')
#     dense_table = dump_summary_table(dense_summary, "Dense Test Metrics (mean ± std)")
#     print(dense_table)
#
#     # nondense 测试集：3次结果取均值 ± 标准差
#     nondense_summary = evaluate_three_models(model_paths, nondense_yaml, 'nondense')
#     nondense_table = dump_summary_table(nondense_summary, "NonDense Test Metrics (mean ± std)")
#     print(nondense_table)
#
#     # 保存结果
#     save_dir = r'D:\python_space\ultralytics-yolo11-main-origin\runs\dense_and_nodense_eval'
#     os.makedirs(save_dir, exist_ok=True)
#
#     with open(os.path.join(save_dir, 'paper_summary.txt'), 'w', encoding='utf-8') as f:
#         f.write(str(info_table) + '\n\n')
#         f.write(str(dense_table) + '\n\n')
#         f.write(str(nondense_table) + '\n')


'''
基础代码的测试
'''
import warnings

warnings.filterwarnings('ignore')

from pathlib import Path
import os
import yaml
import numpy as np
from prettytable import PrettyTable
from ultralytics import YOLO
from ultralytics.utils.torch_utils import model_info


def get_weight_size(path):
    stats = os.stat(path)
    return f'{stats.st_size / 1024 / 1024:.1f}'


def run_val(model, data_yaml, run_name):
    result = model.val(
        data=data_yaml,
        split='test',
        imgsz=640,
        batch=8,
        conf=0.25,
        project=r'D:\python_space\ultralytics-yolo11-main-origin\runs\yolov11-LEGM_eval',
        name=run_name,
        save_json=False,
        save=True,
        plots=True,
        show_conf=False,
        save_txt=False,
    )
    return result


def safe_div(a, b):
    return a / b if b != 0 else 0.0



def resolve_test_image_dir(data_yaml):
    """
    从 data.yaml 中解析 test 图像目录。
    兼容：
        path: xxx
        test: images/test
    或 test 为绝对路径。
    """
    data = load_data_yaml(data_yaml)

    test_path = data.get("test", None)
    if test_path is None:
        raise ValueError("data.yaml 中未找到 test 字段")

    yaml_dir = os.path.dirname(os.path.abspath(data_yaml))
    dataset_root = data.get("path", None)

    if dataset_root is not None and not os.path.isabs(dataset_root):
        dataset_root = os.path.join(yaml_dir, dataset_root)

    if not os.path.isabs(test_path):
        if dataset_root is not None:
            test_path = os.path.join(dataset_root, test_path)
        else:
            test_path = os.path.join(yaml_dir, test_path)

    test_path = os.path.abspath(test_path)

    if not os.path.isdir(test_path):
        raise FileNotFoundError(f"test 图像目录不存在: {test_path}")

    return test_path


def save_detection_boxes_xyxy(
    model_path,
    data_yaml,
    save_dir,
    run_name="yolov11_LEGM_boxes",
    imgsz=640,
    conf=0.25,
    iou=0.5,
    exts=(".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
):
    """
    重新用 YOLOv11_LEGM 对 test 图像推理，并保存检测框。

    保存格式：
        1) 每张图一个 txt：
           cls x1 y1 x2 y2 conf

        2) 一个总 csv：
           image_name,cls,x1,y1,x2,y2,conf
    """
    model = YOLO(model_path)
    image_dir = resolve_test_image_dir(data_yaml)

    save_dir = Path(save_dir)
    txt_dir = save_dir / run_name / "labels_xyxy"
    txt_dir.mkdir(parents=True, exist_ok=True)

    csv_path = save_dir / run_name / "detections_xyxy.csv"

    image_paths = []
    for root, _, files in os.walk(image_dir):
        for file in files:
            if file.lower().endswith(exts):
                image_paths.append(os.path.join(root, file))

    image_paths = sorted(image_paths)

    if len(image_paths) == 0:
        raise RuntimeError(f"未在 test 图像目录中找到图像: {image_dir}")

    csv_lines = ["image_name,cls,x1,y1,x2,y2,conf"]

    for img_path in image_paths:
        results = model.predict(
            source=img_path,
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            save=False,
            verbose=False
        )

        img_name = Path(img_path).stem
        txt_path = txt_dir / f"{img_name}.txt"

        lines = []

        for r in results:
            if r.boxes is None or len(r.boxes) == 0:
                break

            boxes_xyxy = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            clss = r.boxes.cls.cpu().numpy().astype(int)

            for box, score, cls_id in zip(boxes_xyxy, confs, clss):
                x1, y1, x2, y2 = box.tolist()

                line = f"{cls_id} {x1:.2f} {y1:.2f} {x2:.2f} {y2:.2f} {score:.4f}"
                lines.append(line)

                csv_lines.append(
                    f"{Path(img_path).name},{cls_id},{x1:.2f},{y1:.2f},{x2:.2f},{y2:.2f},{score:.4f}"
                )

        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("\n".join(csv_lines))

    print(f"[OK] Saved xyxy txt boxes to: {txt_dir}")
    print(f"[OK] Saved xyxy csv boxes to: {csv_path}")

    return str(txt_dir), str(csv_path)


def calc_det_acc_from_pr(p, r, n_gt):
    """
    自定义检测ACC:
    ACC = TP / (TP + FP + FN)
    """
    tp = r * n_gt
    fp = tp * (1.0 / p - 1.0) if p > 0 else 0.0
    fn = n_gt - tp
    return safe_div(tp, tp + fp + fn)


def load_data_yaml(data_yaml):
    with open(data_yaml, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    return data


def infer_label_dir_from_test_path(test_path):
    """
    YOLO数据集常见结构:
    images/test  <->  labels/test
    """
    test_path = os.path.abspath(test_path)

    # 情况1：路径中含 images
    if f'{os.sep}images{os.sep}' in test_path:
        return test_path.replace(f'{os.sep}images{os.sep}', f'{os.sep}labels{os.sep}')

    # 情况2：如果 test_path 就是 test 目录，则尝试同级 labels
    parent = os.path.dirname(test_path)
    candidate = os.path.join(parent, 'labels')
    if os.path.isdir(candidate):
        return candidate

    raise FileNotFoundError(f'无法根据 test 路径推断 labels 路径: {test_path}')


def count_nt_per_class_from_labels(data_yaml):
    """
    从测试集 labels 目录统计每个类别的GT数量
    兼容 data.yaml 里的 path + test 写法
    """
    data = load_data_yaml(data_yaml)

    names = data.get('names', None)
    if isinstance(names, dict):
        class_names = [names[i] for i in sorted(names.keys())]
    elif isinstance(names, list):
        class_names = names
    else:
        raise ValueError("data.yaml 中未找到合法的 names 字段")

    num_classes = len(class_names)

    test_path = data.get('test', None)
    if test_path is None:
        raise ValueError("data.yaml 中未找到 test 字段")

    yaml_dir = os.path.dirname(os.path.abspath(data_yaml))
    dataset_root = data.get('path', None)

    # 处理 path
    if dataset_root is not None and not os.path.isabs(dataset_root):
        dataset_root = os.path.join(yaml_dir, dataset_root)

    # 处理 test
    if not os.path.isabs(test_path):
        if dataset_root is not None:
            test_path = os.path.join(dataset_root, test_path)
        else:
            test_path = os.path.join(yaml_dir, test_path)

    test_path = os.path.abspath(test_path)

    label_dir = infer_label_dir_from_test_path(test_path)

    if not os.path.isdir(label_dir):
        raise FileNotFoundError(
            f'labels 目录不存在: {label_dir}\n'
            f'解析得到的 test 路径为: {test_path}\n'
            f'请检查 data.yaml 中的 path/test 配置。'
        )

    nt_per_class = np.zeros(num_classes, dtype=np.int64)

    for root, _, files in os.walk(label_dir):
        for file in files:
            if not file.endswith('.txt'):
                continue
            txt_path = os.path.join(root, file)
            with open(txt_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    cls_id = int(float(parts[0]))
                    if 0 <= cls_id < num_classes:
                        nt_per_class[cls_id] += 1

    return class_names, nt_per_class


def dump_nt_per_class_table(class_names, nt_per_class, title="Test Set nt_per_class"):
    table = PrettyTable()
    table.title = title
    table.field_names = ["Class", "nt_per_class"]

    for name, n in zip(class_names, nt_per_class):
        table.add_row([name, int(n)])

    table.add_row(["all", int(np.sum(nt_per_class))])
    return table


def extract_metrics(result, nt_per_class):
    """
    说明：
    - 不再使用 result.nt_per_class
    - 每类指标从 result.box 里取
    - 整体指标从 result.results_dict 里取
    """
    class_names = list(result.names.values())
    num_classes = len(class_names)

    per_class = []
    for i in range(num_classes):
        p = float(result.box.p[i])
        r = float(result.box.r[i])
        f1 = float(result.box.f1[i])
        map50 = float(result.box.ap50[i])
        map75 = float(result.box.all_ap[i, 5])   # IoU=0.75, 对应 0.50:0.05:0.95 的第6列
        map5095 = float(result.box.ap[i])

        n_gt = float(nt_per_class[i]) if i < len(nt_per_class) else 0.0
        acc = calc_det_acc_from_pr(p, r, n_gt)

        per_class.append({
            "Class": class_names[i],
            "nt_per_class": int(n_gt),
            "ACC": acc,
            "Precision": p,
            "Recall": r,
            "F1_score": f1,
            "mAP50": map50,
            "mAP75": map75,
            "mAP50-95": map5095,
        })

    p_all = float(result.results_dict['metrics/precision(B)'])
    r_all = float(result.results_dict['metrics/recall(B)'])
    f1_all = safe_div(2 * p_all * r_all, p_all + r_all)

    total_gt = float(np.sum(nt_per_class))
    acc_all = calc_det_acc_from_pr(p_all, r_all, total_gt)

    overall = {
        "Class": "all",
        "nt_per_class": int(total_gt),
        "ACC": acc_all,
        "Precision": p_all,
        "Recall": r_all,
        "F1_score": f1_all,
        "mAP50": float(result.results_dict['metrics/mAP50(B)']),
        "mAP75": float(np.mean(result.box.all_ap[:num_classes, 5])),
        "mAP50-95": float(result.results_dict['metrics/mAP50-95(B)']),
    }

    return per_class, overall


def summarize_dicts(dict_list, class_name, n_gt=None):
    metrics_keys = ["ACC", "Precision", "Recall", "F1_score", "mAP50", "mAP75", "mAP50-95"]

    summary = {
        "Class": class_name,
        "nt_per_class": int(n_gt) if n_gt is not None else "-"
    }

    for k in metrics_keys:
        vals = np.array([d[k] for d in dict_list], dtype=float)
        mean_v = np.mean(vals)
        std_v = np.std(vals, ddof=1) if len(vals) > 1 else 0.0
        summary[k] = f"{mean_v:.4f} ± {std_v:.4f}"

    return summary


def dump_model_info_table(model, model_path):
    n_l, n_p, n_g, flops = model_info(model.model)

    table = PrettyTable()
    table.title = "Model Info"
    table.field_names = ["GFLOPs", "Parameters", "Model Size"]
    table.add_row([
        f'{flops:.1f}',
        f'{n_p:,}',
        f'{get_weight_size(model_path)}MB'
    ])
    return table


def dump_summary_table(summary_rows, title):
    table = PrettyTable()
    table.title = title
    table.field_names = [
        "Class", "nt_per_class", "ACC", "Precision", "Recall",
        "F1_score", "mAP50", "mAP75", "mAP50-95"
    ]

    for row in summary_rows:
        table.add_row([
            row["Class"],
            row["nt_per_class"],
            row["ACC"],
            row["Precision"],
            row["Recall"],
            row["F1_score"],
            row["mAP50"],
            row["mAP75"],
            row["mAP50-95"],
        ])
    return table


def evaluate_three_models(model_paths, data_yaml, dataset_tag):
    class_names, nt_per_class = count_nt_per_class_from_labels(data_yaml)

    all_runs_per_class = []
    all_runs_overall = []

    for idx, model_path in enumerate(model_paths, start=1):
        model = YOLO(model_path)
        run_name = f"{dataset_tag}_run{idx}"

        result = run_val(model, data_yaml, run_name)
        per_class, overall = extract_metrics(result, nt_per_class)

        all_runs_per_class.append(per_class)
        all_runs_overall.append(overall)

    num_classes = len(all_runs_per_class[0])
    summary_rows = []

    for cls_idx in range(num_classes):
        class_name = all_runs_per_class[0][cls_idx]["Class"]
        cls_runs = [run[cls_idx] for run in all_runs_per_class]
        summary_rows.append(
            summarize_dicts(
                cls_runs,
                class_name,
                n_gt=nt_per_class[cls_idx]
            )
        )

    summary_rows.append(
        summarize_dicts(
            all_runs_overall,
            "all",
            n_gt=np.sum(nt_per_class)
        )
    )

    return class_names, nt_per_class, summary_rows


if __name__ == '__main__':
    model_paths = [
        r'D:\python_space\ultralytics-yolo11-main-origin\runs\roi_classifier\Swin_Hier1_KD1_train2p0_test2p0\best_seed0.pt',
        r'D:\python_space\ultralytics-yolo11-main-origin\runs\roi_classifier\Swin_Hier1_KD1_train2p0_test2p0\best_seed42.pt',
        r'D:\python_space\ultralytics-yolo11-main-origin\runs\roi_classifier\Swin_Hier1_KD1_train2p0_test2p0\best_seed2024.pt',
    ]

    data_yaml = r'D:\python_space\ultralytics-yolo11-main-origin\detect_dataset\data.yaml'

    # 模型信息表（以第一个模型为例）
    model_for_info = YOLO(model_paths[0])
    info_table = dump_model_info_table(model_for_info, model_paths[0])
    print(info_table)

    # 统计测试集类别样本数 + 三个模型评估并做均值±标准差
    class_names, nt_per_class, dataset_summary = evaluate_three_models(
        model_paths, data_yaml, 'dataset'
    )

    nt_table = dump_nt_per_class_table(class_names, nt_per_class, "Dataset Test nt_per_class")
    print(nt_table)

    dataset_table = dump_summary_table(dataset_summary, "Dataset Test Metrics (mean ± std)")
    print(dataset_table)

    # 保存结果
    save_dir = r'D:\python_space\ultralytics-yolo11-main-origin\runs\test\Swin_Hier1_KD1_train2p0_test2p0'
    os.makedirs(save_dir, exist_ok=True)

    with open(os.path.join(save_dir, 'paper_summary.txt'), 'w', encoding='utf-8') as f:
        f.write(str(info_table) + '\n\n')
        f.write(str(nt_table) + '\n\n')
        f.write(str(dataset_table) + '\n\n')

    # =====================================================
    # 新增：保存 YOLOv11_LEGM 检测框，供 ROI Grad-CAM 使用
    # =====================================================
    det_weight_for_vis = model_paths[1]  # 推荐用 seed_42 的 best.pt

    save_detection_boxes_xyxy(
        model_path=det_weight_for_vis,
        data_yaml=data_yaml,
        save_dir=save_dir,
        run_name="D:\\python_space\\ultralytics-yolo11-main-origin\\detect_dataset\\seed42_test_boxes_xyxy",
        imgsz=640,
        conf=0.25,
        iou=0.5
    )