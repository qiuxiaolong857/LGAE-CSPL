import warnings
warnings.filterwarnings('ignore')

from ultralytics import YOLO
import sys
import os
import time

if __name__ == '__main__':
    seed = int(sys.argv[1])

    model_yaml = r'D:\python_space\ultralytics-yolo11-main-origin\ultralytics\cfg\models\11\yolo11n-C3k2-LEGM.yaml'
    project = r'D:\python_space\ultralytics-yolo11-main-origin\runs\train\yolo11n-LEGM'
    run_name = f'exp_seed_{seed}'

    weights_dir = os.path.join(project, run_name, 'weights')
    best_pt = os.path.join(weights_dir, 'best.pt')
    last_pt = os.path.join(weights_dir, 'last.pt')

    if os.path.exists(best_pt):
        print(f"✅ seed={seed} 已训练完成，跳过")
        exit()

    print(f"\n========== seed={seed} ==========\n")

    print(f"\n========== seed={seed} ==========\n")

    # ✅ 判断是否需要 resume
    if os.path.exists(last_pt):
        print("🔁 检测到 last.pt，继续训练（resume=True）")
        model = YOLO(last_pt)
        resume_flag = True
    else:

        print("🆕 未检测到 last.pt，从头训练")
        model = YOLO(model_yaml)
        resume_flag = False

    # 🚀 开始训练
    model.train(
        data=r'D:\python_space\ultralytics-yolo11-main-origin\detect_dataset\data.yaml',
        imgsz=640,
        epochs=300,
        batch=8,
        seed=seed,
        project=project,
        name=run_name,
        save_period=10,
        resume=resume_flag,
    )

    # =============================
    # ✅ 训练结束后：权重校验
    # =============================
    print("\n🔍 开始检查权重文件完整性...")

    def check_weight(path, name):
        if not os.path.exists(path):
            print(f"❌ {name} 不存在")
            return

        # 等待文件写入稳定
        time.sleep(2)

        size_mb = os.path.getsize(path) / 1024 / 1024
        print(f"{name} 大小: {size_mb:.2f} MB")

        if size_mb < 1:
            print(f"⚠️ 警告：{name} 可能损坏（过小）")
        else:
            print(f"✅ {name} 正常")

    check_weight(best_pt, "best.pt")
    check_weight(last_pt, "last.pt")

    print("\n🎉 训练流程结束\n")