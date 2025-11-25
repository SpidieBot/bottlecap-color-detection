# tools/test_cpu_inference_time.py  ← FINAL VERSION
import argparse
import os
import time

import cv2
import ncnn
import numpy as np


def test_ncnn(model_dir: str, image_path: str, num_runs: int = 200):
    # === Load model ===
    net = ncnn.Net()
    net.opt.use_vulkan_compute = False  # CPU only
    net.opt.num_threads = os.cpu_count()  # use all cores

    param_path = os.path.join(model_dir, "model.ncnn.param")
    bin_path = os.path.join(model_dir, "model.ncnn.bin")

    if not (os.path.exists(param_path) and os.path.exists(bin_path)):
        raise FileNotFoundError(f"Missing model files in {model_dir}")

    net.load_param(param_path)
    net.load_model(bin_path)

    # === Auto-detect blob names ===
    in_name = net.input_names()[0]
    out_name = net.output_names()[0]
    print(f"Detected input='{in_name}'  output='{out_name}'")

    # === Load and preprocess image exactly as Ultralytics does ===
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(image_path)
    img = cv2.resize(img, (320, 320))
    img = img.astype(np.float32)
    img = img / 255.0  # 0~1
    img = img.transpose(2, 0, 1)  # HWC → CHW
    img = np.expand_dims(img, axis=0)  # (1,3,320,320)
    mat_in = ncnn.Mat(img).clone()  # correct shape!

    # === Warm-up ===
    ex = net.create_extractor()
    ex.input(in_name, mat_in)
    _, _ = ex.extract(out_name)
    print("Warm-up done")

    # === Benchmark ===
    times = []
    for i in range(num_runs):
        ex = net.create_extractor()
        ex.input(in_name, mat_in)
        start = time.perf_counter()
        ex.extract(out_name, _)
        elapsed_ms = (time.perf_counter() - start) * 1000
        times.append(elapsed_ms)

        if (i + 1) % 50 == 0:
            print(f"  Run {i+1}/{num_runs} → {elapsed_ms:.2f} ms")

    avg = np.mean(times)
    std = np.std(times)
    fps = 1000 / avg

    print(f"\n[NCNN] Average inference : {avg:.2f} ms ± {std:.2f} ms")
    print(f"       → {fps:.1f} FPS  ({num_runs} runs on CPU)")
    print(f"       Model folder : {model_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NCNN CPU benchmark – YOLO11n 2025")
    parser.add_argument("--model-dir", type=str, required=True)
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--runs", type=int, default=200)
    args = parser.parse_args()

    print("NCNN CPU Inference Benchmark")
    print(f"Model   : {args.model_dir}")
    print(f"Image   : {args.image}")
    print(f"Runs    : {args.runs}\n")

    test_ncnn(args.model_dir, args.image, args.runs)
