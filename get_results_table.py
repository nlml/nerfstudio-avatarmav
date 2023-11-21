from copy import deepcopy
import numpy as np
from time import sleep
from PIL import Image
from shutil import copyfile
from pathlib import Path
import pandas as pd
import os
import json

EXTERNAL_DATA_SERVER_PATH = None
DATA_DIR = os.environ.get("DATA_DIR", "/data/shenhan-cvpr")

run_names = os.listdir("./")
run_names = [x for x in run_names if os.path.isdir(x)]

json_str_test = "UNION_{id}_EMO1234EXP234589_v16_DS2-0.5x_lmkSTAR_teethV3_SMOOTH_offsetS_whiteBg_maskBelowLine/transforms_test.json/results.json"
json_str_vali = "UNION_{id}_EMO1234EXP234589_v16_DS2-0.5x_lmkSTAR_teethV3_SMOOTH_offsetS_whiteBg_maskBelowLine/transforms_val.json/results.json"

self_reenact_default_path = (
    "{run_id}-UNION/nersemble_masked/{run_id}/"
    "UNION_{run_id}_EMO1234EXP234589_v16_DS2-0.5x_lmkSTAR_teethV3_SMOOTH_offsetS_whiteBg_maskBelowLine"
    "/transforms_test.json/images/"
)

novel_view_default_path = (
    "{run_id}-UNION/nersemble_masked/{run_id}/"
    "UNION_{run_id}_EMO1234EXP234589_v16_DS2-0.5x_lmkSTAR_teethV3_SMOOTH_offsetS_whiteBg_maskBelowLine"
    "/transforms_val.json/images/"
)


def load_single_results(path):
    with open(path, "r") as cross_re_fname:
        results = json.load(cross_re_fname)
    out = results["results"]
    keep = ["psnr", "ssim", "lpips"]
    for k in list(out.keys()):
        if k not in keep:
            del out[k]
        else:
            if k == "lpips":
                out[k] = round(out[k], 3)
            elif k == "ssim":
                out[k] = round(out[k], 3)
            elif k == "psnr":
                out[k] = round(out[k], 1)
    return out


def get_results(run_name):
    run_id = run_name.split("-")[0]
    out = {}
    p = os.path.join(
        run_name, "nersemble_masked", run_id, json_str_vali.format(id=run_id)
    )
    if os.path.exists(p):
        out["vali"] = load_single_results(p)
    p = os.path.join(
        run_name, "nersemble_masked", run_id, json_str_test.format(id=run_id)
    )
    if os.path.exists(p):
        out["test"] = load_single_results(p)
    return out


results = {}

for run_name in run_names:
    this_results = None
    if "UNION" not in run_name:
        continue
    try:
        this_results = get_results(run_name)
    except Exception as e:
        print("Failed for", run_name)
        print(e)
    if this_results is not None:
        for k in this_results:
            results[run_name + "_" + k] = this_results[k]
# print(pd.DataFrame(results).T.loc["264-UNION_test", :])
results = pd.DataFrame(results)
for c in results.columns:
    results[c] = results[c].astype(str).str.replace(".", ",,,")
    results[c] = '"' + results[c] + '"'
printstr = results.T.__str__()
printstr = printstr.replace("-UNION_", "-")
# printstr = printstr.replace(".", ",")
while "  " in printstr:
    printstr = printstr.replace("  ", " ")
print(printstr)
# print(results)

novel_view_desired_filenames = [
    "074_val_00386_avmav.png",
    "074_val_00622_avmav.png",
    "074_val_0386_avmav.png",
    "104_val_00091_avmav.png",
    "104_val_00242_avmav.png",
    "104_val_00307_avmav.png",
    "218_val_00908_avmav.png",
    "253_val_00297_avmav.png",
    "302_val_00100_avmav.png",
    "304_val_00139_avmav.png",
    "304_val_01567_avmav.png",
    "306_val_00416_avmav.png",
    "306_val_00439_avmav.png",
    "460_val_01252_avmav.png",
]


self_reenactment_desired_filenames = [
    "074_test_01258_avmav.png",
    "104_test_00240_avmav.png",
    "264_test_01827_avmav.png",
    "302_test_00381_avmav.png",
    "302_test_00383_avmav.png",
    "304_test_00311_avmav.png",
    "306_test_01024_avmav.png",
]


def run_id_and_frame_id_from_desired_filename(cross_re_fname):
    run_id = cross_re_fname.split("_")[0]
    frame_id = cross_re_fname.split("_")[2].split("_")[0]
    return run_id, frame_id


def crop_right_half_and_save_im(src, out_dir, cross_re_fname):
    im = Image.open(src)
    # crop the image to the right half:
    # im = im.crop((im.width // 2, 0, im.width, im.height))
    im.save(out_dir / (cross_re_fname.split(".")[0] + ".png"))


novel_view_out_dir = Path("novel_view")
novel_view_out_dir.mkdir(exist_ok=True)
for cross_re_fname in novel_view_desired_filenames:
    run_id, frame_id = run_id_and_frame_id_from_desired_filename(cross_re_fname)
    print(run_id, frame_id)
    src = (
        Path(novel_view_default_path.format(run_id=run_id, frame_id=frame_id))
        / cross_re_fname"{int(frame_id):06d}-pr.png"
    )
    if not src.exists():
        print("Not found", src)
    else:
        crop_right_half_and_save_im(src, novel_view_out_dir, cross_re_fname)

self_reenact_out_dir = Path("self_reenactment")
self_reenact_out_dir.mkdir(exist_ok=True)
for cross_re_fname in self_reenactment_desired_filenames:
    run_id, frame_id = run_id_and_frame_id_from_desired_filename(cross_re_fname)
    print(run_id, frame_id)
    src = (
        Path(self_reenact_default_path.format(run_id=run_id, frame_id=frame_id))
        / cross_re_fname"{int(frame_id):06d}-pr.png"
    )
    if not src.exists():
        print("Not found", src)
    else:
        crop_right_half_and_save_im(src, self_reenact_out_dir, cross_re_fname)


cross_reenact_desired_filenames = [
    "074-165_FREE_00028_avmav.png",
    "104-165_FREE_00028_avmav.png",
    "104-165_FREE_00072_avmav.png",
    "104-165_FREE_00082_avmav.png",
    "104-165_FREE_00137_avmav.png",
    "104-264_FREE_00149_avmav.png",
    "104-264_FREE_00181_avmav.png",
    "218-140_FREE_00127_avmav.png",
    "218-140_FREE_00232_avmav.png",
    "218-210_FREE_00079_avmav.png",
    "218-210_FREE_00116_avmav.png",
    "218-210_FREE_00132_avmav.png",
    "218-210_FREE_00168_avmav.png",
    "218-210_FREE_00209_avmav.png",
    "218-210_FREE_00224_avmav.png",
    "218-253_FREE_00038_avmav.png",
    "218-253_FREE_00064_avmav.png",
    "218-253_FREE_00132_avmav.png",
    "218-253_FREE_00148_avmav.png",
    "218-253_FREE_00209_avmav.png",
    "218-264_FREE_00164_avmav.png",
    "218-304_FREE_00030_avmav.png",
    "218-304_FREE_00100_avmav.png",
    "218-460_FREE_00331_avmav.png",
    "253-140_FREE_00071_avmav.png",
    "253-140_FREE_00096_avmav.png",
    "253-140_FREE_00128_avmav.png",
    "253-140_FREE_00155_avmav.png",
    "253-210_FREE_00089_avmav.png",
    "253-210_FREE_00132_avmav.png",
    "253-210_FREE_00240_avmav.png",
    "253-264_FREE_00262_avmav.png",
    "302-140_FREE_00046_avmav.png",
    "302-140_FREE_00128_avmav.png",
    "302-140_FREE_00239_avmav.png",
    "302-210_FREE_00169_avmav.png",
    "302-218_FREE_00110_avmav.png",
    "302-253_FREE_00041_avmav.png",
    "304-210_FREE_00168_avmav.png",
    "306-460_FREE_00139_avmav.png",
    "460-264_FREE_00181_avmav.png",
]

cross = {}
try_scp = False
new_actor_source_combo = False
for cross_re_fname in sorted(cross_reenact_desired_filenames):
    if (
        "140" not in cross_re_fname
        and "218-264" not in cross_re_fname
        and "302-218" not in cross_re_fname
        and "304-210" not in cross_re_fname
        and "304-210" not in cross_re_fname
    ):
        continue
    actor_id = cross_re_fname.split("-")[0]
    source_id = cross_re_fname.split("-")[1].split("_")[0]
    actor_source_str = f"{actor_id}-{source_id}"
    if actor_source_str not in cross:
        new_actor_source_combo = True
        cross[actor_source_str] = []
    source_expression_id = cross_re_fname.split("-")[1].split("_")[0]
    json_path = (
        Path("nersemble_masked")
        / source_expression_id
        / f"{source_expression_id}_FREE_v16_DS2-0.5x_lmkSTAR_teethV3_SMOOTH_offsetS_whiteBg_maskBelowLine"
        / "transforms.json"
    )
    assert os.path.exists(Path("../../") / json_path)
    if EXTERNAL_DATA_SERVER_PATH is not None and try_scp and not os.path.exists(Path("../../") / json_path):
        Path(f"{DATA_DIR}/{json_path}").parent.mkdir(
            exist_ok=True, parents=True
        )
        cmd = f"scp {EXTERNAL_DATA_SERVER_PATH}/{str(json_path)} {DATA_DIR}/{str(json_path)}"
        # print(cmd)
        ret = os.system(cmd)
        if ret != 0:
            # print("Failed to download", json_path)
            continue

    # print(Path("../../") / json_path)
    if os.path.exists(Path("../../") / json_path):
        with open(Path("../../") / json_path, "r") as f:
            transforms = json.load(f)
        valid_frames = [fr for fr in transforms["frames"] if fr["camera_index"] == 8]
        frame_index = str(cross_re_fname).split("_FREE_")[1].split("_")[0]
        # print(frame_index)
        # print(cross_re_fname, actor_id, actor_source_str, source_expression_id, int(frame_index))
        frame = valid_frames[int(frame_index)]
        if new_actor_source_combo:
            cross[actor_source_str] = {}  # deepcopy(transforms)
            cross[actor_source_str]["frames"] = []
            new_actor_source_combo = False
        for path_key_to_fix in ["file_path", "fg_mask_path", "flame_param_path"]:
            frame[path_key_to_fix] = str(
                (Path("../../") / json_path).absolute().parent / frame[path_key_to_fix]
            )
        frame["desired_filename"] = cross_re_fname
        cross[actor_source_str]["frames"].append(frame)


include_list = [264, 140, 218, 210]
cross_re_outdir = Path("cross_reenactment")
cross_re_outdir.mkdir(exist_ok=True)
for actor_source_str in cross:
    actor, source = actor_source_str.split("-")
    if int(source) not in include_list:
        continue
    print()
    print(actor_source_str)
    print(len(cross[actor_source_str]["frames"]))
    json_frame_to_run = f"/tmp/cross_{actor}-{source}.json"
    with open(json_frame_to_run, "w") as f:
        json.dump(cross[actor_source_str], f)
    # run ns-eval here

    cmd = f"cd {DATA_DIR}/ && ./eval_script.sh {actor}-UNION /tmp/cross_{actor}-{source}.json"

    print(cmd)
    os.system(cmd)
    outs_dir = f"{DATA_DIR}/fixed-intrinsics/eval_outputs/{actor}-UNION/tmp/cross_{actor}-{source}.json"
    pngpaths = sorted(
        [
            Path(outs_dir) / "images" / i
            for i in os.listdir(Path(outs_dir) / "images")
            if "-pr.png" in i
        ]
    )
    frames = cross[actor_source_str]["frames"]

    for i, pngpath in enumerate(pngpaths):
        outname = cross_re_outdir / frames[i]["desired_filename"]
        copyfile(pngpath, outname)
