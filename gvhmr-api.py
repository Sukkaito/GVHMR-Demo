
from pydantic import BaseModel
from uuid import uuid4
from fastapi import status
from fastapi import BackgroundTasks
import os
import shutil
from pathlib import Path
import torch
from fastapi import FastAPI, UploadFile, File, Query, HTTPException, Security, Depends
from fastapi.security import APIKeyHeader
from fastapi.responses import FileResponse

from typing import List, Optional, Union, Dict, Any
import numpy as np
from hmr4d.utils.eval.eval_utils import (
    batch_align_by_pelvis,
    batch_compute_similarity_transform_torch,
    compute_jpe,
    convert_joints22_to_24
)

from tools.demo.demo import(
    run_preprocess,
    load_data_dict,
    render_incam,
    render_global
)

from hmr4d.configs import register_store_gvhmr
from hydra import initialize_config_module, compose
import hydra
from hmr4d.utils.net_utils import detach_to_cpu
from hmr4d.model.gvhmr.gvhmr_pl_demo import DemoPL

from hmr4d.utils.video_io_utils import get_video_lwh

import ffmpeg

app = FastAPI(title="GVHMR API")

UPLOAD_DIR = Path("input/temp_upload")
OUTPUT_DIR = Path("output/result")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


API_KEY_NAME = "X-API-Key"

# Load các biến môi trường từ file .env
try:
    # pyrefly: ignore [missing-import]
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Lấy API_KEY từ file .env. Nếu không có thì dùng chuỗi mặc định.
API_KEY = os.getenv("API_KEY")

api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=True)

def verify_token(api_key: str = Security(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token không hợp lệ hoặc bị thiếu"
        )
    return api_key


model = None
hydra_initalized = False

class JobCreateRequest(BaseModel):
    video_id: str
    static_cam: bool = True
    use_dpvo: bool = True


class CalculateMetricsRequest(BaseModel):
    pred_j3d: Optional[List[Any]] = None          # (F, J, 3) hoặc (J, 3)
    target_j3d: Optional[List[Any]] = None        # (F, J, 3) hoặc (J, 3)
    target_file_path: Optional[str] = None        # Đường dẫn file GT (.npy hoặc .pt) trên server
    job_id: Optional[str] = None                  # Tùy chọn: lấy pred_j3d từ job_id đã hoàn thành
    pelvis_idxs: List[int] = [1, 2]
    unit: str = "mm"



def get_api_config(video_path: Path, static_cam: bool = False, use_dpvo: bool = False):
    """
        Ham khoi tao cau hinh hydra dong cho moi video yeu cau xu ly
    """
    global hydra_initalized

    if not hydra_initalized:

        initialize_config_module(
            version_base="1.3",
            config_module = "hmr4d.configs"
            )

        hydra_initalized = True
         
    overrides = [
            f"video_name={video_path.stem}",
            f"static_cam={str(static_cam).lower()}",
            f"verbose=False",
            f"use_dpvo={str(use_dpvo).lower()}"
        ]
        
    register_store_gvhmr()
    
    config = compose(config_name="demo", overrides=overrides)
        
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    Path(config.preprocess_dir).mkdir(parents=True, exist_ok=True)
        
    return config

def extract_fps_and_copy(input_path: Path, output_path: Path):
    """
    Kiểm tra và trả về FPS gốc của video, sau đó copy video sang output_path.
    Không ép hay thay đổi FPS nữa.
    """
    fps = 30.0
    try:
        probe = ffmpeg.probe(str(input_path))
        video_stream = None
        for stream in probe.get("streams", []):
            if stream["codec_type"] == "video":
                video_stream = stream
                break
        if video_stream and 'avg_frame_rate' in video_stream:
            num, den = video_stream['avg_frame_rate'].split('/')
            fps = float(num) / float(den)
    except Exception as e:
        print(f"Loi doc FPS: {e}")
        
    import shutil
    shutil.copy(input_path, output_path)
    return fps

            



@app.on_event("startup")
def load_model():
    """
        Su kien chay khi khoi dong server.Tai mo hinh vao GPU truoc de tang toc API
    """
    global model
    ckpt_path = "inputs/checkpoints/gvhmr/gvhmr_siga24_release.ckpt"

    if not Path(ckpt_path).exists():
        print(f"[Cảnh báo]: Không tìm thấy checkpoint tại {ckpt_path}. Vui lòng tải về trước.")
        return
    print("Loading GVHMR model into GPU...")

    global hydra_initalized
    if not hydra_initalized:
        initialize_config_module(version_base="1.3", config_module = "hmr4d.configs")
        hydra_initalized = True

        register_store_gvhmr()
        config = compose(config_name="demo")
        
        model = hydra.utils.instantiate(config.model, _recursive_=False)
        model.load_pretrained_model(ckpt_path)
        model = model.eval().cuda()
        print("Model loaded successfully")

@app.get("/ping")
@app.get("/api/v1/health")
def health_check():
    """
    Kiểm tra trạng thái server
    """
    return {
        "status": "healthy",
        "gpu_available": torch.cuda.is_available(),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None"
    }

@app.post("/api/v1/video/upload", dependencies=[Depends(verify_token)])
async def upload_video(video: UploadFile = File(...)):
    #Tạo thư mục upload nếu chưa tồn tại
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    from uuid import uuid4
    video_id = str(uuid4())
    
    #Đường dẫn lưu video sau khi upload
    file_path = UPLOAD_DIR / f"{video_id}_{video.filename}"

    try:
        # ghi nội dung file được tải lên vào ổ đĩa
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(video.file, buffer)
            
            
        return {
            "status": "success",
            "video_id": video_id,
            "filename": video.filename,
            "saved_path": str(file_path)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Không thể lưu video: {str(e)}")

jobs_db = {}
def process_video_task(job_id: str, video_path: Path, static_cam: bool, use_dpvo: bool):
    """
        Xu ly tac vu video
    """
    global model
    jobs_db[job_id] = {
        "status": "PROCESSING", 
        "progress": "Tiền xử lý video (tracking/keypoints)",
         "result": None
         }
    
    try:
        # 1.Tao cau hinh Hydra dong cho video
        cfg = get_api_config(video_path, static_cam, use_dpvo)


        jobs_db[job_id]["progress"] = f"Extracting FPS"
        actual_fps = extract_fps_and_copy(video_path, cfg.video_path)
        
        # Tiêm thông số fps thực tế vào config để hệ thống tự động nhận
        from omegaconf import OmegaConf
        OmegaConf.set_struct(cfg, False)
        cfg.video_fps = actual_fps
        OmegaConf.set_struct(cfg, True)

        jobs_db[job_id]["progress"] = "Running preprocessing"
        run_preprocess(cfg)

        # # Copy video sang thu muc lam viec trong pipeline
        # shutil.copy(video_path, cfg.video_path)

        # # 2.Tien xu ly
        # run_preprocess(cfg)

        # 3.Gom du lieu vao mo hinh du doan
        data = load_data_dict(cfg)

        # Lọc sạch NaN ở dữ liệu đầu vào nếu có
        for k in ["kp2d", "bbx_xys", "cam_angvel", "f_imgseq"]:
            if k in data and isinstance(data[k], torch.Tensor):
                data[k] = torch.nan_to_num(data[k], nan=0.0)

        with torch.no_grad():
            pred = model.predict(data, static_cam=cfg.static_cam)
        pred = detach_to_cpu(pred)

        # Lọc sạch NaN trong dict dự đoán trước khi lưu và render
        def clean_nans(d):
            if isinstance(d, dict):
                return {k: clean_nans(v) for k, v in d.items()}
            elif isinstance(d, torch.Tensor):
                return torch.nan_to_num(d, nan=0.0)
            return d

        pred = clean_nans(pred)
        torch.save(pred, cfg.paths.hmr4d_results)

        # 4. Render ket qua tra ve 
        jobs_db[job_id]["progress"] = "Danh render video mesh 3d"
        
        render_incam(cfg)
        
        render_global(cfg)

        from hmr4d.utils.video_io_utils import merge_videos_horizontal

        merge_videos_horizontal([cfg.paths.incam_video, cfg.paths.global_video], cfg.paths.incam_global_horiz_video)

        result_filename = Path(cfg.paths.incam_global_horiz_video).name
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        final_video_path = OUTPUT_DIR / f"{job_id}_{result_filename}"
        shutil.copy(cfg.paths.incam_global_horiz_video, final_video_path)

        jobs_db[job_id] = {
            "status": "COMPLETED",
            "progress": "Hoàn thành",
            "result": {
                "output_video_url": f"/api/v1/download/{final_video_path.name}",
                "result_file_path": str(cfg.paths.hmr4d_results)
            }
        }
    except Exception as e:
        import traceback
        print(f"\n[ERROR] Task {job_id} failed with error:\n")
        traceback.print_exc()
        jobs_db[job_id] = {
            "status": "FAILED",
            "progress": str(e),
            "result": None
        }

@app.get("/api/v1/download/{filename}", dependencies=[Depends(verify_token)])
def download_result(filename: str):
    file_path = OUTPUT_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Không tìm thấy file kết quả.")
    return FileResponse(path=str(file_path), filename=filename, media_type="video/mp4")

@app.post("/api/v1/jobs", dependencies=[Depends(verify_token)])
async def create_job(
    request: JobCreateRequest,
    background_tasks: BackgroundTasks):
    
    """
    Tao tac vu xu ly video moi
    """
    
    global model
    if model is None:
        raise HTTPException(status_code=503, detail="Model chua duoc tai xong")
    
    # 1.Tim file dua tren video_id
    matching_file = list(UPLOAD_DIR.glob(f"{request.video_id}_*"))

    if not matching_file:
        raise HTTPException(status_code=404, detail="Không tìm thấy video")
    video_path = matching_file[0]
    
    # 2.Tao 1 job_id duy nhat
    job_id = str(uuid4())

    #Khoi tao trang thai job
    jobs_db[job_id] = {
        "status": "PENDING",
        "progress": "Dang cho xu ly",
        "result": None
    }

    # 3. Dua vao ham xu ly vao backgroud de chay ngam
    background_tasks.add_task(
        process_video_task,
        job_id,
        video_path,
        request.static_cam,
        request.use_dpvo
    )
    return {
        "status": "success",
        "message": "Da bat dau tac vu xu ly tac vu ngam",
        "job_id": job_id
    }


@app.get("/api/v1/jobs/{job_id}", dependencies=[Depends(verify_token)])
def get_job_status(job_id: str):
    """
        Kiem tra trang thai api
    """
    if job_id not in jobs_db:
        raise HTTPException(status_code=404, detail="Không tìm thấy tác vụ (job_id) được yêu cầu.")

    return jobs_db[job_id]


@app.post("/api/v1/metrics/evaluate", dependencies=[Depends(verify_token)])
async def evaluate_metrics(request: CalculateMetricsRequest):
    """
    Tính toán sai số MPJPE (Mean Per Joint Position Error) và PA-MPJPE (Procrustes-Aligned MPJPE)
    giữa tọa độ khớp 3D dự đoán (pred_j3d) và tọa độ khớp 3D thực tế (target_j3d).
    Có thể truyền trực tiếp `pred_j3d` hoặc cung cấp `job_id` để lấy kết quả từ tác vụ đã hoàn thành.
    """
    pred_data = request.pred_j3d

    # Nếu truyền job_id, tự động tìm pred_j3d trong kết quả công việc đã thực hiện
    if request.job_id:
        if request.job_id not in jobs_db:
            raise HTTPException(status_code=404, detail=f"Không tìm thấy tác vụ với job_id: {request.job_id}")
        
        job_info = jobs_db[request.job_id]
        if job_info.get("status") != "COMPLETED":
            raise HTTPException(status_code=400, detail=f"Tác vụ {request.job_id} chưa hoàn thành (trạng thái: {job_info.get('status')})")
        
        result_info = job_info.get("result", {})
        result_file_path = result_info.get("result_file_path") if result_info else None
        if not result_file_path or not Path(result_file_path).exists():
            raise HTTPException(status_code=404, detail="Không tìm thấy file kết quả của tác vụ.")
        
        try:
            pt_data = torch.load(result_file_path, map_location="cpu")
            if "j3d_cam" in pt_data:
                pred_data = pt_data["j3d_cam"]
            elif "j3d_glob" in pt_data:
                pred_data = pt_data["j3d_glob"]
            elif "pred_j3d" in pt_data:
                pred_data = pt_data["pred_j3d"]
            elif "smpl_params_incam" in pt_data or "smpl_params_global" in pt_data:
                smpl_params = pt_data.get("smpl_params_incam") or pt_data.get("smpl_params_global")
                from hmr4d.model.gvhmr.utils.endecoder import EnDecoder
                endecoder = EnDecoder()
                with torch.no_grad():
                    body_pose = torch.as_tensor(smpl_params["body_pose"], dtype=torch.float32)
                    betas = torch.as_tensor(smpl_params["betas"], dtype=torch.float32)
                    global_orient = torch.as_tensor(smpl_params["global_orient"], dtype=torch.float32)
                    transl = torch.as_tensor(smpl_params["transl"], dtype=torch.float32)
                    if body_pose.dim() == 1 or body_pose.dim() == 2:
                        body_pose = body_pose.reshape(1, -1, 63)
                    if betas.dim() == 1 or betas.dim() == 2:
                        betas = betas.reshape(1, -1, 10)
                    if global_orient.dim() == 1 or global_orient.dim() == 2:
                        global_orient = global_orient.reshape(1, -1, 3)
                    if transl.dim() == 1 or transl.dim() == 2:
                        transl = transl.reshape(1, -1, 3)
                    pred_data = endecoder.fk_v2(body_pose, betas, global_orient, transl)[0]  # (F, 22, 3)
            else:
                raise HTTPException(status_code=400, detail="File kết quả không chứa dữ liệu j3d hoặc smpl_params hợp lệ.")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Lỗi khi đọc file kết quả job: {str(e)}")

    target_data = request.target_j3d

    # Nếu truyền target_file_path, tự động nạp ma trận GT từ file .npy hoặc .pt
    if request.target_file_path:
        gt_path = Path(request.target_file_path)
        if not gt_path.exists():
            raise HTTPException(status_code=404, detail=f"Không tìm thấy file GT: {request.target_file_path}")
        
        try:
            if gt_path.suffix == ".npy":
                loaded_npy = np.load(gt_path, allow_pickle=True)
                if isinstance(loaded_npy, np.ndarray):
                    if loaded_npy.dtype == object and loaded_npy.ndim == 0:
                        dict_data = loaded_npy.item()
                        target_data = dict_data.get("j3d") or dict_data.get("gt_j3d") or dict_data.get("joints3d") or dict_data.get("target_j3d") or dict_data.get("joints")
                    else:
                        target_data = loaded_npy
                elif isinstance(loaded_npy, dict):
                    target_data = loaded_npy.get("j3d") or loaded_npy.get("gt_j3d") or loaded_npy.get("joints3d") or loaded_npy.get("target_j3d")
            elif gt_path.suffix in [".pt", ".pth"]:
                loaded_pt = torch.load(gt_path, map_location="cpu")
                if isinstance(loaded_pt, dict):
                    target_data = loaded_pt.get("j3d") or loaded_pt.get("j3d_cam") or loaded_pt.get("j3d_glob") or loaded_pt.get("target_j3d") or loaded_pt.get("gt_j3d")
                else:
                    target_data = loaded_pt
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Lỗi khi nạp file GT {request.target_file_path}: {str(e)}")

    if target_data is None:
        raise HTTPException(status_code=400, detail="Vui lòng cung cấp ma trận 'target_j3d' hoặc đường dẫn 'target_file_path' hợp lệ.")

    try:
        pred_tensor = torch.tensor(pred_data, dtype=torch.float32)
        target_tensor = torch.tensor(target_data, dtype=torch.float32)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Không thể chuyển đổi dữ liệu đầu vào thành Tensor: {str(e)}")

    if pred_tensor.dim() == 2:
        pred_tensor = pred_tensor.unsqueeze(0)
    if target_tensor.dim() == 2:
        target_tensor = target_tensor.unsqueeze(0)

    if pred_tensor.dim() != 3 or target_tensor.dim() != 3:
        raise HTTPException(status_code=400, detail="Dữ liệu ma trận khớp 3D phải có dạng (F, J, 3) hoặc (J, 3).")

    if pred_tensor.shape[-1] != 3 or target_tensor.shape[-1] != 3:
        raise HTTPException(status_code=400, detail="Chiều cuối cùng của tọa độ khớp 3D phải bằng 3 (x, y, z).")

    # 1. Tự động cắt theo số lượng frame nhỏ hơn nếu F khác nhau
    if pred_tensor.shape[0] != target_tensor.shape[0]:
        min_f = min(pred_tensor.shape[0], target_tensor.shape[0])
        pred_tensor = pred_tensor[:min_f]
        target_tensor = target_tensor[:min_f]

    # 2. Tự động chuyển đổi hoặc căn chỉnh khớp 22 vs 24 joints
    j_pred = pred_tensor.shape[1]
    j_target = target_tensor.shape[1]

    if j_pred != j_target:
        if j_pred == 22 and j_target == 24:
            pred_tensor = convert_joints22_to_24(pred_tensor)
        elif j_pred == 24 and j_target == 22:
            target_tensor = convert_joints22_to_24(target_tensor)
        elif j_pred in (22, 24) and j_target == 17:
            # Map SMPL 22/24 joints to H36M 17 joints (Pelvis, R_Hip, R_Knee, R_Ankle, L_Hip, L_Knee, L_Ankle, Spine, Neck, Head, Head, L_Shoulder, L_Elbow, L_Wrist, R_Shoulder, R_Elbow, R_Wrist)
            smpl_to_h36m17 = [0, 2, 5, 8, 1, 4, 7, 3, 12, 15, 15, 13, 16, 18, 14, 17, 19]
            pred_tensor = pred_tensor[:, smpl_to_h36m17, :]
        elif j_pred == 17 and j_target in (22, 24):
            smpl_to_h36m17 = [0, 2, 5, 8, 1, 4, 7, 3, 12, 15, 15, 13, 16, 18, 14, 17, 19]
            target_tensor = target_tensor[:, smpl_to_h36m17, :]
        else:
            min_j = min(j_pred, j_target)
            pred_tensor = pred_tensor[:, :min_j, :]
            target_tensor = target_tensor[:, :min_j, :]

    num_frames, num_joints, _ = pred_tensor.shape

    # 3. Tự động quy đổi đơn vị: nếu ma trận GT tính bằng mm (> 50.0) còn pred tính bằng m (<= 50.0) -> chuyển GT về m
    if target_tensor.abs().max() > 50.0 and pred_tensor.abs().max() <= 50.0:
        target_tensor = target_tensor / 1000.0
    elif pred_tensor.abs().max() > 50.0 and target_tensor.abs().max() <= 50.0:
        pred_tensor = pred_tensor / 1000.0

    # 4. Kiểm tra pelvis_idxs có hợp lệ với num_joints hay không (nếu là 17 khớp thì dùng [0])
    pelvis_idxs = request.pelvis_idxs
    if max(pelvis_idxs) >= num_joints:
        pelvis_idxs = [0]

    dummy_verts_pred = torch.zeros((num_frames, 1, 3), dtype=pred_tensor.dtype, device=pred_tensor.device)
    dummy_verts_target = torch.zeros((num_frames, 1, 3), dtype=target_tensor.dtype, device=target_tensor.device)

    try:
        # Align by pelvis (Root Alignment)
        pred_aligned, target_aligned, _, _ = batch_align_by_pelvis(
            [pred_tensor, target_tensor, dummy_verts_pred, dummy_verts_target],
            pelvis_idxs=pelvis_idxs
        )

        scale_factor = 1000.0 if request.unit.lower() == "mm" else 1.0

        # MPJPE (Pelvis root aligned error)
        mpjpe_per_frame = compute_jpe(pred_aligned, target_aligned) * scale_factor

        # PA-MPJPE (Procrustes rigid alignment error)
        if num_frames in [2, 3]:
            S1 = pred_aligned.permute(0, 2, 1)
            S2 = target_aligned.permute(0, 2, 1)
            mu1 = S1.mean(dim=-1, keepdim=True)
            mu2 = S2.mean(dim=-1, keepdim=True)
            X1 = S1 - mu1
            X2 = S2 - mu2
            var1 = torch.sum(X1**2, dim=1).sum(dim=1)
            K = X1.bmm(X2.permute(0, 2, 1))
            U, s, V = torch.svd(K)
            Z = torch.eye(U.shape[1], device=S1.device).unsqueeze(0).repeat(U.shape[0], 1, 1)
            Z[:, -1, -1] *= torch.sign(torch.det(U.bmm(V.permute(0, 2, 1))))
            R = V.bmm(Z.bmm(U.permute(0, 2, 1)))
            scale = torch.cat([torch.trace(x).unsqueeze(0) for x in R.bmm(K)]) / var1
            t = mu2 - (scale.unsqueeze(-1).unsqueeze(-1) * (R.bmm(mu1)))
            S1_hat = (scale.unsqueeze(-1).unsqueeze(-1) * R.bmm(S1) + t).permute(0, 2, 1)
        else:
            S1_hat = batch_compute_similarity_transform_torch(pred_aligned, target_aligned)

        pa_mpjpe_per_frame = compute_jpe(S1_hat, target_aligned) * scale_factor

        return {
            "status": "success",
            "unit": request.unit,
            "num_frames": int(num_frames),
            "num_joints": int(num_joints),
            "mpjpe": {
                "mean": float(np.mean(mpjpe_per_frame)),
                "min": float(np.min(mpjpe_per_frame)),
                "max": float(np.max(mpjpe_per_frame)),
                "std": float(np.std(mpjpe_per_frame)),
                "per_frame": [float(val) for val in mpjpe_per_frame]
            },
            "pa_mpjpe": {
                "mean": float(np.mean(pa_mpjpe_per_frame)),
                "min": float(np.min(pa_mpjpe_per_frame)),
                "max": float(np.max(pa_mpjpe_per_frame)),
                "std": float(np.std(pa_mpjpe_per_frame)),
                "per_frame": [float(val) for val in pa_mpjpe_per_frame]
            }
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Lỗi khi tính toán chỉ số MPJPE/PA-MPJPE: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)



        
