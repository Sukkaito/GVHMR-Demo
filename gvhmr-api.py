
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
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


        
