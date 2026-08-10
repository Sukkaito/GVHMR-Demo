import unittest
import torch
import numpy as np
from fastapi.testclient import TestClient

from importlib.util import spec_from_file_location, module_from_spec
import sys
from pathlib import Path

# Load gvhmr-api module dynamically
api_path = Path(__file__).resolve().parents[2] / "gvhmr-api.py"
spec = spec_from_file_location("gvhmr_api", str(api_path))
gvhmr_api = module_from_spec(spec)
sys.modules["gvhmr_api"] = gvhmr_api
spec.loader.exec_module(gvhmr_api)

app = gvhmr_api.app
API_KEY = gvhmr_api.API_KEY or "secret_key"
gvhmr_api.API_KEY = "test_api_key"

client = TestClient(app)
HEADERS = {"X-API-Key": "test_api_key"}


class TestMetricsAPI(unittest.TestCase):

    def test_auth_failure(self):
        response = client.post("/api/v1/metrics/evaluate", json={})
        self.assertEqual(response.status_code, 401)

    def test_identical_poses(self):
        # Create 10 frames, 24 joints, 3D coordinates
        num_frames = 10
        num_joints = 24
        joints = np.random.randn(num_frames, num_joints, 3).tolist()

        payload = {
            "pred_j3d": joints,
            "target_j3d": joints,
            "unit": "mm"
        }
        response = client.post("/api/v1/metrics/evaluate", headers=HEADERS, json=payload)
        self.assertEqual(response.status_code, 200, response.text)
        data = response.json()

        self.assertEqual(data["status"], "success")
        self.assertEqual(data["num_frames"], num_frames)
        self.assertEqual(data["num_joints"], num_joints)
        self.assertAlmostEqual(data["mpjpe"]["mean"], 0.0, places=3)
        self.assertAlmostEqual(data["pa_mpjpe"]["mean"], 0.0, places=3)

    def test_translated_poses(self):
        # Shifted by translation vector -> root alignment should cancel it
        num_frames = 5
        num_joints = 24
        target = np.random.randn(num_frames, num_joints, 3)
        pred = target + np.array([1.5, -2.0, 3.2])

        payload = {
            "pred_j3d": pred.tolist(),
            "target_j3d": target.tolist(),
            "unit": "mm"
        }
        response = client.post("/api/v1/metrics/evaluate", headers=HEADERS, json=payload)
        self.assertEqual(response.status_code, 200, response.text)
        data = response.json()

        self.assertAlmostEqual(data["mpjpe"]["mean"], 0.0, places=2)
        self.assertAlmostEqual(data["pa_mpjpe"]["mean"], 0.0, places=2)

    def test_scaled_and_rotated_poses(self):
        # Scale and rotation -> MPJPE > 0, PA-MPJPE ~ 0
        num_frames = 8
        num_joints = 24
        target = np.random.randn(num_frames, num_joints, 3)
        
        # Scale by 1.2
        pred = target * 1.2

        payload = {
            "pred_j3d": pred.tolist(),
            "target_j3d": target.tolist(),
            "unit": "mm"
        }
        response = client.post("/api/v1/metrics/evaluate", headers=HEADERS, json=payload)
        self.assertEqual(response.status_code, 200, response.text)
        data = response.json()

        self.assertGreater(data["mpjpe"]["mean"], 0.0)
        self.assertAlmostEqual(data["pa_mpjpe"]["mean"], 0.0, places=2)
        self.assertLess(data["pa_mpjpe"]["mean"], data["mpjpe"]["mean"])

    def test_shape_mismatch(self):
        # 2D coordinates instead of 3D (last dim is 2 instead of 3) -> should return 400
        payload = {
            "pred_j3d": np.random.randn(5, 24, 2).tolist(),
            "target_j3d": np.random.randn(5, 24, 2).tolist(),
        }
        response = client.post("/api/v1/metrics/evaluate", headers=HEADERS, json=payload)
        self.assertEqual(response.status_code, 400)

    def test_target_file_path_npy(self):
        import tempfile
        num_frames = 6
        num_joints = 24
        gt_matrix = np.random.randn(num_frames, num_joints, 3)

        with tempfile.NamedTemporaryFile(suffix="_h36m.npy", delete=False) as tmp:
            np.save(tmp.name, gt_matrix)
            tmp_path = tmp.name

        try:
            payload = {
                "pred_j3d": gt_matrix.tolist(),
                "target_file_path": tmp_path,
                "unit": "mm"
            }
            response = client.post("/api/v1/metrics/evaluate", headers=HEADERS, json=payload)
            self.assertEqual(response.status_code, 200, response.text)
            data = response.json()
            self.assertAlmostEqual(data["mpjpe"]["mean"], 0.0, places=2)
        finally:
            if Path(tmp_path).exists():
                Path(tmp_path).unlink()

    def test_joint_22_vs_24_conversion(self):
        # 22 joints pred, 24 joints target -> auto converts to 24 joints
        pred_22 = np.random.randn(5, 22, 3)
        target_24 = np.random.randn(5, 24, 3)

        payload = {
            "pred_j3d": pred_22.tolist(),
            "target_j3d": target_24.tolist(),
            "unit": "mm"
        }
        response = client.post("/api/v1/metrics/evaluate", headers=HEADERS, json=payload)
        self.assertEqual(response.status_code, 200, response.text)
        data = response.json()
        self.assertEqual(data["status"], "success")
        self.assertEqual(data["num_joints"], 24)


    def test_h36m_gt_file_eval(self):
        h36m_path = "D:/GVHMR-Demo/dataset/S1/Axel_1_cam_1_h36m.npy"
        if Path(h36m_path).exists():
            gt_data = np.load(h36m_path, allow_pickle=True)
            num_frames = gt_data.shape[0]
            # Create dummy pred in meters (GT is in mm)
            pred_m = (gt_data / 1000.0) + np.random.normal(0, 0.05, gt_data.shape)
            payload = {
                "pred_j3d": pred_m.tolist(),
                "target_file_path": h36m_path,
                "unit": "mm"
            }
            response = client.post("/api/v1/metrics/evaluate", headers=HEADERS, json=payload)
            self.assertEqual(response.status_code, 200, response.text)
            data = response.json()
            self.assertEqual(data["status"], "success")
            self.assertEqual(data["num_frames"], num_frames)
            self.assertGreater(data["mpjpe"]["mean"], 0.0)


if __name__ == "__main__":
    unittest.main()


