# -*- coding: utf-8 -*-
import os
import cv2
import math
import torch
import numpy as np
import torch.nn.functional as F

from src.model_lib.MiniFASNet import MiniFASNetV1, MiniFASNetV2, MiniFASNetV1SE, MiniFASNetV2SE
from src.data_io import transform as trans
from src.utility import get_kernel, parse_model_name

MODEL_MAPPING = {
    'MiniFASNetV1': MiniFASNetV1,
    'MiniFASNetV2': MiniFASNetV2,
    'MiniFASNetV1SE': MiniFASNetV1SE,
    'MiniFASNetV2SE': MiniFASNetV2SE
}

class AntiSpoofPredict(object):
    def __init__(self, device_id):
        self.device = torch.device("cuda:{}".format(device_id)
                                   if torch.cuda.is_available() else "cpu")
        
        # --- PERBAIKAN 1: Auto-Detect & Load Model di Init ---
        model_dir = "./resources/anti_spoof_models"
        try:
            # Ambil file pertama yang berakhiran .pth
            model_name = [f for f in os.listdir(model_dir) if f.endswith('.pth')][0]
            self.model_path = os.path.join(model_dir, model_name)
            print(f"[AntiSpoof] Loading model: {self.model_path}")
        except IndexError:
            raise FileNotFoundError(f"Tidak ada file .pth di folder {model_dir}")

        self._load_model(self.model_path)

    def _load_model(self, model_path):
        # define model
        model_name = os.path.basename(model_path)
        h_input, w_input, model_type, _ = parse_model_name(model_name)
        self.kernel_size = get_kernel(h_input, w_input,)
        self.model = MODEL_MAPPING[model_type](conv6_kernel=self.kernel_size).to(self.device)

        # load model weight
        state_dict = torch.load(model_path, map_location=self.device)
        keys = iter(state_dict)
        first_layer_name = keys.__next__()
        if first_layer_name.find('module.') >= 0:
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for key, value in state_dict.items():
                name_key = key[7:]
                new_state_dict[name_key] = value
            self.model.load_state_dict(new_state_dict)
        else:
            self.model.load_state_dict(state_dict)
        
        self.model.eval() # Set ke mode evaluasi sekali saja
        return None

    # --- PERBAIKAN 2: Logic Cropping Internal ---
    def get_crop_box(self, box, scale):
        x, y, w, h = box
        size_bb = int(max(w, h) * scale)
        center_x, center_y = x + w // 2, y + h // 2
        
        # Hitung koordinat crop baru
        x1 = max(int(center_x - size_bb // 2), 0)
        y1 = max(int(center_y - size_bb // 2), 0)
        
        # Check batas gambar nanti dilakukan saat slicing
        return x1, y1, size_bb

    def predict(self, img, bbox):
        """
        img: Full frame image
        bbox: [x, y, w, h] dari detection sebelumnya
        """
        # Standar scaling MiniFASNet adalah 2.7
        scale = 2.7 
        
        # 1. Crop Wajah dengan Scale
        x1, y1, size_bb = self.get_crop_box(bbox, scale)
        
        # Handle boundary check agar tidak error jika crop keluar frame
        h_img, w_img, _ = img.shape
        x2 = min(x1 + size_bb, w_img)
        y2 = min(y1 + size_bb, h_img)
        
        # Crop
        cropped_face = img[y1:y2, x1:x2]
        
        # Jika hasil crop kosong/terlalu kecil, kembalikan fake agar aman
        if cropped_face.shape[0] < 10 or cropped_face.shape[1] < 10:
            return [[0.0, 0.0]] # Format dummy result

        # 2. Resize ke 80x80 (Sesuai spesifikasi model default)
        cropped_face = cv2.resize(cropped_face, (80, 80))

        # 3. Transform & Inference
        test_transform = trans.Compose([
            trans.ToTensor(),
        ])
        img_tensor = test_transform(cropped_face)
        img_tensor = img_tensor.unsqueeze(0).to(self.device)

        with torch.no_grad():
            result = self.model.forward(img_tensor)
            result = F.softmax(result, dim=1).cpu().numpy()
        
        return result