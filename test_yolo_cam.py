import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import time

# 80 Kelas standar COCO Dataset
CLASSES = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]

def load_engine(engine_path):
    # Memuat file .engine ke dalam memori
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    trt.init_libnvinfer_plugins(TRT_LOGGER, '')
    with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

def allocate_buffers(engine):
    # Mengalokasikan memori antara CPU (Host) dan GPU (Device)
    inputs, outputs, bindings = [], [], []
    stream = cuda.Stream()
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        bindings.append(int(device_mem))
        if engine.binding_is_input(binding):
            inputs.append({'host': host_mem, 'device': device_mem})
        else:
            outputs.append({'host': host_mem, 'device': device_mem})
    return inputs, outputs, bindings, stream

def main():
    # PATH ENGINE: Pastikan file yolov8n_fp16.engine ada di folder yang sama
    ENGINE_PATH = "yolov8n_fp16.engine"
    
    print("[INFO] Memuat model TensorRT ke VRAM GPU...")
    engine = load_engine(ENGINE_PATH)
    context = engine.create_execution_context()
    inputs, outputs, bindings, stream = allocate_buffers(engine)
    print("[INFO] Model siap digunakan.")

    # Inisialisasi Kamera (0 untuk USB Webcam standar)
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("[ERROR] Kamera tidak terdeteksi. Pastikan kamera terpasang.")
        return

    print("[INFO] Memulai inferensi real-time. Tekan 'q' untuk keluar.")
    
    while True:
        start_time = time.time()
        ret, frame = cap.read()
        if not ret: 
            break

        orig_h, orig_w = frame.shape[:2]

        # --- 1. PRE-PROCESSING ---
        # YOLOv8 meminta input berukuran 640x640, format RGB, dinormalisasi 0-1
        img = cv2.resize(frame, (640, 640))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        # Mengubah struktur matriks dari [Height, Width, Channels] ke [Channels, Height, Width]
        img = np.transpose(img, (2, 0, 1))
        # Meratakan array untuk dimasukkan ke memori CUDA
        img = np.expand_dims(img, axis=0).ravel()

        # --- 2. INFERENSI CUDA ---
        # Copy gambar dari RAM CPU ke VRAM GPU
        np.copyto(inputs[0]['host'], img)
        cuda.memcpy_htod_async(inputs[0]['device'], inputs[0]['host'], stream)
        # Eksekusi model
        context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
        # Copy hasil dari VRAM GPU kembali ke RAM CPU
        cuda.memcpy_dtoh_async(outputs[0]['host'], outputs[0]['device'], stream)
        stream.synchronize()

        # --- 3. POST-PROCESSING ---
        # Output YOLOv8 adalah tensor berukuran [1, 84, 8400]
        # Kita ubah bentuknya agar lebih mudah diproses menjadi 8400 baris (prediksi) x 84 kolom (data)
        output_data = outputs[0]['host'].reshape(84, 8400).transpose()

        boxes = []
        scores = []
        class_ids = []

        # Faktor skala untuk mengembalikan ukuran bounding box ke resolusi asli kamera
        x_factor = orig_w / 640.0
        y_factor = orig_h / 640.0

        for row in output_data:
            prob = row[4:] # Kolom ke-4 sampai akhir adalah nilai probabilitas setiap kelas
            max_prob = np.max(prob)
            
            # Filter deteksi yang nilai kepercayaannya di atas 50%
            if max_prob > 0.5: 
                class_id = np.argmax(prob)
                cx, cy, w, h = row[0], row[1], row[2], row[3]
                
                # Konversi koordinat tengah ke koordinat sudut kiri atas
                left = int((cx - w/2) * x_factor)
                top = int((cy - h/2) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)
                
                boxes.append([left, top, width, height])
                scores.append(float(max_prob))
                class_ids.append(class_id)

        # --- 4. NON-MAXIMUM SUPPRESSION (NMS) ---
        # Menghapus kotak duplikat yang tumpang tindih pada objek yang sama
        indices = cv2.dnn.NMSBoxes(boxes, scores, 0.5, 0.45)

        # --- 5. VISUALISASI ---
        if len(indices) > 0:
            for i in indices.flatten():
                box = boxes[i]
                left, top, w, h = box[0], box[1], box[2], box[3]
                
                # Gambar kotak
                cv2.rectangle(frame, (left, top), (left + w, top + h), (0, 255, 0), 2)
                
                # Tulis label kelas dan persentase
                label = f"{CLASSES[class_ids[i]]}: {scores[i]:.2f}"
                cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Kalkulasi dan tampilkan FPS
        fps = 1.0 / (time.time() - start_time)
        cv2.putText(frame, f"FPS: {fps:.1f} (TensorRT FP16)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # Tampilkan output
        cv2.imshow("YOLOv8 Murni Jetson Nano", frame)
        
        # Tekan 'q' untuk keluar
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Bersihkan memori kamera dan tutup jendela GUI
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()