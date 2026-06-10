import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import threading
import time
import sys

# 80 Kelas standar COCO Dataset
CLASSES = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]

def load_engine(engine_path):
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    trt.init_libnvinfer_plugins(TRT_LOGGER, '')
    with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

def allocate_buffers(engine):
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

class CameraStream:
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Eksekusi bypass parameter Auto-Exposure (Jika didukung hardware)
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        self.ret, self.frame = self.cap.read()
        self.stopped = False
        self.lock = threading.Lock()

    def start(self):
        threading.Thread(target=self.update, daemon=True).start()
        return self
    
    def update(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            with self.lock:
                self.ret = ret
                self.frame = frame

    def read(self):
        with self.lock:
            return self.ret, self.frame.copy() if self.frame is not None else None
        
    def stop(self):
        self.stopped = True
        self.cap.release()

def main():
    # PATH ENGINE: Sesuaikan jika nama file berbeda
    ENGINE_PATH = "yolov8n_416_fp16.engine"
    
    print("[INFO] Memuat model TensorRT ke VRAM GPU...")
    engine = load_engine(ENGINE_PATH)
    context = engine.create_execution_context()
    inputs, outputs, bindings, stream = allocate_buffers(engine)
    print("[INFO] Model siap digunakan.")

    print("[INFO] Menyalakan Thread Kamera...")
    cam_stream = CameraStream(0).start()
    time.sleep(1.0) # Memberi waktu buffer kamera terisi

    if not cam_stream.ret:
        print('[ERROR] Kamera tidak terdeteksi. Program dihentikan.')
        sys.exit(1)

    print("\n" + "="*50)
    print("[INFO] BENCHMARK DIMULAI")
    print("[INFO] Program akan berjalan otomatis selama 60 detik.")
    print("[INFO] Dilarang menekan tombol apapun selama proses ini.")
    print("="*50 + "\n")
    
    # --- VARIABEL BENCHMARK ---
    BENCHMARK_DURATION = 60.0 # Waktu benchmark dalam detik (1 Menit)
    total_frames = 0
    benchmark_start_time = time.time()
    
    while True:
        current_time = time.time()
        elapsed_time = current_time - benchmark_start_time
        
        # Hentikan loop jika sudah mencapai 60 detik
        if elapsed_time >= BENCHMARK_DURATION:
            break

        loop_start_time = time.time()
        
        ret, frame = cam_stream.read()
        if not ret: 
            print("[WARN] Frame drop dari kamera.")
            continue

        orig_h, orig_w = frame.shape[:2]

        # --- 1. PRE-PROCESSING ---
        img = cv2.resize(frame, (416, 416))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0).ravel()

        # --- 2. INFERENSI CUDA ---
        np.copyto(inputs[0]['host'], img)
        cuda.memcpy_htod_async(inputs[0]['device'], inputs[0]['host'], stream)
        context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
        cuda.memcpy_dtoh_async(outputs[0]['host'], outputs[0]['device'], stream)
        stream.synchronize()

        # --- 3. POST-PROCESSING ---
        output_data = outputs[0]['host'].reshape(84, 3549).transpose()
        boxes_data = output_data[:, :4]
        probs_data = output_data[:, 4:]

        scores_array = np.max(probs_data, axis=1)
        class_ids_array = np.argmax(probs_data, axis=1)
        valid_indices = np.where(scores_array > 0.5)[0]

        boxes = []
        scores = []
        class_ids = []

        x_factor = orig_w / 416.0
        y_factor = orig_h / 416.0

        for i in valid_indices:
            cx, cy, w, h = boxes_data[i]
            left = int((cx - w/2) * x_factor)
            top = int((cy - h/2) * y_factor)
            width = int(w * x_factor)
            height = int(h * y_factor)
            
            boxes.append([left, top, width, height])
            scores.append(float(scores_array[i]))
            class_ids.append(int(class_ids_array[i]))

        # --- 4. NON-MAXIMUM SUPPRESSION (NMS) ---
        indices = cv2.dnn.NMSBoxes(boxes, scores, 0.5, 0.45)

        # --- 5. VISUALISASI ---
        if len(indices) > 0:
            for i in indices.flatten():
                box = boxes[i]
                left, top, w, h = box[0], box[1], box[2], box[3]
                cv2.rectangle(frame, (left, top), (left + w, top + h), (0, 255, 0), 2)
                label = f"{CLASSES[class_ids[i]]}: {scores[i]:.2f}"
                cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Hitung real-time FPS untuk ditampilkan di layar
        realtime_fps = 1.0 / (time.time() - loop_start_time)
        cv2.putText(frame, f"Sisa Waktu: {int(BENCHMARK_DURATION - elapsed_time)}s", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        cv2.putText(frame, f"FPS: {realtime_fps:.1f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imshow("ROV Jetson Benchmark", frame)
        cv2.waitKey(1)
        
        # Tambahkan hitungan total frame
        total_frames += 1

    # --- PENGHENTIAN PROGRAM & KALKULASI HASIL ---
    cam_stream.stop()
    cv2.destroyAllWindows()
    
    # Kalkulasi Matematika Final
    actual_elapsed_time = time.time() - benchmark_start_time
    average_fps = total_frames / actual_elapsed_time
    
    print("\n" + "="*50)
    print("[LAPORAN DIAGNOSTIK BENCHMARK ROV]")
    print(f"Durasi Aktual      : {actual_elapsed_time:.2f} detik")
    print(f"Total Frame Diproses : {total_frames} frame")
    print(f"AVERAGE FPS FINAL    : {average_fps:.2f} FPS")
    print("="*50 + "\n")
    
    if average_fps < 20.0:
        print("[ANALISIS]: Performa tertahan di bawah 20 FPS.")
        print("Jika Anda sudah menjalankan 'sudo nvpmodel -m 0' dan 'sudo jetson_clocks',")
        print("maka bottleneck 100% dipastikan berasal dari limitasi hardware kamera USB Anda")
        print("atau suhu CPU Jetson yang mengalami Thermal Throttling.")

if __name__ == '__main__':
    main()