import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import threading
import time

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
        # Tambahkan API Preference cv2.CAP_V4L2 untuk Linux/Jetson
        self.cap = cv2.VideoCapture(src, cv2.CAP_V4L2)
        # self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'YUYV'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

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

def process_and_infer(frame, context, inputs, outputs, bindings, stream):
    """
    Fungsi untuk menangani pre-processing, inferensi, dan anotasi pada satu frame.
    Menggunakan buffer PyCUDA yang sama untuk efisiensi VRAM.
    """
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
    stream.synchronize() # Paksa sinkronisasi sebelum mengambil output

    # --- 3. POST-PROCESSING ---
    output_data = outputs[0]['host'].reshape(84, 3549).transpose()
    boxes_data = output_data[:, :4]
    probs_data = output_data[:, 4:]

    scores_array = np.max(probs_data, axis=1)
    class_ids_array = np.argmax(probs_data, axis=1)
    valid_indices = np.where(scores_array > 0.5)[0]

    boxes, scores, class_ids = [], [], []
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
            
    return frame

def main():
    ENGINE_PATH = "yolov8n_416_fp16.engine"
    
    print("[INFO] Memuat model TensorRT ke VRAM GPU...")
    engine = load_engine(ENGINE_PATH)
    context = engine.create_execution_context()
    inputs, outputs, bindings, stream = allocate_buffers(engine)
    print("[INFO] Model siap digunakan.")

    # --- KONFIGURASI DUAL KAMERA ---
    print("[INFO] Menyalakan Thread Kamera 1 dan Kamera 2...")
    cam1 = CameraStream(0).start()
    cam2 = CameraStream(1).start() # Pastikan index '1' sesuai dengan hardware (bisa jadi 2 atau 4 di Linux)
    time.sleep(1.0) 

    if not cam1.ret or not cam2.ret:
        print('[ERROR] Salah satu atau kedua kamera tidak terdeteksi. Cek koneksi USB/CSI.')
        return

    print("[INFO] Memulai inferensi real-time dual camera. Tekan 'q' untuk keluar.")
    
    while True:
        start_time = time.time()
        
        # Ambil frame secara paralel dari kedua thread
        ret1, frame1 = cam1.read()
        ret2, frame2 = cam2.read()
        
        if not ret1 or not ret2: 
            break

        # --- EKSEKUSI SEKUENSIAL DENGAN BUFFER GPU YANG SAMA ---
        frame1_out = process_and_infer(frame1, context, inputs, outputs, bindings, stream)
        frame2_out = process_and_infer(frame2, context, inputs, outputs, bindings, stream)

        # --- OPTIMASI RENDER UI ---
        # Gabungkan kedua frame secara horizontal agar hanya butuh satu instance cv2.imshow
        combined_frame = cv2.hconcat([frame1_out, frame2_out])

        fps = 1.0 / (time.time() - start_time)
        cv2.putText(combined_frame, f"Total FPS: {fps:.1f} (Dual Cam)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imshow("Dual YOLOv8 TensorRT", combined_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam1.stop()
    cam2.stop()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()