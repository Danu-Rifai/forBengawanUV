#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class YoloTRTNode:
    def __init__(self, engine_path):
        rospy.init_node('yolo_tensorrt_node', anonymous=True)
        self.bridge = CvBridge()
        
        # Inisialisasi TensorRT
        self.logger = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(self.logger, '')
        
        rospy.loginfo("Memuat TensorRT Engine...")
        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        
        # Alokasi Memori (Host dan Device)
        self.inputs = []
        self.outputs = []
        self.bindings = []
        self.stream = cuda.Stream()
        
        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding)) * self.engine.max_batch_size
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            self.bindings.append(int(device_mem))
            
            if self.engine.binding_is_input(binding):
                self.inputs.append({'host': host_mem, 'device': device_mem})
            else:
                self.outputs.append({'host': host_mem, 'device': device_mem})
                
        rospy.loginfo("Engine berhasil dimuat. Menginisialisasi Subscriber...")
        
        # Sesuaikan dengan topic kamera robot Anda
        self.image_sub = rospy.Subscriber("/camera/image_raw", Image, self.image_callback, queue_size=1)
        self.image_pub = rospy.Publisher("/yolo/result_image", Image, queue_size=1)

    def preprocess_image(self, img):
        # Resize ke ukuran standar YOLO (640x640) dan normalisasi
        img_resized = cv2.resize(img, (640, 640))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_normalized = img_rgb.astype(np.float32) / 255.0
        # Ubah format HWC (Height, Width, Channel) ke CHW (Channel, Height, Width)
        img_chw = np.transpose(img_normalized, (2, 0, 1))
        # Tambahkan dimensi batch
        return np.expand_dims(img_chw, axis=0).ravel()

    def image_callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except Exception as e:
            rospy.logerr(f"Gagal mengkonversi gambar: {e}")
            return

        # 1. Pre-processing
        input_data = self.preprocess_image(cv_image)
        np.copyto(self.inputs[0]['host'], input_data)

        # 2. Transfer Data ke GPU (Host to Device)
        cuda.memcpy_htod_async(self.inputs[0]['device'], self.inputs[0]['host'], self.stream)

        # 3. Eksekusi Inferensi via CUDA
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)

        # 4. Transfer Hasil kembali ke CPU (Device to Host)
        cuda.memcpy_dtoh_async(self.outputs[0]['host'], self.outputs[0]['device'], self.stream)
        self.stream.synchronize()

        # 5. Output mentah hasil inferensi (Post-processing NMS diperlukan di sini)
        trt_outputs = self.outputs[0]['host']
        
        # (Opsional) Implementasikan Non-Maximum Suppression (NMS) pada array trt_outputs 
        # untuk menggambar Bounding Box, lalu publish hasilnya.
        
        # Publish gambar asli sementara sebagai penanda node berjalan
        result_msg = self.bridge.cv2_to_imgmsg(cv_image, "bgr8")
        self.image_pub.publish(result_msg)

if __name__ == '__main__':
    # Pastikan path mengarah ke file .engine yang Anda buat sebelumnya
    ENGINE_PATH = "/home/username_jetson/yolov8n_fp16.engine" 
    try:
        node = YoloTRTNode(ENGINE_PATH)
        rospy.spin()
    except rospy.ROSInterruptException:
        pass