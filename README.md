
# **iPhone WiFi People Counter Using YOLO**

This project uses an iPhone as a webcam to stream video via OBS (on macOS) and detect/count people in real time using the YOLOv4 object detection model.

---

## **Table of Contents**
1. [Features](#features)
2. [Project Setup](#project-setup)
3. [Installation](#installation)
4. [iPhone and OBS Configuration](#iphone-and-obs-configuration)
5. [Usage](#usage)
6. [Folder Structure](#folder-structure)
7. [Future Plans](#future-plans)
8. [License](#license)

---

## **Features**
- Streams video from an iPhone using OBS.
- Detects and counts people using the **YOLOv4** object detection model.
- Outputs real-time video with bounding boxes and a counter overlay.
- Easy setup with Python and OpenCV.

---

## **Project Setup**

### **1. Clone the Repository**
```bash
git clone https://github.com/philippogol/iphone-wifi-people-counter.git
cd iphone-wifi-people-counter
```

### **2. Virtual Environment Setup**
```bash
python3 -m venv venv
source venv/bin/activate
```

### **3. Install Required Libraries**
```bash
pip install -r requirements.txt
```

---

## **iPhone and OBS Configuration**

### **Step 1: Set Up the iPhone Webcam**
1. Download the **OBS Camera** app on your iPhone: [OBS Camera on App Store](https://apps.apple.com/us/app/camera-for-obs-studio/id1352834008).
2. Download and install the **OBS Camera driver** for macOS: [OBS Camera Driver](https://obs.camera/docs/getting-started/ios-camera-plugin-usb/_).
3. Connect your iPhone to your MacBook using a **USB cable**.

### **Step 2: Install OBS Studio for macOS**
1. Download OBS Studio for macOS: [OBS Studio 31.0.0 (Intel)](https://cdn-fastly.obsproject.com/downloads/obs-studio-31.0.0-macos-intel.dmg).
2. Open OBS and add a **Video Capture Device**:
   - Select **OBS Camera** as the source.

### **Step 3: Start OBS Virtual Camera**
- In OBS, click **“Start Virtual Camera”**.

### **Step 4: macOS Security Preferences**
1. Go to **System Preferences → Security & Privacy → Privacy**.  
2. Under **Camera**, ensure OBS and Python are allowed.  
3. Restart OBS if changes are made.

---

## **Usage**

### **Run the People Counter**
1. Activate the virtual environment:
   ```bash
   source venv/bin/activate
   ```

2. Download the YOLOv4 model files:
   - [Download YOLOv4 Package](https://www.tigers-agreement.com/material/YOLOv4-package.zip)
   - Unzip the files into the `yolo/` folder.

3. Run the YOLOv4 people counter script:
   ```bash
   python3 main-yolo4-counter.py
   ```

4. **Output**:
   - The video stream will display bounding boxes around detected people.
   - A **people counter** will show at the top-left corner of the window.

### **Run Other Test Scripts**
- Simple Camera Stream Test:
   ```bash
   python3 main-just-cam.py
   ```
- Simple People Counter (basic method):
   ```bash
   python3 main-simple-people-counter.py
   ```

---

## **Folder Structure**

```plaintext
iphone-wifi-people-counter/
│
├── camera_stream/         # Camera stream code
├── docs/                  # Documentation and images
├── people_counter/        # People counter logic
├── venv/                  # Python virtual environment (excluded)
├── yolo/                  # YOLO files (cfg, names, weights)
│   ├── yolov4.cfg
│   ├── yolov4.weights     # Download manually
│   └── coco.names
│
├── README.md              # Project documentation
├── requirements.txt       # Project dependencies
├── main-just-cam.py       # Camera stream test
├── main-simple-people-counter.py   # Simple people counter
└── main-yolo4-counter.py  # YOLOv4-based people counter
```

---

## **Future Plans**
- Upgrade to **YOLOv8** for improved performance and accuracy.
- Integrate logging functionality to save people counts with timestamps.
- Add plotting to visualize counts over time.

---

## **License**
This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

### **Notes**
- Ensure you download the YOLOv4 weights package from: [YOLOv4 Package](https://www.tigers-agreement.com/material/YOLOv4-package.zip).
- For GPU support, configure OpenCV to utilize CUDA (if available).
