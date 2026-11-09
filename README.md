🛒 ShelfSense — Retail Shelf Product Detection Using YOLO

ShelfSense checks a supermarket shelf from a single photo. It detects every product, finds the empty spaces, calculates how full the shelf is, and reports whether the shelf needs restocking. Results are shown in a simple Gradio web app.

Subject Based Mini Project — Deep Learning & Reinforcement Learning Department of Artificial Intelligence and Machine Learning, P.A. College of Engineering, Mangalore

ShelfSense result

✨ Features
Product detection — finds bottles, cans, cartons and boxes of any brand (green boxes)
Empty-space detection — marks gaps where products are missing (red boxes)
Overlap filter — ignores "empty" boxes that are mostly covered by a product
Shelf occupancy % and stock status: Fully stocked / Partially stocked / Restock urgently
Gradio web app with adjustable confidence sliders and a public demo link
🧠 How It Works
Shelf photo ──► Resize to 640×640 ──┬──► YOLOv8s Product detector (SKU-110K)    ──┐
                                    └──► YOLOv8s Empty-space detector (Policenta) ──┤
                                                                                    ▼
                       Overlap filter ──► Shelf analytics ──► Status + annotated image

Shelf occupancy = product area ÷ (product area + empty area) × 100

Occupancy	Status
≥ 90%	✅ Fully stocked
60% – 90%	🟡 Partially stocked
< 60%	🔴 Restock urgently

📊 Datasets
Model	Dataset	Details
Product detector	SKU-110K (CVPR 2019)	11,743 densely packed shelf images, 1 class; 20% of the training set used
Empty-space detector	Empty shelf detection — Policenta (Roboflow Universe)	1,050 shelf images, class Empty-space

Datasets are not included in this repository. They download automatically when you run the notebook.

⚙️ Training Configuration
Parameter	Value
Model	YOLOv8s, pretrained on COCO (transfer learning)
Image size	640 × 640
Epochs (max)	25 (product), 50 (empty-space)
Batch size	16
Early stopping patience	8 (product), 12 (empty-space)
Confidence threshold	0.30 (product), 0.35 (empty-space)
Hardware	Google Colab, NVIDIA T4 GPU

📈 Results
Model	mAP@0.5	mAP@0.5:0.95	Precision	Recall
Product detector (SKU-110K)	____	____	____	____
Empty-space detector	____	____	____	____
Test image	Products	Empty regions	Occupancy	Status
Soft-drink aisle	84	59	38.4%	🔴 Restock urgently
Pharmacy shelf	160	39	61.5%	🟡 Partially stocked

Gradio interface

🗂️ Repository Structure
ShelfSense-Retail-Shelf-Detection/
├── ShelfSense_v2_YOLOv8.ipynb   # Training, evaluation, analysis and Gradio demo
├── weights/
│   ├── products_best.pt         # Trained product detector
│   └── empty_space_best.pt      # Trained empty-space detector
├── results/                     # Output images, training curves, confusion matrices
├── report/
│   └── Retail_Shelf_Product_Detection_Report.pdf
└── README.md

🚀 How to Run
Open ShelfSense_v2_YOLOv8.ipynb in Google Colab.
Go to Runtime → Change runtime type → T4 GPU.
Get a free API key from Roboflow (Settings → API Keys) and paste it where the notebook says YOUR_ROBOFLOW_API_KEY.
Runtime → Run all.

To skip training: upload the files from weights/ to Colab, load them as shown below, then run the analysis and Gradio cells.

python
from ultralytics import YOLO
prod_model  = YOLO("weights/products_best.pt")
empty_model = YOLO("weights/empty_space_best.pt")
🛠️ Tech Stack

Python · Ultralytics YOLOv8 · PyTorch · OpenCV · NumPy · Pandas · Matplotlib · Roboflow · Gradio · Google Colab

⚠️ Limitations
Counts products but does not identify brands, because SKU-110K has a single class.
The empty-space model sometimes draws overlapping boxes on the same gap, which lowers the occupancy value.
Very small or heavily overlapping products can be missed at 640 px.
Only 20% of SKU-110K was used because of Colab time limits.

🔮 Future Scope
Brand recognition on product crops (classifier or CLIP embeddings)
Merge overlapping empty-space boxes before calculating area
Planogram compliance checking
Live CCTV video monitoring with automatic restock alerts
Export to ONNX / TensorRT for edge devices

📚 References
E. Goldman et al., "Precise Detection in Densely Packed Scenes", CVPR, 2019.
J. Redmon et al., "You Only Look Once: Unified, Real-Time Object Detection", CVPR, 2016.
Ultralytics YOLOv8 Documentation
Gradio Documentation

👤 Author
Farhad Shajahan V K
