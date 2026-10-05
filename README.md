\# FaceMood 🎭



\*\*FaceMood\*\* is a real-time facial expression recognition application built with Python, OpenCV, MediaPipe, NumPy, Pillow, and ONNX Runtime.



The application uses a webcam to detect faces, track facial landmarks, analyze facial expressions, and display a dynamic emoji-style avatar based on the detected expression.



\## ✨ Features



\* 🎥 Real-time webcam processing

\* 👤 Face detection and tracking

\* 📍 468-point facial landmark visualization

\* 😊 Facial expression recognition

\* 🎭 Dynamic emoji-style expression avatar

\* 📊 Emotion confidence score

\* ⚡ Real-time FPS monitoring

\* ⏱️ Landmark inference time monitoring

\* ⏱️ Emotion inference time monitoring

\* 🖥️ Fully local processing

\* 🔒 No cloud-based face processing



\## 🧠 Supported Expressions



FaceMood supports the following expression categories:



\* Neutral

\* Happy

\* Surprise

\* Sad

\* Angry

\* Disgust

\* Fear

\* Contempt



\## 🛠️ Technologies



| Technology   | Purpose                             |

| ------------ | ----------------------------------- |

| Python       | Core application                    |

| OpenCV       | Webcam processing and visualization |

| MediaPipe    | Face landmarks and facial analysis  |

| NumPy        | Numerical processing                |

| Pillow       | Image processing                    |

| ONNX Runtime | Local emotion model inference       |

| ONNX         | Emotion recognition model format    |



\## 📁 Project Structure



```text

FaceMood/

│

├── models/

│   ├── blaze\_face\_short\_range.tflite

│   ├── face\_landmarker.task

│   └── emotion-ferplus-8.onnx

│

├── src/

│   └── app.py

│

├── utils/

│

├── assets/

│

├── README.md

├── requirements.txt

└── .gitignore

```



\## 🚀 Installation



\### 1. Clone the repository



```bash

git clone <YOUR\_GITHUB\_REPOSITORY\_URL>

cd FaceMood

```



\### 2. Create a virtual environment



```bash

python -m venv venv

```



\### 3. Activate the virtual environment



\*\*Windows PowerShell:\*\*



```powershell

.\\venv\\Scripts\\Activate.ps1

```



\### 4. Install dependencies



```powershell

pip install -r requirements.txt

```



\## ▶️ Run the Application



From the project root:



```powershell

python src\\app.py

```



If `app.py` is located directly in the root directory:



```powershell

python app.py

```



\## 🎮 Controls



| Key | Action                     |

| --- | -------------------------- |

| `Q` | Quit application           |

| `R` | Reset expression smoothing |



\## 📊 Performance Monitoring



FaceMood displays real-time performance information including:



\* FPS

\* Face landmark inference time

\* Emotion inference time

\* Detected expression

\* Expression confidence



This makes the application useful not only as an expression recognition demo but also as a practical computer-vision performance monitoring project.



\## 🔐 Privacy



FaceMood is designed for local processing.



Webcam frames are processed locally on the user's machine and are not required to be uploaded to a cloud service for the application's core functionality.



\## 📦 Models



The project uses:



\* MediaPipe Face Landmarker for facial landmark detection.

\* FER+ ONNX model for facial expression classification.



Model files are stored locally inside the `models/` directory.



\## 🎯 Use Cases



FaceMood can be used for:



\* Computer vision demonstrations

\* Human-computer interaction

\* Emotion-aware interfaces

\* AI/ML portfolio projects

\* Real-time facial analysis

\* Educational computer-vision applications

\* Interactive avatar systems



\## 🔮 Future Improvements



Possible future enhancements include:



\* Multi-face emotion tracking

\* Expression history graphs

\* Emotion timeline visualization

\* More advanced facial expression scoring

\* Improved face alignment

\* GPU acceleration

\* Web-based interface

\* Emotion analytics dashboard

\* Personalized avatar animations



\## 👩‍💻 Author



\*\*Kainat Zafar\*\*



Full-Stack Developer / Generative AI Application Developer



GitHub: \[Add your GitHub profile]



LinkedIn: \[Add your LinkedIn profile]



\## 📄 License



This project is intended for educational, portfolio, and demonstration purposes.



