# 🛡️ DriveShield – Real-Time Collision Risk Intelligence Platform

**EN:** End-to-end collision prediction platform powered by **Nexar's BADAS-Open** model and the **Nexar Dashcam Collision Prediction** dataset.  
**FA:** پلتفرم پیش‌بینی تصادف سرتاسری با استفاده از مدل **BADAS-Open** شرکت Nexar و دیتاست **Nexar Dashcam Collision Prediction**.

> ⚠️ **EN:** Research & demo only – not for safety-critical or real-world driving decisions.  
> ⚠️ **FA:** فقط برای تحقیق و دمو – برای تصمیم‌گیری در رانندگی واقعی استفاده نکنید.

---

## 🎬 **Live Demo / دمو زنده**

**EN:** Watch the DriveShield dashboard in action:  
**FA:** مشاهده داشبورد DriveShield در عمل:

![DriveShield UI Demo](https://raw.githubusercontent.com/MahdiNavaei/DriveShield/main/driveshield-demo.gif)

**EN:** The GIF demonstrates the complete workflow: video upload, real-time prediction, bilingual interface (EN/FA toggle), and interactive risk timeline visualization.  
**FA:** این GIF فرآیند کامل را نشان می‌دهد: آپلود ویدئو، پیش‌بینی در لحظه، رابط دوزبانه (تغییر EN/FA) و مصورسازی تعاملی نمودار زمانی ریسک.

---

## 💼 **For Recruiters / برای کارفرمایان**

**EN:** If you want to quickly understand my ownership and contribution to this project:  
**FA:** اگر می‌خواهید به سرعت درک کنید که من چه نقشی در این پروژه داشته‌ام:

### **What I Built / آنچه که ساختم**

- ✅ **Full architecture design and implementation** – End-to-end system design from data ingestion to user interface
- ✅ **Backend API development** – Production-ready FastAPI REST API with async endpoints, error handling, and dependency injection
- ✅ **Frontend development** – Modern React + TypeScript application with Tailwind CSS, responsive design, and state management
- ✅ **Model integration** – Successfully integrated state-of-the-art vision-language model (BADAS-Open) with local offline inference
- ✅ **Evaluation pipelines** – Comprehensive benchmarking system with industry-standard metrics (AUC-ROC, AP, Precision/Recall)
- ✅ **UX/UI design** – Bilingual user interface (English/Persian) with real-time language switching and interactive data visualization

### **Technical Skills Demonstrated / مهارت‌های فنی نشان داده شده**

- 🔬 **Vision ML & Deep Learning** – Working with cutting-edge computer vision models (V-JEPA2 backbone)
- 🚀 **MLOps Mindset** – Production-grade ML systems with proper error handling, logging, and monitoring
- 🏗️ **Production Engineering** – Scalable backend architecture, clean code practices, type safety
- 🌐 **Full-Stack Development** – Seamless integration between Python backend and React frontend
- 📊 **Data Science** – Evaluation metrics, statistical analysis, and visualization pipelines

---

## 📄 **Related Research / تحقیقات مرتبط**

**EN:** This project is built on state-of-the-art research in collision prediction and autonomous driving:  
**FA:** این پروژه بر پایه تحقیقات پیشرفته در پیش‌بینی تصادف و رانندگی خودران بنا شده است:

- **Goldshmidt et al. (2025)** – **BADAS: Context Aware Collision Prediction Using Real-World Dashcam Data**  
  *[arXiv: Coming Soon]* | [Hugging Face Model](https://huggingface.co/nexar-ai/BADAS-Open)

- **Meta AI (2024)** – **Video Joint-Embedding Predictive Architecture (V-JEPA)** – The foundational vision transformer used in BADAS-Open

**EN:** The BADAS-Open model represents the current state-of-the-art in ego-centric collision prediction, achieving superior performance on the Nexar Dashcam Collision Prediction dataset through its innovative use of self-supervised learning and context-aware temporal modeling.  
**FA:** مدل BADAS-Open نمایانگر آخرین پیشرفت در پیش‌بینی تصادف از دیدگاه خودرو است که از طریق استفاده نوآورانه از یادگیری خودنظارتی و مدل‌سازی زمانی آگاه از زمینه، عملکرد برتری روی دیتاست Nexar Dashcam Collision Prediction به دست می‌آورد.

---

## ✨ **Key Features / ویژگی‌های کلیدی**

### 🎯 **State-of-the-Art Collision Prediction Engine**
**EN:** Uses **Nexar's BADAS-Open model** (V-JEPA2 backbone) for ego-centric collision & near-miss detection with cutting-edge vision-language understanding.  
**FA:** از **مدل BADAS-Open شرکت Nexar** (با بک‌بون V-JEPA2) برای تشخیص تصادف و نزدیک‌به‌تصادف از دیدگاه خودرو استفاده می‌کند.

### 🔒 **100% Local, Offline Inference**
**EN:** All model weights and configurations are loaded from local files – **zero HuggingFace API calls at runtime**. Perfect for air-gapped environments and production deployments.  
**FA:** تمام وزن‌های مدل و تنظیمات از فایل‌های محلی بارگذاری می‌شوند – **بدون هیچ فراخوانی HuggingFace در زمان اجرا**. مناسب برای محیط‌های ایزوله و استقرارهای production.

### 🚀 **Production-Ready FastAPI Backend**
**EN:** Scalable REST API with health checks, full prediction endpoints, and optimized response models.  
**FA:** REST API مقیاس‌پذیر با بررسی سلامت، endpointهای پیش‌بینی کامل و مدل‌های پاسخ بهینه‌شده.

- `GET /status` – Health check with model readiness status
- `POST /predict/badas` – Full risk timeline with per-frame probabilities
- `POST /predict/badas/summary` – Lightweight summary endpoint

### 🌐 **Bilingual Web Dashboard (English / فارسی)**
**EN:** Modern, responsive React dashboard with **real-time language switching** and dark theme.  
**FA:** داشبورد React مدرن و واکنش‌گرا با **تغییر زبان در لحظه** و تم تاریک.

**EN:** Key UI features:  
**FA:** ویژگی‌های کلیدی رابط کاربری:

- **Language toggle (EN / FA)** – Instant UI translation
- **Dark, modern design** – Tailwind CSS with slate color scheme
- **Video upload interface** – Drag-and-drop or file selection
- **Risk level badges** – Visual indicators (Safe / Medium / High)
- **Prediction summary card** – Comprehensive results overview
- **Interactive risk timeline chart** – Per-frame probability visualization

### 📊 **Advanced Risk Visualization**
**EN:** **Recharts-powered line chart** showing collision probability over time with threshold overlay, enabling visual analysis of high-risk moments.  
**FA:** **نمودار خطی با Recharts** که احتمال تصادف در طول زمان را با خط آستانه نمایش می‌دهد و امکان تحلیل بصری لحظات پرخطر را فراهم می‌کند.

### 📈 **Comprehensive Evaluation Pipeline**
**EN:** Automated benchmarking system that computes industry-standard metrics and generates publication-ready visualizations.  
**FA:** سیستم ارزیابی خودکار که متریک‌های استاندارد صنعتی را محاسبه کرده و نمودارهای آماده انتشار تولید می‌کند.

- **AUC-ROC** – Area under the receiver operating characteristic curve
- **Average Precision (AP)** – Precision-recall area under curve
- **Precision & Recall @ threshold** – Configurable threshold metrics
- **mTTA (approximate)** – Mean Time-To-Accident estimation
- **Automated plotting** – ROC and Precision-Recall curves

---

## 🏗️ **Architecture Overview / نمای کلی معماری**

**EN:** DriveShield follows a clean, production-ready architecture with clear separation between components:  
**FA:** DriveShield از یک معماری تمیز و آماده production با جداسازی واضح بین کامپوننت‌ها پیروی می‌کند:

```
┌─────────────────────────────────────────────────────────────────────┐
│                         🌐 User Interface Layer                      │
│  ┌───────────────────────────────────────────────────────────────┐ │
│  │   React 19 + TypeScript + Tailwind CSS                        │ │
│  │   • Bilingual UI (EN/FA toggle)                               │ │
│  │   • Video Upload (drag-and-drop)                              │ │
│  │   • Real-time Risk Visualization (Recharts)                   │ │
│  └───────────────────────────────────────────────────────────────┘ │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
                    HTTP/REST (JSON)
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      ⚡ API Layer (FastAPI)                          │
│  ┌───────────────────────────────────────────────────────────────┐ │
│  │   Endpoints:                                                  │ │
│  │   • GET  /status              → Health check                  │ │
│  │   • POST /predict/badas       → Full prediction timeline      │ │
│  │   • POST /predict/badas/summary → Lightweight summary         │ │
│  └───────────────────────────────────────────────────────────────┘ │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
                     Service Calls
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   🧠 ML Inference Layer                              │
│  ┌───────────────────────────────────────────────────────────────┐ │
│  │   BADAS-Open Model (Local Weights, V-JEPA2 Backbone)          │ │
│  │   • Ego-centric collision prediction                          │ │
│  │   • Per-frame probability estimation                          │ │
│  │   • Temporal context awareness                                │ │
│  │   • Zero external API calls (100% offline)                    │ │
│  └───────────────────────────────────────────────────────────────┘ │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
                     Video Processing
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   📹 Input Layer                                     │
│  ┌───────────────────────────────────────────────────────────────┐ │
│  │   Dashcam Video Files (.mp4, .avi, etc.)                      │ │
│  │   • Frame extraction                                           │ │
│  │   • Preprocessing pipeline                                     │ │
│  └───────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
```

**EN:** **Deployment Architecture:** The system can be deployed as a monolith (single server) or with separated backend/frontend (e.g., FastAPI on port 8002, React on port 5173). All model weights are stored locally, enabling air-gapped deployments.  
**FA:** **معماری استقرار:** سیستم می‌تواند به صورت monolithic (سرور واحد) یا با جداسازی backend/frontend (مثلاً FastAPI روی پورت 8002، React روی پورت 5173) استقرار یابد. تمام وزن‌های مدل به صورت محلی ذخیره می‌شوند که امکان استقرار در محیط‌های ایزوله را فراهم می‌کند.

---

## 🛠️ **Tech Stack / فناوری‌های استفاده شده**

### **Backend / بک‌اند**
- **Python 3.10+** – Modern Python features and type hints
- **FastAPI** – High-performance async web framework
- **Uvicorn** – ASGI server for production deployment
- **PyTorch** – Deep learning framework
- **BADAS-Open (Nexar)** – V-JEPA2-based collision prediction model
- **Pydantic** – Data validation and settings management
- **NumPy, Pandas, scikit-learn** – Data processing and evaluation metrics

### **Frontend / فرانت‌اند**
- **React 19** – Latest React with hooks and context API
- **TypeScript** – Type-safe JavaScript
- **Vite** – Next-generation build tool
- **Tailwind CSS** – Utility-first CSS framework
- **Recharts** – Composable charting library
- **Axios** – Promise-based HTTP client

---

## 📂 **Project Structure / ساختار پروژه**

```
DriveShield/
├── backend/
│   ├── api/                    # FastAPI application
│   │   ├── main.py            # API endpoints & service singleton
│   │   ├── config.py          # Configuration management
│   │   ├── schemas.py         # Pydantic response models
│   │   └── utils.py           # File upload utilities
│   │
│   ├── dataset/               # Dataset loading & preprocessing
│   │   ├── nexar_loader.py    # Nexar dataset loader
│   │   └── validate_loader.py # DataLoader validation
│   │
│   ├── models/                # Model loading & inference
│   │   ├── badas_loader.py    # BADAS-Open model loader
│   │   ├── badas_wrapper.py   # BadasOpenService wrapper
│   │   ├── badas_open.pth     # Model weights (not in Git)
│   │   └── config.json        # Model configuration
│   │
│   ├── evaluation/            # Evaluation & benchmarking
│   │   ├── eval_dataset.py   # Evaluation runner
│   │   ├── metrics.py         # Metric computation
│   │   ├── plots.py           # Visualization generation
│   │   └── run_eval_badas.py  # CLI evaluation entrypoint
│   │
│   └── training/              # Training scripts (optional)
│       ├── cnn_lstm.py        # Baseline model architecture
│       └── train_cnn_lstm.py  # Training pipeline
│
└── frontend/                  # React frontend application
    ├── src/
    │   ├── api/               # API client (Axios)
    │   ├── i18n/              # Internationalization (EN/FA)
    │   ├── components/        # React components
    │   ├── pages/             # Page components
    │   └── styles/            # Tailwind CSS
    └── package.json
```

---

## ⚙️ **Setup & Installation / راه‌اندازی و نصب**

### **1. Clone the Repository / کلون کردن مخزن**

```bash
git clone https://github.com/MahdiNavaei/DriveShield.git
cd DriveShield
```

### **2. Backend Setup / راه‌اندازی بک‌اند**

**EN:** Create a virtual environment and install dependencies.  
**FA:** یک محیط مجازی ایجاد کرده و وابستگی‌ها را نصب کنید.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install fastapi uvicorn pydantic pydantic-settings
pip install torch torchvision
pip install numpy pandas scikit-learn
pip install albumentations transformers huggingface-hub
```

**EN:** **Note:** You may need `ffmpeg` installed for video processing.  
**FA:** **نکته:** ممکن است برای پردازش ویدئو نیاز به نصب `ffmpeg` داشته باشید.

### **3. Download BADAS-Open Weights / دانلود وزن‌های BADAS-Open**

**EN:** Due to licensing and size (~4 GB), the BADAS-Open weights are not included in this repository.  
**FA:** به دلیل محدودیت‌های لایسنس و حجم (~4 گیگابایت)، وزن‌های BADAS-Open در این مخزن قرار ندارند.

1. Visit the [BADAS-Open model on Hugging Face](https://huggingface.co/nexar-ai/BADAS-Open)
2. Accept the terms if required
3. Download the following files:
   - `badas_loader.py`
   - `config.json`
   - `weights/badas_open.pth`
4. Place them under `backend/models/`:
   ```
   backend/models/
       badas_loader.py
       config.json
       badas_open.pth
   ```
5. Add `badas_open.pth` to your `.gitignore`

### **4. Download Nexar Dataset (Optional) / دانلود دیتاست Nexar (اختیاری)**

**EN:** Required only for evaluation. For API + UI usage, this is optional.  
**FA:** فقط برای ارزیابی لازم است. برای استفاده از API و UI، اختیاری است.

Download the [Nexar Dashcam Collision Prediction dataset](https://huggingface.co/datasets/nexar-ai/nexar-dashcam-collision-prediction) and place it under:

```
./nexar_collision_prediction/
    train/
    test-public/
    test-private/
```

### **5. Frontend Setup / راه‌اندازی فرانت‌اند**

```bash
cd frontend
npm install
```

---

## 🚀 **Running the System / اجرای سیستم**

### **1. Start the Backend / راه‌اندازی بک‌اند**

**EN:** From the project root directory:  
**FA:** از پوشه ریشه پروژه:

```bash
uvicorn backend.api.main:app --host 0.0.0.0 --port 8002 --reload
```

**EN:** Verify the API is running:  
**FA:** بررسی کنید که API در حال اجرا است:

```bash
curl http://localhost:8002/status
```

### **2. Start the Frontend / راه‌اندازی فرانت‌اند**

**EN:** From the `frontend/` directory:  
**FA:** از پوشه `frontend/`:

```bash
cd frontend
npm run dev
```

**EN:** Open the dashboard in your browser:  
**FA:** داشبورد را در مرورگر خود باز کنید:

```
http://localhost:5173
```

### **3. Using the UI / استفاده از رابط کاربری**

1. **EN:** Use the **EN / FA toggle** in the header to switch languages  
   **FA:** از **دکمه EN / FA** در هدر برای تغییر زبان استفاده کنید

2. **EN:** Check the **API status badge** to ensure the backend is connected  
   **FA:** **نشانگر وضعیت API** را بررسی کنید تا مطمئن شوید بک‌اند متصل است

3. **EN:** **Upload a dashcam video** (.mp4, .avi, etc.)  
   **FA:** **یک ویدئوی داش‌کم آپلود کنید** (.mp4, .avi و غیره)

4. **EN:** Click **"Run Prediction"** to start inference  
   **FA:** روی **"اجرای پیش‌بینی"** کلیک کنید تا اینفرانس شروع شود

5. **EN:** View results:  
   **FA:** مشاهده نتایج:
   - **Prediction Summary Card** – File name, max probability, high-risk frame count
   - **Risk Timeline Chart** – Interactive line chart with threshold overlay

---

## 📊 **Evaluation Pipeline / پایپ‌لاین ارزیابی**

**EN:** Run comprehensive evaluation on the Nexar dataset to benchmark model performance.  
**FA:** اجرای ارزیابی جامع روی دیتاست Nexar برای بنچمارک عملکرد مدل.

### **Running Evaluation / اجرای ارزیابی**

```bash
python -m backend.evaluation.run_eval_badas \
    --split test-public \
    --num_samples 50 \
    --output_dir eval_output
```

**EN:** This will generate:  
**FA:** این دستور تولید می‌کند:

- `eval_output/results.csv` – Per-sample predictions and ground truth
- `eval_output/metrics.json` – Computed metrics (AUC-ROC, AP, etc.)
- `eval_output/roc_curve.png` – ROC curve visualization
- `eval_output/pr_curve.png` – Precision-Recall curve visualization

### **Metrics Computed / متریک‌های محاسبه شده**

- **AUC-ROC** – Area under the receiver operating characteristic curve
- **Average Precision (AP)** – Precision-recall area under curve
- **Precision & Recall @ threshold** – Configurable threshold metrics (default: 0.8)
- **mTTA (approximate)** – Mean Time-To-Accident estimation based on high-risk frame analysis

**EN:** **Note:** If evaluating on a very small subset containing only one class, some metrics (e.g., AUC) may be NaN. This is expected behavior, not a bug.  
**FA:** **نکته:** اگر ارزیابی روی زیرمجموعه بسیار کوچکی که فقط یک کلاس دارد انجام شود، برخی متریک‌ها (مثل AUC) ممکن است NaN شوند. این رفتار طبیعی است، نه باگ.

---

## 🌐 **Bilingual User Experience / تجربه کاربری دوزبانه**

**EN:** The frontend is fully bilingual (English / Persian) with **client-side language switching** that requires no page reload. All UI elements, labels, buttons, and chart axes are translated in real-time.  
**FA:** فرانت‌اند کاملاً دوزبانه (انگلیسی / فارسی) است با **تغییر زبان در سمت کلاینت** که نیاز به رفرش صفحه ندارد. تمام عناصر UI، برچسب‌ها، دکمه‌ها و محورهای نمودار به صورت لحظه‌ای ترجمه می‌شوند.

**EN:** This design is particularly useful for:  
**FA:** این طراحی به ویژه برای موارد زیر مفید است:

- **Fleet management systems** where operators may use different languages
- **International deployments** requiring multi-language support
- **Research collaborations** across different regions

---

## 🔬 **Possible Extensions / توسعه‌های ممکن**

**EN:** Ideas for future enhancements and research directions:  
**FA:** ایده‌هایی برای بهبودهای آینده و جهت‌های تحقیقاتی:

- **Model Distillation** – Distil BADAS-Open into a lightweight student model (e.g., CNN+LSTM) for edge devices
- **Multi-Modal Fusion** – Integrate dashcam video with telemetry data (speed, acceleration, GPS) for richer risk modeling
- **Frame-Level Visualization** – Overlay risk heatmaps directly on video frames for enhanced interpretability
- **Cloud Integration** – Connect with cloud storage and fleet management systems for scalable deployments
- **Production Deployment** – Deploy on VPS with HTTPS, authentication, and monitoring
- **Real-Time Streaming** – Process live video streams for real-time collision risk alerts

---

## 👨‍💻 **About the Developer / درباره توسعه‌دهنده**

**Mahdi Navaei** – AI Engineer & Data Scientist

**EN:** Results-driven AI Engineer with **7+ years of experience** designing and deploying scalable, data-driven solutions. Specialized in advanced machine learning, deep learning, NLP, and production-grade intelligent systems.  
**FA:** مهندس هوش مصنوعی با **بیش از 7 سال تجربه** در طراحی و استقرار راه‌حل‌های مقیاس‌پذیر و مبتنی بر داده. متخصص در یادگیری ماشین پیشرفته، یادگیری عمیق، پردازش زبان طبیعی و سیستم‌های هوشمند production-grade.

### **Key Achievements / دستاوردهای کلیدی**

- 🏆 **2nd place** in Tehran Provincial AI Competition (2022)
- 🎓 **Membership** in Iran's National Elites Foundation
- 📊 **Kaggle Notebooks Master**
- 📝 **Published research** in peer-reviewed journals (Health Science Reports, ICVPR, AMLAI)

### **Professional Experience / تجربه حرفه‌ای**

**EN:** Currently working as a **Data Scientist at Daria Hamrah Paytakht**, building autonomous NL-to-SQL agents, large-scale hybrid recommender systems, LLM-powered call-center intelligence pipelines, and enterprise RAG knowledge engines.  
**FA:** در حال حاضر به عنوان **دیتا ساینتیست در داریا هم‌راه پرداخت** مشغول به کار است و در حال ساخت agentهای خودکار NL-to-SQL، سیستم‌های پیشنهاددهنده ترکیبی در مقیاس بزرگ، پایپ‌لاین‌های هوشمند مرکز تماس مبتنی بر LLM و موتورهای دانش RAG سازمانی است.

**EN:** Previously led a team of 7 data scientists at **Diar-e Kohan CO.**, implementing ML solutions that increased sales by 5%.  
**FA:** پیش از این تیمی متشکل از 7 دیتا ساینتیست را در **شرکت دیار کهن** رهبری کرده و راه‌حل‌های ML را پیاده‌سازی کرده که منجر به افزایش 5 درصدی فروش شد.

### **Connect / ارتباط**

- 📧 Email: mahdinavaei1367@gmail.com
- 💼 LinkedIn: [linkedin.com/in/mahdinavaei](https://linkedin.com/in/mahdinavaei)
- 💻 GitHub: [github.com/MahdiNavaei](https://github.com/MahdiNavaei)
- 📊 Kaggle: [kaggle.com/mahdinavaei](https://kaggle.com/mahdinavaei)

**EN:** 📱 **Share on LinkedIn:** Ready-to-use post templates available in [`LINKEDIN_POST.md`](LINKEDIN_POST.md)  
**FA:** 📱 **اشتراک در LinkedIn:** قالب‌های آماده پست در [`LINKEDIN_POST.md`](LINKEDIN_POST.md) موجود است

---

---

## 🙏 **Acknowledgements / قدردانی**

**EN:** This project stands on the shoulders of excellent public work by **Nexar**:  
**FA:** این پروژه بر پایه کارهای عمومی عالی **شرکت Nexar** بنا شده است:

- **[BADAS-Open](https://huggingface.co/nexar-ai/BADAS-Open)** – State-of-the-art collision prediction model
- **[Nexar Dashcam Collision Prediction Dataset](https://huggingface.co/datasets/nexar-ai/nexar-dashcam-collision-prediction)** – Comprehensive dashcam dataset

**EN:** Please refer to their original repositories and papers for detailed model and dataset descriptions.  
**FA:** لطفاً برای توضیحات دقیق مدل و دیتاست به مخازن و مقالات اصلی آنها مراجعه کنید.

---

## ⚖️ **License & Usage / لایسنس و استفاده**

**EN:** The DriveShield code (excluding BADAS-Open weights and third-party code) can be used under a standard open-source license (e.g., MIT/Apache-2.0).  
**FA:** کد DriveShield (به جز وزن‌های BADAS-Open و کدهای شخص ثالث) می‌تواند تحت یک لایسنس open-source استاندارد (مثل MIT/Apache-2.0) استفاده شود.

**EN:** **Important Notes:**  
**FA:** **نکات مهم:**

- BADAS-Open weights and loader code are subject to **Nexar's license** and Hugging Face terms
- You must **accept those terms** and download the weights yourself
- This repository **does not redistribute** the model weights
- This system is **for research and demonstration only** and must **not be used** in safety-critical applications

---

## 📈 **Project Statistics / آمار پروژه**

**EN:** This project demonstrates:  
**FA:** این پروژه نشان می‌دهد:

- ✅ **End-to-end ML pipeline** – From data loading to production deployment
- ✅ **Production-grade architecture** – Scalable backend with modern frontend
- ✅ **Best practices** – Type safety, error handling, bilingual UX
- ✅ **Comprehensive evaluation** – Industry-standard metrics and visualizations
- ✅ **Real-world applicability** – Designed for fleet management and research use cases

---

<div align="center">

**EN:** Made with ❤️ by [Mahdi Navaei](https://github.com/MahdiNavaei)  
**FA:** ساخته شده با ❤️ توسط [مهدی نوایی](https://github.com/MahdiNavaei)

⭐ **Star this repo if you find it useful!** ⭐

</div>

