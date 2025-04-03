# XrayGPT-Chest-Radiographs-Summariser

### Value Proposition

XrayGPT-CloudMed is an AI-powered chest radiograph analysis system designed to assist radiologists by automatically summarizing X-ray findings and providing interactive diagnostic assistance. The current workflow in radiology involves manual interpretation and report writing, which is **time-consuming, prone to variability, and susceptible to human error**. Our system uses **XrayGPT**, a vision-language model fine-tuned for medical imaging, to generate structured summaries and answer queries in real-time, improving diagnostic consistency and reducing workload.

Misclassifying sensitive data is an ethics concern and there is a possibilty that our system will make mistakes. To mitigate this risk and inform the radiologists, we will attach a confidence score along with the output to give an idea about the quality of the prediction and use their discretion to make an informed decision. 

**Business Metric:** Reduction in **radiologist review time** and improved **diagnostic accuracy**.

---
### Contributors

| Name                            | Responsible for | Link to their commits in this repo |
|---------------------------------|-----------------|------------------------------------|
| All team members                | System Design, DevOps, CI/CD & Monitoring, Deployment | TBD |
| Monish Raman Vishakraman                   | Model Development & Training | TBD |
| Shreyas Bhaktharam                   | Data Pipeline & Experiment Monitoring | TBD |
| Rohan Dhengale                   | Infrastructure & Cloud Deployment | TBD |

---
### System Diagram



---
### Summary of Outside Materials (add links to the dataset)

| Name              | How it was created | Conditions of use |
|------------------|--------------------|-------------------|
| MIMIC-CXR Dataset  | Collected from real-world radiology reports, de-identified | Public, research use (PhysioNet license) |
| OpenI Dataset     | Chest X-rays from Indiana University hospital network | Public, research use (NIH license) |
| MedClip (Vision Encoder) | Pretrained model for medical images | Open-source, research use |
| Vicuna (LLM)      | Fine-tuned LLaMA model for dialogue tasks | Open-source, non-commercial use |

---
### Summary of Infrastructure Requirements

| Requirement     | How many/when                                     | Justification |
|-----------------|---------------------------------------------------|---------------|
| `m1.large` VMs | 3 for entire project duration                     | Manage training, and monitoring services |
| `gpu_mi100`     | 4-hour block twice a week                         | Distributed model training |
| Floating IPs    | 1 for entire project duration, 1 for sporadic use | API hosting and testing |
| Persistent Storage | 500GB for project duration                      | Store datasets, models, and logs |
| `gpu_a10` | 2 for entire project duration | Vision-language models require GPU acceleration for real-time clinical use

---
### Detailed Design Plan

#### Model Training and Training Platforms

- **Components:** Uses **MedClip for vision encoding**, **Vicuna for text generation**, and a **linear transformation layer** for modality alignment.
- **Justification:** XrayGPT requires domain-specific adaptation; fine-tuning on **217k radiology reports** enhances its performance.
- **Lecture Relevance:** Implements **DDP/FSDP** for large-scale training and **Ray Tune** for hyperparameter tuning.
- **Difficulty Points:** Implements **multi-GPU scaling** and **automated re-training**.

#### Model Serving and Monitoring Platforms

- **Strategy:** Deploy model as a **FastAPI service** with **ONNX optimizations** for low-latency inference.
- **Components:** Uses **Kubernetes for scaling**, **Grafana for monitoring**, and **Prometheus for logging**.
- **Justification:** Ensures **high availability** and **low response times** for clinical use.
- **Lecture Relevance:** Implements **quantization, tensor optimizations**, and **multi-model inference handling**.

#### Data Pipeline

- **Strategy:** **ETL pipeline** processes **raw radiology reports** into structured summaries.
- **Components:** Uses **Kafka for real-time ingestion**, **Apache Spark for processing**, and **PostgreSQL for storage**.
- **Justification:** Ensures high-quality data for model retraining and evaluation.
- **Lecture Relevance:** Implements **data validation, storage management, and streaming analytics**.
- **Difficulty Points:** Includes **real-time data simulation** and **interactive data dashboards**.

#### Continuous X (CI/CD, Training, Deployment)

- **Strategy:** **GitOps-driven CI/CD pipeline** for **automated retraining, testing, and deployment**.
- **Components:** Uses **GitHub Actions, Argo Workflows, Terraform, and Helm**.
- **Justification:** Ensures **reproducible, automated infrastructure**.
- **Lecture Relevance:** Implements **immutable infrastructure principles and microservices deployment**.

---
### Next Steps
1. Set up **Chameleon Cloud infrastructure** (VMs, GPUs, Storage).
2. Implement **data preprocessing pipeline** and **ETL workflow**.
3. Train **XrayGPT models** using distributed training strategies.
4. Develop and deploy **inference API** with real-time monitoring.
5. Implement **CI/CD, staged deployment, and monitoring dashboards**.

**Final Goal:** A fully automated, scalable, and cloud-native chest X-ray analysis system for radiologists and healthcare providers.
