# Deployment & Persistent Storage Guide: Medical RAG System

This guide explains how to deploy the Medical Agentic RAG application with **Persistent Storage** so that all uploaded research papers, Google Drive syncs, and FAISS/ChromaDB vector stores are permanently retained across container restarts, updates, and redeployments.

---

## 1. Quick Start: Local Docker Deployment

### Prerequisites
* [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running.
* Ensure you have your `.env` file configured with your `OPENAI_API_KEY`:
  ```bash
  OPENAI_API_KEY=your_openai_api_key_here
  ```

### Build & Run Container
From the project root directory, run:

```bash
# Build the Docker image and start the application in the background
docker compose up -d --build
```

### Accessing the Web App
Open your browser and navigate to:
```text
http://localhost:8501
```

### Stopping the Container
```bash
docker compose down
```

---

## 2. How Persistent Storage Works

In [`docker-compose.yml`](docker-compose.yml), the storage volume is configured as:

```yaml
volumes:
  - ./data:/app/data
```

### What this achieves:
| Host Directory | Container Path | Purpose |
| :--- | :--- | :--- |
| `./data/raw_pdfs/` | `/app/data/raw_pdfs/` | Stores all uploaded clinical PDFs and Google Drive downloads permanently. |
| `./data/processed/faiss_index/` | `/app/data/processed/faiss_index/` | Stores the FAISS vector database indices. |
| `./data/processed/chroma_db/` | `/app/data/processed/chroma_db/` | Stores ChromaDB collections. |

* **Zero Data Loss:** Even if you delete or rebuild the Docker container (`docker compose down`), all your research papers and indexed vector stores remain intact in your host `./data/` folder.
* **Live Pre-loading:** You can drop PDF files directly into `./data/raw_pdfs/` from your host OS at any time, then open the Streamlit web app and click **"🔄 Trigger Corpus Re-Indexing"**.

---

## 3. Cloud Deployment Options

### Option A: Railway.app / Render (Easiest Managed Cloud)
1. Push your repository to GitHub.
2. Sign up on [Railway](https://railway.app) or [Render](https://render.com).
3. Create a new service from your GitHub repo. It will automatically detect the [`Dockerfile`](Dockerfile).
4. **Attach a Persistent Volume:**
   * **Mount path:** `/app/data`
   * **Size:** 5 GB – 10 GB
5. **Set Environment Variables:**
   * `OPENAI_API_KEY`: Your OpenAI API key
6. Deploy! The service will expose a public HTTPS URL.

---

### Option B: Cloud Virtual Machine (AWS EC2 / DigitalOcean Droplet / Hetzner)
1. Launch an Ubuntu 22.04 / 24.04 instance (Recommended: 2 vCPU, 4GB+ RAM, e.g. `t3.medium` on AWS or $12/mo droplet on DigitalOcean).
2. Install Docker and Docker Compose:
   ```bash
   sudo apt update && sudo apt install -y docker.io docker-compose
   ```
3. Clone your repository:
   ```bash
   git clone <YOUR_GIT_REPO_URL>
   cd MEDICAL_RAG_KP
   ```
4. Create your `.env` file:
   ```bash
   nano .env
   # Add: OPENAI_API_KEY=your_key
   ```
5. Start the container:
   ```bash
   docker-compose up -d --build
   ```
6. (Optional) Put Nginx + Certbot in front for your custom domain and free SSL:
   ```nginx
   server {
       server_name yourdomain.com;
       location / {
           proxy_pass http://localhost:8501;
           proxy_http_version 1.1;
           proxy_set_header Upgrade $http_upgrade;
           proxy_set_header Connection "upgrade";
           proxy_set_header Host $host;
       }
   }
   ```

---

### Option C: Hugging Face Spaces (100% Free Forever - Recommended ⭐)

1. **Create Space on Hugging Face:**
   * Go to [huggingface.co/spaces](https://huggingface.co/spaces) and click **"Create new Space"**.
   * Give it a name (e.g., `medical-rag-assistant`).
   * Select License: `apache-2.0` or `mit`.
   * **Space SDK:** Choose **`Docker`** -> **`Blank`**.
   * Hardware: Keep **CPU basic (2 vCPU, 16 GB RAM)** — **Free**.
   * Visibility: Public or Private.
   * Click **"Create Space"**.

2. **Add Environment Secret:**
   * In your new Space, click on the **⚙️ Settings** tab.
   * Scroll down to **"Variables and secrets"**.
   * Under **"Secrets"**, click **"New secret"**:
     * Name: `OPENAI_API_KEY`
     * Value: `sk-proj-...` (your actual OpenAI API key)

3. **Push Your Code to Hugging Face:**
   In your local PowerShell terminal, add Hugging Face as a git remote and push:
   ```powershell
   # Add Hugging Face remote (replace USERNAME and SPACE_NAME with yours)
   git remote add hf https://huggingface.co/spaces/YOUR_HF_USERNAME/YOUR_SPACE_NAME

   # Stage and commit your files
   git add .
   git commit -m "Deploy Medical RAG to Hugging Face Spaces"

   # Push to Hugging Face
   git push hf main
   ```
   *(When prompted for credentials, use your Hugging Face username and your **User Access Token** with 'Write' permissions from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) as the password).*

4. **Watch Build & Launch:**
   * Hugging Face will automatically detect your [`Dockerfile`](Dockerfile), build the environment, and launch the Streamlit app on port `7860`.
   * Your app will be live at `https://YOUR_HF_USERNAME-YOUR_SPACE_NAME.hf.space`!

