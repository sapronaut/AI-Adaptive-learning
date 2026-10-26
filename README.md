
adaptive learning & concept gap analyzer

an intelligent, research-inspired adaptive learning system designed to track student mastery in programming concepts using Bayesian knowledge tracing (bkt). 

unlike traditional platforms that only check whether an answer is right or wrong, this system evaluates response times, difficulty levels, and prerequisite dependency graphs to identify why a student is struggling—tracing gaps back to foundational concept deficiencies rather than surface-level errors.

---

## key features

* Bayesian knowledge tracing (bkt): updates concept mastery probabilities in real time after every submission, factoring in slip, guess, and learning rates.
* prerequisite root-cause diagnostics: maps relationships between concepts (e.g., `recursion` requiring `functions` and `recursion_base_case`). if a student fails advanced exercises, the system evaluates upstream nodes to isolate root causes.
* classroom heatmap & matrix: provides an intuitive bird's-eye view of class-wide performance, highlighting at-risk students and overall weak concepts.
* adaptive recommendations: dynamically selects targeted practice materials based on the learner's current mastery state.
* decoupled architecture: built with a lightweight FastAPI backend and a static, responsive frontend dashboard.

---

## system architecture

the project is split into two independent layers:

1. backend (FastAPI): exposes rest api endpoints for processing student submission logs, computing bkt state updates, running graph diagnostics, and generating diagnostic reports.
2. frontend (dashboard): a minimalist interface that visualizes class heatmaps, student reports, and mastery distribution cards.

---

## project structure

```text
.
├── api.py                  # FastAPI server endpoints and cors middleware
├── bkt_model.py            # core Bayesian knowledge tracing algorithm & state tracking
├── prerequisite_graph.py   # concept dag & root-cause traversal logic
├── adaptive_recommender.py # dynamic practice recommendation generator
├── explainability.py       # human-readable gap explanations for teachers/students
├── config.py               # model hyperparameters, thresholds, and concept graph definitions
├── status.py               # classification utilities (at_risk, developing, mastered)
├── dashboard.html          # static ui dashboard (index.html)
├── requirements.txt        # Python dependencies
└── Procfile                # production startup command for deployment

```

---

## local setup & development

### 1. prerequisites

make sure you have Python 3.10+ installed on your system.

### 2. clone the repository

```bash
git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
cd your-repo-name

```

### 3. create a virtual environment & install dependencies

```bash
# create virtual environment
python -m venv venv

# activate on Windows (PowerShell)
.\venv\Scripts\Activate.ps1

# activate on Linux/macOS
source venv/bin/activate

# install requirements
pip install -r requirements.txt

```

### 4. run the backend server

start the FastAPI server using Uvicorn:

```bash
uvicorn api:app --reload --port 8000

```

the api will be available at `http://127.0.0.1:8000`. you can inspect the interactive OpenAPI documentation at `http://127.0.0.1:8000/docs`.

### 5. access the dashboard

open `dashboard.html` directly in your web browser. if you are running locally, ensure the `API_BASE_URL` at the top of the `<script>` tag in `dashboard.html` points to your local server:

```javascript
const API_BASE_URL = "[http://127.0.0.1:8000](http://127.0.0.1:8000)";

```

---

## deployment configuration

this project is configured for a split deployment strategy:

### backend (Render)

1. create a web service on Render connected to your repository.
2. set the environment to Python.
3. set the start command to:
```bash
uvicorn api:app --host 0.0.0.0 --port $PORT

```


4. push your changes. Render will build and host the live api.

### frontend (Vercel)

1. rename `dashboard.html` to `index.html`.
2. import the repository into Vercel.
3. set the framework preset to other (or static).
4. in `index.html`, ensure `API_BASE_URL` points to your production backend url:
```javascript
const API_BASE_URL = "[https://your-render-backend-url.onrender.com](https://your-render-backend-url.onrender.com)";

```


5. deploy. Vercel will host the dashboard statically on global cdns.

> note on Render free tier cold starts:
> free-tier instances on Render automatically go to sleep after periods of inactivity. the initial request (e.g., clicking load demo class) may take 30–40 seconds while the backend wakes up. subsequent requests will be instant.

---

## core api endpoints

| method | endpoint | description |
| --- | --- | --- |
| `GET` | `/` | system status, current concepts, and bkt threshold parameters |
| `POST` | `/responses` | submit a single student attempt to update bkt mastery |
| `GET` | `/class/matrix` | retrieve the class-wide mastery matrix across all concepts |
| `GET` | `/class/summary` | get aggregated class statistics and identify the weakest concept |
| `GET` | `/students/{id}/report` | detailed diagnostic report, root causes, and recommendations for a student |
| `POST` | `/demo/seed` | populate in-memory store with mock student performance data |

---

## license

MIT license. open for modification and research use.

```

```
