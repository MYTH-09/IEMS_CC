
git clone https://github.com/MYTH-09/IEMS_CC.git

-- Create a virtual env --
```
python3 -m venv venv
```
```
source venv/bin/activate
```

-- Installing modules --
```
pip install -r requirements.txt
pip install pymoo
```

-- Solar Data Pipeline (NASA POWER) --
```
# List all 50+ supported Indian cities (Bengaluru, Chennai, Hyderabad, Mysuru, Mumbai, Delhi, etc.):
python scripts/02_solar_pipeline.py --list

# Fetch and simulate solar data for specific cities:
python scripts/02_solar_pipeline.py --city Bengaluru
python scripts/02_solar_pipeline.py --cities Bengaluru Chennai Hyderabad Mysuru Mumbai Delhi

# Fetch for all Indian cities:
python scripts/02_solar_pipeline.py --all
```

-- starting backend -- http://127.0.0.1:8000
```
uvicorn app:app --reload --port 8000
# or on Windows:
.\venv\Scripts\python.exe -m uvicorn app:app --reload --port 8000
```

-- starting frontend -- 
```
Open "pipeline_demo.html" or "iems_contractor_ui.html" via Live Server / browser:
python -m http.server 5500
```
http://localhost:5500/pipeline_demo.html

All major Indian cities (Bengaluru, Chennai, Hyderabad, Mysuru, Delhi, Mumbai, Kolkata, etc.) are available in the City dropdown!


