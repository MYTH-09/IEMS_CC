
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

-- starting backend -- 
```
uvicorn app:app --reload --port 8000
```

-- starting frontend -- 

```
I think you guys will have live server extension. if not install it. 
open the "pipeline_demo.html" 
click on "Go live" 
```


-- Note -- 
### If you are looking at the files don't me overwhelmed there are sooo many unnecessary files. specially files related to MPC.
