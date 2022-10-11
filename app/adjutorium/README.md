## Adjutorium Breast cancer app

This repository contains the logic behind the [Adjutorium breast-cancer paper](https://www.vanderschaar-lab.com/million-patient-study-shows-strength-of-machine-learning-in-recommending-breast-cancer-therapies/).

### Installation

```
pip install -r requirements.txt
```

### Usage

```
PORT=8080 python ./app.py
```

or 

```
gunicorn app:server --timeout 300
```

### Heroku

Deployment: https://adjutorium-breastcancer.herokuapp.com/

For admin access, please contact bcc38@cam.ac.uk or nm736@cam.ac.uk.
