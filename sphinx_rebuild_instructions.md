1. In folder `morebs`:
- do `sphinx-quickstart`

2. Add to `conf.py` file after line 15:
```
import sys,os
sys.path.insert(0, os.path.abspath('.'))
extensions = ['sphinx.ext.todo', 'sphinx.ext.viewcode', 'sphinx.ext.autodoc']
```

3. Do
`sphinx-apidoc --ext-autodoc -o . ./morebs2`

4. Do `make clean` and `make html`.