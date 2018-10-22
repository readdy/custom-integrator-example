Clone this repo
```bash
git clone https://github.com/readdy/custom-integrator-example.git myintegrator
cd myintegrator
git submodule init
git submodule update
```

Be in a conda environment `test` that has readdy installed and build the custom integrator
```bash
(test) $ python setup.py develop
```

Test it
```bash
(test) $ python run.py
```
