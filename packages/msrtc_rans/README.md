# Introduction
`msrtc_rans` is a static library and python package implementing rANS entropy coding algorithm
(both rans_byte and rans64 variants).

# Installation
```shell
pip install <path-to>/packages/msrtc_rans
```

# Development
1. Create a separate venv for development
2. Install dependencies: `pip install scikit_build_core[pyproject,wheels] pytest`
3. Install editable package:
    ```shell
    cd <path-to>/packages
    pip install --no-build-isolation --editable msrtc_rans -C build-dir=build.d/msrtc_rans
    ```
4. Run tests: `pytest msrtc_rans`

Changes to python code will take into effect immediately. To update c++ module there are 3 variants:
1. Run cmake install:
   - build extension module in Visual Studio or with cmake: `cmake --build build.d/msrtc_rans --config Release`
   - run install: `cmake --install build.d/msrtc_rans --prefix <path-to-venv>/Lib/site-packages --config Release`
2. Use scikit-build build-in-import feature: `pip install ... -C editable.rebuild=true`
3. Modify `<path-to-venv>/Lib/site-packages/_msrtc_rans_editable.py`: set `'msrtc.rans._msrtc_rans'` to point
   to module build location

# Building AzureML (Ubuntu) packages
```shell
<path-to>/packages/linux-docker-build/build-linux-wheels.ps1
```
The script will put the built package in `<path-to>/packages/stage.d` folder.
