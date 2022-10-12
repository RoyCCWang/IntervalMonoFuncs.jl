# MonotonicFuntions.jl Documentation

```@contents
```

## Setup
NMRDataSetup.jl does not load NMR experiments from file. Currently, it only estimates the 0 ppm and solvent resonance components quickly.

In our examples, we use the [nmrglue](https://www.nmrglue.com/) Python package to read the free-induction decay (FID) data as well as the experiment settings information. We discuss how to do this in our examples. However, alternative software packages can be used to obtain these values.

## Usage examples
[Julia example usage guide](../jl_example.html)

[Julia Jupyter notebook file](../jl_example.ipynb)

[Python example usage guide](../py_example.html)

[Python Jupyter notebook file](../py_example.ipynb)

## Function Reference
```@docs
NMRDataSetup.loadspectrum
```

```@docs
NMRDataSetup.getwraparoundDFTfreqs
```

```@docs
NMRDataSetup.computeDTFT
```

```@docs
NMRDataSetup.getDFTfreqrange
```

```@docs
NMRDataSetup.gettimerange
```

```@docs
NMRDataSetup.evalcomplexLorentzian
```


## About
This package originated from the [AI-4-Design collaboration program](https://nrc.canada.ca/en/research-development/research-collaboration/programs/artificial-intelligence-design-challenge-program) between [Carleton University](https://carleton.ca/) and the [National Research Council of Canada](https://nrc.canada.ca/en).

## Citation
If you use NMRDataSetup.jl in your work, please cite this repository using the GitHub citation option on the main repository page.

## License
MIT License

Copyright (c) 2022 Roy Wang and contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.