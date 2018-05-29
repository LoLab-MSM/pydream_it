# pydream_it
------
![Python version badge](https://img.shields.io/badge/python-2.7-blue.svg)
[![license](https://img.shields.io/github/license/mashape/apistatus.svg)](LICENSE)
------
####  pydream_it is script to that reads/imports [PySB](http://pysb.org/) models and generates a template script that is used for [PyDREAM](https://github.com/LoLab-VU/PyDREAM) calibration of the model. pydream_it inspects the model and identifies the kinetic parameters and generates the SampledParam code lines. pydream_it currently has limited functionality for #PYDREAM_IT directives that can be added to the model code alter pydream_it behavior.

## Install and basic usage

**Warning:** pydream_it is still under heavy development and may rapidly change.

#### pydream_it run dependencies
pydream_it does not have any special dependencies itself, but [PySB](http://pysb.org/) is of course needed to run the PySB models:

pydream_it does not require any special installation to use, just clone or download the GitHub repo
```
git clone https://github.com/blakeaw/pydream_it.git
```
Then to use pydream_it, simply call the script with python and passing as input the model file
```
python pydream_it.py model.py
```
The file model.py should be a PySB model file.
*NOTE:* pydream_it currently doesn't inspect for the location of model.py, so you need to be in the same directory as the model.py file when you call pydream_it.
In this case pydream_it will generate a file `run_pydream_model.py` which will contain some basic code to calibrate the model using PyDREAM.

###pydream_it directives
pydream_it currently has the following directives that can be added to the model code file:
    * #PYDREAM_IT prior [param_name, param_index] [norm, uniform] - tell pydream_it to use a specific prior (other than the default) for the specified parameter. The parameter can be specified by either it name (param_name) or its index (param_index). pydream_it currently supports using normal distribution (norm) of uniform priors.
    * #PYDREAM_IT no-sample [param_name, param_index] - define a fixed parameter, i.e., pydream_it will not add it to the sample parameters list for the PyDREAM run.
The #PYDREAM_IT directives will appear as comments in the model code and are parsed by pydream_it at runtime.
------

## Contact

To report problems or bugs please open a
[GitHub Issue](https://github.com/blakeaw/pydream_it/issues). Additionally, any
comments, suggestions, or feature requests for pydream_it can also be submitted as
a
[GitHub Issue](https://github.com/blakeaw/pydream_it/issues).

For any other inquiries, including questions about pydream_it use or
implementation, you can contact Blake directly via e-mail at either
blake.a.wilson@vanderbilt.edu or blakeaw1102@gmail.com; please include "pydream_it
inquiry" in the e-mail subject line.

------

## Contributing

If you would like to contribute directly to pydream_it's development please
 1. Fork the repo (https://github.com/blakeaw/pydream_it/fork)
 2. Create a new branch for your feature (git checkout -b feature/foo_bar)
 3. Create test code for your feature
 4. Once your feature passes its own test, run all the tests using [pytest](https://docs.pytest.org/en/latest/) (python -m pytest)
 5. Once your feature passes all the tests, commit your changes (git commit -am 'Add the foo_bar feature.')
 6. Push to the branch (git push origin feature/foo_bar)
 7. Create a new Pull Request

------

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

------

## Acknowledgments

* This code and the template run scripts it generates are based on the [examples](https://github.com/LoLab-VU/PyDREAM/tree/master/pydream/examples) given in the [PyDREAM](https://github.com/LoLab-VU/PyDREAM) repo.

------
