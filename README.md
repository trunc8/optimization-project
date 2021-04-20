# Optimization Project
#### Team Cars
Course project for AE755 [Optimization in Engineering]

### Usage
```
git clone https://github.com/trunc8/optimization-project.git
cd optimization-project
pip3 install -r requirements.txt
```

To check help menu and find list of algorithms
```
python3 code/suspension_optimization.py -h
```

To run the script against, say, Simulated Annealing
```
python3 code/suspension_optimization.py -a SA
```

To view intermediate design variable values, set the verbose flag (note that this will hide the progress bar)
```
python3 code/suspension_optimization.py -a SA -v
```

The results are automatically written to csv file with the corresponding algorithm name in the `results` directory.


Finally, in order to compare performance of all algorithms against the [test objective functions](code/test_objectives.py), execute
```
python3 code/testing.py
```
