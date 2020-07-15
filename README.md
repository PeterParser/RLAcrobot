Hello everyone, this my project for the Autonomous and Adaptive systems class of the University of Bologna.

# How to install:

Install the libraries
```commandline
pip3 install tensorflow --user 
pip3 install gym --user
```
N.B be sure that tensorflow version is at least 2.0

Download the repository:
```commandline
git clone https://github.com/PeterParser/RLAcrobot.git
```
or download the zip from github and extract  it.


# How to start it

If you want to train a simple DQN model just use "--train"
```commandline
python main.py --train
```

to use Prioritized experience replay:
```commandline
python main.py --train --per
```

to use actor critic instead of dqn:
```commandline
python main.py --train --ac
```

If you want to test a model:

```commandline
python main.py --test --model model_file
```


