# Teaching a computer how to play Tic-tac-toe - A Trip Report



## From classic algorithms to Reinforcement learning with Neural Networks

In this series of articles and [Jupyter](https://jupyter.org/) notebooks we will explore a number of different approaches, from the Min Max algorithm to Neural Networks, with the aim of teaching / training a computer how to play the well known board game [Tic-tac-toe](https://en.wikipedia.org/wiki/Tic-tac-toe).

This is not a set of tutorials by an expert in the field who is sharing his knowledge with the world. This is more of a trip report by someone who is trying to learn some new skills, namely Reinforcement Learning and Neural Networks.

Not everything we will try in this series will work very well; and often the reasons why something worked or didn’t work ultimately will remain unclear. I still hope the articles are informative, may provide some insights, and in the end will help you with getting a deeper understanding of Reinforcement Learning and Neural Networks — how to use them as well as the challenges you may encounter when trying to do so.

***

Most people will be familiar with [Tic-tac-toe](https://en.wikipedia.org/wiki/Tic-tac-toe) and more than likely you will have played it at some stage of your life.
The game is rather simple. With respect to its rules as well as its strategy. In fact, it can be played by young children and Tic-tac-toe boards can often be found at playgrounds:

![Title](./Images/tic-tac-toe-355090_640.jpg)

[Source](https://pixabay.com/en/tic-tac-toe-game-tick-tack-toe-355090)


If you have played Tic-tac-toe a couple of times you will have quickly realised that it is quite easy to master. You will most likely also have discovered that when both players play good moves, the game will always end in a draw.

In the following, we will use the example of Tic-tac-toe to look at various approaches which can be used to teach or train a computer to play this game. Not because Tic-tac-toe is particularly challenging, but because it gives us a consistent, easy to understand target.

We will look at the classic Min Max algorithm, a Tabular Reinforcement Learning approach as well as a couple of Neural Network based Reinforcement Learning approaches:
    
* <a href="https://github.com/fcarsten/tic-tac-toe-pytorch/blob/master/Part%201%20-%20Computer%20Tic%20Tac%20Toe%20Basics.ipynb" target="_blank" rel="noopener">Part 1 - Computer Tic-tac-toe Basics</a>
* [Part 1 - Computer Tic-tac-toe Basics](https://github.com/fcarsten/tic-tac-toe-pytorch/blob/master/Part%201%20-%20Computer%20Tic%20Tac%20Toe%20Basics.ipynb){:target="_blank" rel="noopener"}
* [Part 2 - The Min Max Algorithm](./Part%202%20-%20The%20Min%20Max%20Algorithm.ipynb) 
* [Part 3 - Tabular Q-Learning](./Part%203%20-%20Tabular%20Q-Learning.ipynb)
* [Part 4 - Neural Network Q-Learning](./Part%204%20-%20Neural%20Network%20Q-Learning.ipynb)
* [Part 5 - Q Network review and becoming less greedy](./Part%205%20-%20Q%20Network%20review%20and%20becoming%20less%20greedy.ipynb)
* [Part 6 - Double Duelling Q Network with Experience Replay](./Part%206%20-%20Double%20Duelling%20Q%20Network%20with%20Experience%20Replay.ipynb)
* [Part 7 - This is deep. In a convoluted way](./Part%207%20-%20This%20is%20deep.%20In%20a%20convoluted%20way.ipynb)
* [Part 8 - Tic-tac-toe with Policy Gradient Descent](./Part%208%20-%20Tic%20Tac%20Toe%20with%20Policy%20Gradient%C2%A0Descent.ipynb)

The source code and Jupyter notebooks for this series are available at [GitHub](https://github.com/fcarsten/tic-tac-toe-pytorch).

You can also run the notebooks online via [Binder](https://mybinder.org/v2/gh/fcarsten/tic-tac-toe-pytorch/master)
