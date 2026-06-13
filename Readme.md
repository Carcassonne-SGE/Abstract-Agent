### Abstract Agent

This repository implements various base agents that provide functionality to 
implement MCTS agents which work with the https://github.com/Carcassonne-SGE/Carcassonne-Environment

#### Abstract Determinized Agent

Carcassonne is a stochastic game. Which  means that the order in which the tiles appear are random. There are various ways to handle this stochastic nature of which a few were tested. 
The results have shown that derterminizing the tile order works best in Carcassonne. This means
that one Tile order is fixed and apart from that normal MCTS search is performed.The Abstract Determinized Agent implements such behavior.


#### Abstract Determinized Ensemble Agent

This Agent uses the Abstract Determinized Agent to optimize against multiple possible draw orders. Each order produces a seperate search tree. Those trees are then combined to pick the action that worked best with the combined information

#### Core

the Core java package provides some helper functions and an absolut base class for the agents. This was used to implement the other approaches aprat from determinizing the order.
