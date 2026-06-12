### Abstract Agent

This repository implements various base agents that provide functionality to 
implement MCTS agents which work with the https://github.com/Carcassonne-SGE/Carcassonne-Environment

#### Abstract Determinized Agent

Carcassonne is a stochastic game. Which tile comes next is random. There are various ways to handle this.
Many approaches suffer from blowing up the search space by introducing chance nodes. The Agents of this 
Repository derminize the game. For Caracssonne this means they fix one possible draw order and assume
that that order it the real one. Experiemnts have shown that this approach works best in Carcassonne 
compared to other tested ones. The Abstract Determinized Agent implements such behavior.


#### Abstract Determinized Ensemble Agent

This Agent uses the Abstract Determinized Agent to optimize agains multiple possible draw orders and 
pick the best action based on the combined information.

#### Core

the Core java package provides base functionality and helper functions for fast and non repetitive a
MCTS Agent implementation

