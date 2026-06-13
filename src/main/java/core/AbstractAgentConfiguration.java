package core;

import java.util.Random;

///
/// @param c exploration hyperparameter  for UCB
/// @param rollouts number of rollouts per simulation for leaf evaluation
/// @param meepleProb  with what probability may a meeple be placed for random action
/// @param checkPointInterval in which interval are state saved in the tree
public record AbstractAgentConfiguration (float c, int rollouts, float meepleProb, int checkPointInterval) {
}