package core;

import model.bits.CarcassonneActionLayoutBit;
import model.state.PerformActionManager;
import model.state.PossibleActionManager;
import model.state.State;
import sge.CarcassonneAction;

/// AbstractNode
///
/// base node for the MCTS Agents used to build the Tree
/// provides or enforces the methods for select, expand, simulate, and backpropagate
/// as well as some helpers
public abstract class AbstractNode<A extends AbstractMctsAgent<N>, N extends AbstractNode<A, N>> {
    // used to get the Hyperparameters
    protected final A agent;
    protected final N parent;

    // utility value of that node
    protected float value;
    protected int visits;
    protected int depth;

    protected State checkPoint;

    /// AbstractNode
    ///
    /// Constructor for the abstract node. Initializes the agent reference,
    /// parent node, and state checkpoint, and calculates the depth of the node.
    ///
    /// @param agent the agent executing the search
    /// @param parent the parent node in the MCTS tree, or null if root
    /// @param checkPoint the game state checkpoint at this node, or null
    protected AbstractNode(A agent, N parent, State checkPoint) {
        this.agent = agent;
        this.parent = parent;
        this.checkPoint = checkPoint;

        if(parent != null){
            depth = parent.depth+1;
        }
    }

    /// select
    ///
    /// Main part of the MCTS loop select searches in the tree for the node that should be
    /// expanded next. May use UCB but depends on the implementation
    public abstract N select();

    /// expand
    ///
    /// Part of MCTS loop responsible to generate new child for that node
    /// and prepare for rollout.
    ///
    ///  @param state state of the current node already recreated should not be mutated
    /// used to get informs for expanding giving it context
    ///  @apiNote for performance reasons state as parameter
    public abstract N expand(State state);

    /// simulate
    ///
    /// part of MCTS loop produces a rollouts so does the MCTS simulations
    /// @param state state of the current node already recreated should not be mutated
    /// @apiNote for performance reasons state as parameter
    public abstract float simulate(State state);

    /// backpropagate
    ///
    /// again part of MCTS loop propagates the values of the utility up the
    /// tree to parent if there is a parent also update visits
    public final void backpropagate(float rolloutValue) {
        visits += 1;
        value += rolloutValue;
        if (parent != null) {
            parent.backpropagate(rolloutValue);
        }
    }



    /// recreateState
    ///
    /// recreates a state form a initial state recursively  uses the applyLocalTransition
    /// @apiNote Recursivly applies action and uses checkpoints
    public final State recreateState() {
        if (checkPoint != null){
            // recursion base case
            return checkPoint.deepCopy();
        }else {
            // need to recreate State
            State parentState = parent.recreateState();
            applyLocalTransition(parentState);
            // create checkpoint if necessary
            if (agent.config.checkPointInterval() > 0 && depth % agent.config.checkPointInterval() == 0) {
                checkPoint = parentState.deepCopy();
            }
            return parentState;
        }
    }


    /// applyLocalTransition
    ///
    /// applies the action/card pop of the current node
    protected abstract void applyLocalTransition(State state);

    /// getChildren
    ///
    /// Returns the array of child nodes generated from this node.
    ///
    /// @return an array containing the children nodes, or null/empty if none
    public abstract N[] getChildren();

    /// self
    ///
    /// Utility method to cast the node instance to the generic node type N.
    ///
    /// @return the node cast to type N
    @SuppressWarnings("unchecked")
    protected final N self() {
        return (N) this;
    }


    /// getSelectionScore
    ///
    /// calculates the ucb using the transform of the agent can be used by the nodes
    public  float getSelectionScore() {
        if (visits == 0) {
            return Float.MAX_VALUE;
        }

        float q = agent.ucbTransform(value / (float) visits);
        if (parent == null || parent.visits <= 1) {
            return q;
        }
        return q + agent.config.c() * (float) Math.sqrt(Math.log(parent.visits) / visits);
    }

    /// rollout
    ///
    /// given a state this function performs one random playout and returns the utility of that
    /// outcome
    ///
    /// @apiNote changes state to give it only a deepcopy
    protected final float rollout(State state) {
        while (!state.isGameOver()) {
            // check if need a new tile
            if (state.getCurrentTile() == null) {
                int tileId = PerformActionManager.determineNextDrawAction(state, agent.rand);
                if (tileId >= 0) {
                    PerformActionManager.performDrawAction(state, tileId);
                } else {
                    PerformActionManager.performRandomDrawAction(state, agent.rand);
                }
            }else{
                int action = PossibleActionManager.getRandomAction(state, agent.rand, agent.config.meepleProb());
                PerformActionManager.performAction(state, new CarcassonneAction(action));
            }
        }
        return agent.utility(state);
    }

    /// averageRollout
    ///
    /// performs k rollouts(k is in the agent) and takes the average from there
    /// @param state reconstructed or start state is not mutated does only work on copies
    protected final float averageRollout(State state) {
        float utilitySum = 0;
        int completedRollouts = 0;
        for (int i = 0; i < agent.config.rollouts(); i++) {
            State rolloutState = (i < agent.config.rollouts() - 1) ? state.deepCopy() : state;
            utilitySum += rollout(rolloutState);
            completedRollouts++;
        }
        if (completedRollouts == 0) {
            return agent.utility(state);
        }
        return utilitySum / completedRollouts;
    }

    /// shuffleChildren
    ///
    /// helper function to shuffle an Array randomly given the rand object of the agent
    /// @apiNote the children are managed in the individual nodes them self this method
    /// helps for UCT to break the ties randomly in a simple manner
    protected final <T extends N> void shuffleChildren(T[] children) {
        if (children != null && children.length >= 2) {
            for (int i = children.length - 1; i > 0; i--) {
                int j = agent.rand.nextInt(i + 1);
                T temp = children[i];
                children[i] = children[j];
                children[j] = temp;
            }
        }
    }


    /// selectUnvisitedOrBestUsb
    ///
    /// takes a children array iterates over it to find a child with zero visits. Returns such a child
    /// otherwise returns the one with highs Ucb value
    ///
    /// @param children array where the children are located in may be filled with null at the end
    /// @param count how many elements are actually in the array
    /// @apiNote uses selectBestChildByUcb internally
    protected final <T extends N> T selectUnvisitedOrBestUsb(T[] children, int count) {
        shuffleChildren(children);
        for (int i = 0; i < count; i++) {
            T child = children[i];
            if (child != null && child.visits == 0) {
                return child;
            }
        }
        return selectBestChildByUcbNoShuffle(children, count);
    }

    /// selectBestChildByUcb
    /// iterates of the array and returns the element with highest ucb
    /// @param children array where the children are located in may be filled with null at the end
    /// @param count how many elements are actually in the array
    /// @apiNote breaks ties
    protected final <T extends N> T selectBestChildByUcb(T[] children, int count) {
        shuffleChildren(children);
        return selectBestChildByUcbNoShuffle(children, count);
    }

    /// selectBestChildByUcbNoShuffle
    ///
    /// Selects and returns the child node with the highest UCB score
    /// without shuffling the array first.
    ///
    /// @param children the array containing child nodes
    /// @param count the number of elements in the array
    /// @return the child node with the highest UCB score, or null if none found
    private <T extends N> T selectBestChildByUcbNoShuffle(T[] children, int count) {
        T best = null;
        float bestScore = Float.NEGATIVE_INFINITY;
        for (int i = 0; i < count; i++) {
            T child = children[i];
            if (child != null) {
                float score = child.getSelectionScore();
                if (score > bestScore) {
                    bestScore = score;
                    best = child;
                }
            }
        }
        return best;
    }

    /// bestVisitedChild
    ///
    /// returns the child with highest average value, falls back to first child if none visited
    protected final <T extends N> T bestVisitedChild(T[] children) {
        T fallback = null;
        if (children != null && children.length > 0) {
            fallback = children[0];
        }

        T best = null;
        float bestScore = Float.NEGATIVE_INFINITY;
        if (children != null) {
            for (T child : children) {
                if (child != null && child.visits > 0) {
                    float score = child.value / child.visits;
                    if (score > bestScore) {
                        bestScore = score;
                        best = child;
                    }
                }
            }
        }
        return best != null ? best : fallback;
    }

    // getters

    /// getVisits
    ///
    /// Returns the number of times this node has been visited during search.
    ///
    /// @return the visit count
    public final int getVisits() {
        return visits;
    }

    /// getValue
    ///
    /// Returns the accumulated simulation value of this node.
    ///
    /// @return the accumulated value
    public final float getValue() {
        return value;
    }

    /// getDepth
    ///
    /// Returns the depth of the node in the search tree.
    ///
    /// @return the depth of the node
    public int getDepth() {
        return depth;
    }
}
