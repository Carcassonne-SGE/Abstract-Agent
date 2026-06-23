package abstractDeterminized;

import core.AbstractNode;
import model.bits.CarcassonneActionLayoutBit;
import model.collections.ActionSet;
import model.state.PerformActionManager;
import model.state.State;

/// DeterminizedActionNode
///
/// Determinized MCTS node that eagerly expands all legal children once the node is visited.
/// It stores checkpoints for the current node and each created child so the new checkpoint-based
/// AbstractNode API has usable state objects for selection.
public class DeterminizedActionNode<N extends DeterminizedActionNode<N>> extends AbstractNode<AbstractDeterminizedAgent<N>, N> {
    protected final int action;
    protected Integer drawnTile;
    protected N[] children;

    /// DeterminizedActionNode
    ///
    /// Constructor for the determinized MCTS node. Initializes the node with the
    /// executing agent, parent node, action that led to this node, and checkpoint state.
    ///
    /// @param agent the determinized MCTS agent
    /// @param parent the parent node in the MCTS tree
    /// @param action the action that led to this node
    /// @param checkpoint the game state checkpoint at this node
    public DeterminizedActionNode(AbstractDeterminizedAgent<N> agent, N parent, int action, State checkpoint) {
        super(agent, parent, checkpoint);
        this.action = action;
    }

    /// newChildrenArray
    ///
    /// Helper method to initialize a new children array of the specified size.
    ///
    /// @param size the size of the array
    /// @return a new array of child nodes
    @SuppressWarnings("unchecked")
    protected N[] newChildrenArray(int size) {
        return (N[]) new DeterminizedActionNode<?>[size];
    }

    /// select
    ///
    /// Selects a child node using UCB or returns the current node if it has no children.
    ///
    /// @return the selected node
    @Override
    public N select() {
        if (children == null || children.length == 0) {
            return self();
        }

        N best = selectUnvisitedOrBestUsb(children, children.length);
        assert best != null;
        return best.getVisits() == 0 ? best : best.select();
    }

    /// expand
    ///
    /// Expands the node by generating all possible unique children actions from the current state.
    ///
    /// @param state the current state of the game
    /// @return the current node after expansion
    @Override
    public N expand(State state) {
        if (state.isGameOver() && children != null) {
            return self();
        }

        ActionSet actions = (ActionSet) state.calculatePossibleActionsUnique();
        if (actions.isEmpty()) {
            children = newChildrenArray(0);
            return self();
        }
        children = newChildrenArray(actions.size());
        for (int i = 0; i < actions.size(); i++) {
            int childAction = actions.get(i);
            N child = agent.childFactory(self(), childAction, null);
            children[i] = child;
        }
        shuffleChildren(children);
        return self();
    }

    /// simulate
    ///
    /// Performs rollouts from the current state and returns the average utility value.
    ///
    /// @param state the current state of the game
    /// @return the average utility value from the simulations
    @Override
    public float simulate(State state) {
        return averageRollout(state);
    }

    /// applyLocalTransition
    ///
    /// Applies the transition (action and/or draw action) associated with this node to the state.
    ///
    /// @param state the state to apply the transition to
    @Override
    protected void applyLocalTransition(State state) {
        if (action != 0) {
            PerformActionManager.performAction(
                    state,
                    CarcassonneActionLayoutBit.getX(action),
                    CarcassonneActionLayoutBit.getY(action),
                    CarcassonneActionLayoutBit.getRotation(action),
                    CarcassonneActionLayoutBit.getAreaId(action)
            );
            if (!state.isGameOver() && state.getCurrentTile() == null) {
                replayResolvedDraw(state);
            }
        }
    }

    /// replayResolvedDraw
    ///
    /// Resolves the deterministic draw once and replays it later by forcing one remembered playable tile id.
    /// Unplaceable tiles are skipped while choosing that tile and are not stored in the node.
    private void replayResolvedDraw(State state) {
        if (drawnTile == null) {
            var playableTile = state.getTileDeck().findNextTileSoft(agent.rand, state.getFrontier());
            drawnTile = playableTile != null ? playableTile.getTileId() : -1;
        }

        if (drawnTile >= 0) {
            PerformActionManager.performDrawAction(state, drawnTile);
            return;
        }

        resolveTerminalDrawState(state);
    }

    /// resolveTerminalDrawState
    ///
    /// Loops through and performs draw actions until a terminal state is reached or a tile is drawn.
    ///
    /// @param state the current state of the game
    private void resolveTerminalDrawState(State state) {
        while (!state.isGameOver() && state.getCurrentTile() == null) {
            int tileId = PerformActionManager.determineNextDrawAction(state, agent.rand);
            if (tileId < 0) {
                break;
            }
            PerformActionManager.performDrawAction(state, tileId);
        }
    }

    /// getChildren
    ///
    /// Returns the array of child nodes generated from this node.
    ///
    /// @return the children array
    @Override
    public N[] getChildren() {
        return children;
    }

    /// getBestChild
    ///
    /// Returns the child node that has been visited the most/has the highest average utility.
    ///
    /// @return the best visited child node
    public N getBestChild() {
        return bestVisitedChild(children);
    }

    /// getAction
    ///
    /// Returns the action associated with this node.
    ///
    /// @return the action integer representation
    public int getAction() {
        return action;
    }
}
