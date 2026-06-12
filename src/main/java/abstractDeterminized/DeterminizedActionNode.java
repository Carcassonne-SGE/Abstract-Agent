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

    public DeterminizedActionNode(AbstractDeterminizedAgent<N> agent, N parent, int action, State checkpoint) {
        super(agent, parent, checkpoint);
        this.action = action;
    }

    @SuppressWarnings("unchecked")
    protected N[] newChildrenArray(int size) {
        return (N[]) new DeterminizedActionNode<?>[size];
    }

    @Override
    public N select() {
        if (children == null || children.length == 0) {
            return self();
        }

        N best = selectUnvisitedOrBestUsb(children, children.length);
        assert best != null;
        return best.getVisits() == 0 ? best : best.select();
    }

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

    @Override
    public float simulate(State state) {
        return averageRollout(state);
    }

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

    private void resolveTerminalDrawState(State state) {
        while (!state.isGameOver() && state.getCurrentTile() == null) {
            int tileId = PerformActionManager.determineNextDrawAction(state, agent.rand);
            if (tileId < 0) {
                break;
            }
            PerformActionManager.performDrawAction(state, tileId);
        }
    }

    @Override
    public N[] getChildren() {
        return children;
    }


    public N getBestChild() {
        return bestVisitedChild(children);
    }

    public int getAction() {
        return action;
    }
}
