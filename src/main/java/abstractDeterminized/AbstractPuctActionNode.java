package abstractDeterminized;

import core.AgentUtil;
import model.bits.CarcassonneActionLayoutBit;
import model.collections.ActionSet;
import model.heuristic.HeuristicConfiguration;
import model.state.HeuristicManager;
import model.state.PerformActionManager;
import model.state.State;

public abstract class AbstractPuctActionNode<N extends AbstractPuctActionNode<N>> extends DeterminizedActionNode<N> {
    @FunctionalInterface
    protected interface HeuristicChildFactory<N> {
        N create(int action, float score);
    }

    @FunctionalInterface
    protected interface RolloutActionSelector {
        int choose(State state, ActionSet actions);
    }

    protected final float heuristic;
    protected final float explorationCoefficient;
    protected final HeuristicConfiguration heuristicConfiguration;

    protected AbstractPuctActionNode(
            AbstractDeterminizedAgent<N> agent,
            N parent,
            int action,
            State checkpoint,
            float heuristic,
            float explorationCoefficient,
            HeuristicConfiguration heuristicConfiguration
    ) {
        super(agent, parent, action, checkpoint);
        this.heuristic = heuristic;
        this.explorationCoefficient = explorationCoefficient;
        this.heuristicConfiguration = heuristicConfiguration;
    }

    protected final float[] computeHeuristicScores(State state, ActionSet actions) {
        float[] scores = new float[actions.size()];
        int cachedPositionRotation = Integer.MIN_VALUE;
        float cachedTileScore = 0f;

        for (int i = 0; i < actions.size(); i++) {
            int candidateAction = actions.get(i);
            int x = CarcassonneActionLayoutBit.getX(candidateAction);
            int y = CarcassonneActionLayoutBit.getY(candidateAction);
            int rotation = CarcassonneActionLayoutBit.getRotation(candidateAction);
            int positionRotationKey = (x << 10) ^ (y << 2) ^ rotation;

            if (positionRotationKey != cachedPositionRotation) {
                cachedPositionRotation = positionRotationKey;
                cachedTileScore = HeuristicManager.tilePlacementScore(state, x, y, rotation, heuristicConfiguration.positionHeuristik());
            }

            scores[i] = HeuristicManager.computePrior(state, candidateAction, cachedTileScore, heuristicConfiguration);
        }

        return scores;
    }

    protected final N expandWithHeuristicScores(State state, boolean normalizeScores, HeuristicChildFactory<N> childFactory) {
        if (children != null || state.isGameOver()) {
            return self();
        }

        ActionSet actions = (ActionSet) state.calculatePossibleActionsUnique();
        if (actions.isEmpty()) {
            children = newChildrenArray(0);
            return self();
        }

        children = newChildrenArray(actions.size());
        float[] scores = computeHeuristicScores(state, actions);
        if (normalizeScores) {
            AgentUtil.normalizeInPlace(scores);
        }

        for (int i = 0; i < actions.size(); i++) {
            children[i] = childFactory.create(actions.get(i), scores[i]);
        }

        shuffleChildren(children);
        return self();
    }

    protected final int chooseGreedyHeuristicAction(State state, ActionSet actions, double temperature) {
        float[] scores = computeHeuristicScores(state, actions);
        AgentUtil.normalizeInPlace(scores);
        return actions.get(AgentUtil.sampleWithTemperature(scores, temperature, agent.rand));
    }

    protected final float averageHeuristicRollout(State state, int rollouts, RolloutActionSelector actionSelector) {
        float utilitySum = 0f;
        for (int i = 0; i < rollouts; i++) {
            State rolloutState = (i < rollouts - 1) ? state.deepCopy() : state;
            utilitySum += heuristicRollout(rolloutState, actionSelector);
        }
        return utilitySum / rollouts;
    }

    protected final float heuristicRollout(State state, RolloutActionSelector actionSelector) {
        while (!state.isGameOver()) {
            if (state.getCurrentTile() == null) {
                int tileId = PerformActionManager.determineNextDrawAction(state, agent.rand);
                if (tileId >= 0) {
                    PerformActionManager.performDrawAction(state, tileId);
                } else {
                    PerformActionManager.performRandomDrawAction(state, agent.rand);
                }
            } else {
                ActionSet actions = (ActionSet) state.calculatePossibleActionsUnique();
                performEncodedAction(state, actionSelector.choose(state, actions));
            }
        }
        return agent.utility(state);
    }

    protected final void performEncodedAction(State state, int action) {
        PerformActionManager.performAction(
                state,
                CarcassonneActionLayoutBit.getX(action),
                CarcassonneActionLayoutBit.getY(action),
                CarcassonneActionLayoutBit.getRotation(action),
                CarcassonneActionLayoutBit.getAreaId(action)
        );
    }
}
