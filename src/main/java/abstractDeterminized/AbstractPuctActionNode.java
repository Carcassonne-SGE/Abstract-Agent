package abstractDeterminized;

import core.AgentUtil;
import model.bits.CarcassonneActionLayoutBit;
import model.collections.ActionSet;
import model.heuristic.HeuristicConfiguration;
import model.state.HeuristicManager;
import model.state.PerformActionManager;
import model.state.State;

/// AbstractPuctActionNode
///
/// Abstract base class for PUCT (Predictor Upper Confidence trees) nodes.
/// Supports computation and expansion using heuristic-guided prior probabilities.
public abstract class AbstractPuctActionNode<N extends AbstractPuctActionNode<N>> extends DeterminizedActionNode<N> {
    /// HeuristicChildFactory
    ///
    /// Functional interface for creating child nodes with a heuristic score.
    @FunctionalInterface
    protected interface HeuristicChildFactory<N> {
        N create(int action, float score);
    }

    /// RolloutActionSelector
    ///
    /// Functional interface for choosing an action during a rollout.
    @FunctionalInterface
    protected interface RolloutActionSelector {
        int choose(State state, ActionSet actions);
    }

    protected final float heuristic;
    protected final float explorationCoefficient;
    protected final HeuristicConfiguration heuristicConfiguration;

    /// AbstractPuctActionNode
    ///
    /// Constructor for the AbstractPuctActionNode. Initializes the node with the
    /// agent, parent, action, checkpoint, and PUCT/heuristic configurations.
    ///
    /// @param agent the determinized MCTS agent
    /// @param parent the parent node in the tree
    /// @param action the action that led to this node
    /// @param checkpoint the game state checkpoint at this node
    /// @param heuristic the weight of the heuristic in selection
    /// @param explorationCoefficient the exploration constant (c)
    /// @param heuristicConfiguration the heuristic configurations used by the agent
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

    /// getSelectionScore
    ///
    /// Computes the selection score based on UCB and PUCT heuristic prior.
    ///
    /// @return the selection score
    @Override
    public float getSelectionScore() {
        if (visits == 0) {
            return Float.MAX_VALUE;
        }
        float q = agent.ucbTransform(value / (float) visits);
        if (parent == null || parent.getVisits() <= 1) {
            return q;
        }
        float b = (float) (Math.sqrt(parent.getVisits()) / (1 + visits));
        return q + explorationCoefficient * heuristic * b;
    }



    /// expandWithHeuristicScores
    ///
    /// Expands the node by calculating heuristic scores for all possible actions,
    /// optionally normalizing them, and creating child nodes using the child factory.
    ///
    /// @param state the current state of the game
    /// @param normalizeScores whether to normalize the computed scores using min-max normalization
    /// @param childFactory the factory to instantiate child nodes
    /// @return the current node after expansion
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
        float[] scores = HeuristicManager.computePriors(state, actions, heuristicConfiguration);
        if (normalizeScores) {
            AgentUtil.normalizeInPlace(scores);
        }

        for (int i = 0; i < actions.size(); i++) {
            children[i] = childFactory.create(actions.get(i), scores[i]);
        }

        shuffleChildren(children);
        return self();
    }

    /// chooseGreedyHeuristicAction
    ///
    /// Chooses an action from the possible actions set greedily using softmax sampling
    /// over the normalized heuristic scores at the given temperature.
    ///
    /// @param state the current state of the game
    /// @param actions the set of possible actions
    /// @param temperature the temperature for softmax sampling
    /// @return the chosen action integer representation
    protected final int chooseGreedyHeuristicAction(State state, ActionSet actions, double temperature) {
        float[] scores = HeuristicManager.computePriors(state, actions, heuristicConfiguration);
        AgentUtil.normalizeInPlace(scores);
        return actions.get(AgentUtil.sampleWithTemperature(scores, temperature, agent.rand));
    }

    /// averageHeuristicRollout
    ///
    /// Performs multiple heuristic-based rollouts and returns the average utility value.
    ///
    /// @param state the starting state of the rollout
    /// @param rollouts the number of rollout simulations to perform
    /// @param actionSelector the selector used to choose actions during the rollouts
    /// @return the average utility value
    protected final float averageHeuristicRollout(State state, int rollouts, RolloutActionSelector actionSelector) {
        float utilitySum = 0f;
        for (int i = 0; i < rollouts; i++) {
            State rolloutState = (i < rollouts - 1) ? state.deepCopy() : state;
            utilitySum += heuristicRollout(rolloutState, actionSelector);
        }
        return utilitySum / rollouts;
    }

    /// heuristicRollout
    ///
    /// Performs a single rollout from the given state using the action selector
    /// to choose actions until the game is over.
    ///
    /// @param state the starting state of the rollout
    /// @param actionSelector the selector used to choose actions during the rollout
    /// @return the utility value of the final terminal state
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

    /// performEncodedAction
    ///
    /// Decodes and performs the specified action on the game state.
    ///
    /// @param state the current state of the game
    /// @param action the action to perform
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
