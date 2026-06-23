package abstractDeterminized;

import at.ac.tuwien.ifs.sge.engine.Logger;
import core.AbstractAgentConfiguration;
import core.AgentHelper;
import core.AgentUtil;
import core.AbstractMctsAgent;
import model.state.State;
import sge.CarcassonneAction;
import sge.CarcassonneGame;

import java.util.Random;
import java.util.concurrent.TimeUnit;

/// AbstractDeterminizedAgent
///
/// handles the stochastic part of Carcassonne by first Determinize the deck
/// so that it is no longer random.
///
/// Keep in mind that the real game is always random, this agent just ignores that fact
public abstract class AbstractDeterminizedAgent<N extends DeterminizedActionNode<N>> extends AbstractMctsAgent<N> {
    /// AbstractDeterminizedAgent
    ///
    /// Constructor for the AbstractDeterminizedAgent.
    ///
    /// @param logger the logger instance used for reporting
    /// @param config the configuration settings for the agent
    /// @param rand the random number generator
    public AbstractDeterminizedAgent(Logger logger, AbstractAgentConfiguration config, Random rand) {
        super(logger, config, rand);
    }

    /// rootFactory
    ///
    /// Abstract factory method to create the root node of the MCTS tree.
    ///
    /// @param initialState the initial state at the root of the search
    /// @return the constructed root node
    protected abstract N rootFactory(State initialState);

    /// childFactory
    ///
    /// Abstract factory method to create a new child node in the MCTS tree.
    ///
    /// @param parent the parent node
    /// @param action the action leading from parent to the new child node
    /// @param checkpoint the game state corresponding to the new child node
    /// @return the constructed child node
    public abstract N childFactory(N parent, int action, State checkpoint);

    /// searchForEnsemble
    ///
    /// Performs MCTS search iterations on a determinized board state and returns the root node.
    /// This is used when the agent is run as part of an ensemble.
    ///
    /// @param rootState the determinized root state
    /// @param computationTime the search budget duration
    /// @param timeUnit the unit of computationTime
    /// @param startTimeNanos the starting timestamp of the overall ensemble search
    /// @return the root node containing the search statistics
    public final DeterminizedActionNode<?> searchForEnsemble(State rootState, long computationTime, TimeUnit timeUnit, long startTimeNanos) {
        N root = rootFactory(rootState);
        iterations(root, computationTime, timeUnit, startTimeNanos);
        return root;
    }

    /// selectAction
    ///
    /// Selects the best action from the root node based on visit counts and action values.
    /// Uses temperature-based softmax sampling to choose the action.
    ///
    /// @param root the root node of the search tree
    /// @return the selected action representation
    protected int selectAction(N root) {
        N[] children = root.getChildren();
        if (children == null || children.length == 0) {
            throw new IllegalStateException("MCTS failed to find any legal action");
        }

        int count = 0;
        for (N child : children) {
            if (child != null && child.getVisits() > 0) {
                count++;
            }
        }

        if (count == 0) {
            for (N child : children) {
                if (child != null) {
                    return child.getAction();
                }
            }
            throw new IllegalStateException("MCTS failed to find any non-null child");
        }

        float[] scores = new float[count];
        int[] actions = new int[count];
        int index = 0;

        for (N child : children) {
            if (child != null && child.getVisits() > 0) {
                float avgValue = child.getValue() / (float) child.getVisits();
                scores[index] = avgValue;
                actions[index] = child.getAction();
                index++;
            }
        }

        AgentUtil.normalizeInPlace(scores);
        int samp = AgentUtil.sampleWithTemperature(scores, 0.018, rand);
        return actions[samp];
    }

    /// computeNextAction
    ///
    /// Computes and returns the next action by determinizing the tile deck, building a search
    /// tree, executing MCTS iterations, and selecting the action from the root node.
    ///
    /// @param game the Carcassonne game instance
    /// @param computationTime the search budget duration
    /// @param timeUnit the unit of computationTime
    /// @return the selected CarcassonneAction
    @Override
    public CarcassonneAction computeNextAction(CarcassonneGame game, long computationTime, TimeUnit timeUnit) {
        long startTimeNanos = System.nanoTime();
        State rootState = game.getBoard().deepCopy();
        rootState.getTileDeck().determinize(rand);
        N root = rootFactory(rootState);
        int iterations = iterations(root, computationTime, timeUnit, startTimeNanos);
        AgentHelper.logSearchSummary(logger, playerId, startTimeNanos, iterations);
        int val = selectAction(root);
        return new CarcassonneAction(val);
    }
}
