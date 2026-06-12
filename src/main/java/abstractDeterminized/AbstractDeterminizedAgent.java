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
/// Keep in mind that the real game is always random this agend just ignores that fact
public abstract class AbstractDeterminizedAgent<N extends DeterminizedActionNode<N>> extends AbstractMctsAgent<N> {
    public AbstractDeterminizedAgent(Logger logger, AbstractAgentConfiguration config, Random rand) {
        super(logger, config, rand);
    }

    protected abstract N rootFactory(State initialState);

    public abstract N childFactory(N parent, int action, State checkpoint);

    public final DeterminizedActionNode<?> searchForEnsemble(State rootState, long computationTime, TimeUnit timeUnit, long startTimeNanos) {
        N root = rootFactory(rootState);
        iterations(root, computationTime, timeUnit, startTimeNanos);
        return root;
    }

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

    @Override
    public CarcassonneAction computeNextAction(CarcassonneGame game, long computationTime, TimeUnit timeUnit) {
        long startTimeNanos = System.nanoTime();
        State rootState = game.getBoard();
        N root = rootFactory(rootState);
        int iterations = iterations(root, computationTime, timeUnit, startTimeNanos);
        AgentHelper.logSearchSummary(logger, playerId, startTimeNanos, iterations);
        int val = selectAction(root);
        return AgentUtil.decodeAction(val);
    }
}
