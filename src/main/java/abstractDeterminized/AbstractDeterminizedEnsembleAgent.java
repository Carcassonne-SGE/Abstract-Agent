package abstractDeterminized;

import at.ac.tuwien.ifs.sge.agent.GameAgent;
import at.ac.tuwien.ifs.sge.engine.Logger;
import core.AbstractAgentConfiguration;
import core.AgentHelper;
import core.AgentUtil;
import model.collections.ActionSet;
import model.state.State;
import sge.CarcassonneAction;
import sge.CarcassonneGame;

import java.util.HashMap;
import java.util.Map;
import java.util.Random;
import java.util.concurrent.TimeUnit;

/// AbstractDeterminizedEnsembleAgent
///
/// Runs several determinized searches and merges their root action statistics.
public abstract class AbstractDeterminizedEnsembleAgent implements GameAgent<CarcassonneGame, CarcassonneAction> {


    /// DeterminizedAgentSupplier
    ///
    /// Functional interface used to supply/create new instances of AbstractDeterminizedAgent.
    @FunctionalInterface
    public interface DeterminizedAgentSupplier {
        AbstractDeterminizedAgent<?> create(Logger logger, AbstractAgentConfiguration config, Random rand);
    }

    /// AggregationMode
    ///
    /// Enum representing the modes used to aggregate search statistics across the ensemble trees.
    protected enum AggregationMode {
        POOLED_BY_VISITS,
        NORMALIZED_PER_TREE
    }

    /// pooledByVisitsAggregationMode
    ///
    /// Returns the aggregation mode POOLED_BY_VISITS.
    ///
    /// @return the POOLED_BY_VISITS aggregation mode
    protected static AggregationMode pooledByVisitsAggregationMode() {
        return AggregationMode.POOLED_BY_VISITS;
    }

    /// normalizedPerTreeAggregationMode
    ///
    /// Returns the aggregation mode NORMALIZED_PER_TREE.
    ///
    /// @return the NORMALIZED_PER_TREE aggregation mode
    protected static AggregationMode normalizedPerTreeAggregationMode() {
        return AggregationMode.NORMALIZED_PER_TREE;
    }

    private final int agentsCount;
    private final AggregationMode aggregationMode;
    protected final AbstractDeterminizedAgent<?>[] agents;
    protected final AbstractAgentConfiguration config;
    protected final Logger logger;
    protected final Random rand;
    protected int playerId;


    /// AbstractDeterminizedEnsembleAgent
    ///
    /// Constructor for the ensemble agent using the default aggregation mode (POOLED_BY_VISITS).
    ///
    /// @param logger the logger instance used for reporting
    /// @param config the configuration settings for the agent
    /// @param agentsCount the number of sub-agents in the ensemble
    /// @param agentSupplier the supplier to create sub-agents
    /// @param rand the random number generator
    protected AbstractDeterminizedEnsembleAgent(
            Logger logger,
            AbstractAgentConfiguration config,
            int agentsCount,
            DeterminizedAgentSupplier agentSupplier,
            Random rand
    ) {
        this(logger, config, agentsCount, AggregationMode.POOLED_BY_VISITS, rand, agentSupplier);
    }

    /// AbstractDeterminizedEnsembleAgent
    ///
    /// Constructor for the ensemble agent with a specified aggregation mode.
    ///
    /// @param logger the logger instance used for reporting
    /// @param config the configuration settings for the agent
    /// @param agentsCount the number of sub-agents in the ensemble
    /// @param aggregationMode the method for combining sub-agent search statistics
    /// @param rand the random number generator
    /// @param agentSupplier the supplier to create sub-agents
    protected AbstractDeterminizedEnsembleAgent(
            Logger logger,
            AbstractAgentConfiguration config,
            int agentsCount,
            AggregationMode aggregationMode,
            Random rand,
            DeterminizedAgentSupplier agentSupplier
    ) {
        this.logger = logger;
        this.config = config;
        this.rand = rand == null ? new Random() : rand;
        if (agentsCount <= 0) {
            throw new IllegalArgumentException("agentsCount must be positive");
        }
        this.agentsCount = agentsCount;
        this.aggregationMode = aggregationMode;
        this.agents = new AbstractDeterminizedAgent<?>[agentsCount];
        for (int i = 0; i < agentsCount; i++) {
            agents[i] = agentSupplier.create(logger, config, this.rand);
        }
    }

    /// setUp
    ///
    /// Sets up the ensemble agent, configuring player ID and logging the agent config.
    ///
    /// @param numberOfPlayers the total number of players
    /// @param playerNumber the player ID assigned to this agent
    @Override
    public void setUp(int numberOfPlayers, int playerNumber) {
        playerId = playerNumber;
        AgentHelper.logAgentConfiguration(logger, playerId, config);
    }

    /// tearDown
    ///
    /// Performs teardown operations when the game or search finishes.
    @Override
    public void tearDown() {
    }

    /// computeNextAction
    ///
    /// Computes and returns the next action by running multiple determinized MCTS searches
    /// in sequential order and merging their results according to the aggregation mode.
    ///
    /// @param game the Carcassonne game instance
    /// @param computationTime the search budget duration
    /// @param timeUnit the unit of computationTime
    /// @return the selected CarcassonneAction
    @Override
    public CarcassonneAction computeNextAction(CarcassonneGame game, long computationTime, TimeUnit timeUnit) {
        long startTimeNanos = System.nanoTime();
        State rootState = game.getBoard();
        long totalBudgetNanos = Math.max(1L, timeUnit.toNanos(computationTime));

        int totalIterations = 0;
        Map<Integer, Float> actionScoreSums = new HashMap<>();
        Map<Integer, Integer> actionWeights = new HashMap<>();

        // Loop over each ensemble member and allocate a portion of the remaining time budget
        for (int i = 0; i < agentsCount; i++) {
            long elapsedNanos = System.nanoTime() - startTimeNanos;
            long remainingBudgetNanos = totalBudgetNanos - elapsedNanos;
            if (remainingBudgetNanos <= 0) {
                break;
            }

            // Distribute remaining search budget equally among remaining sub-agents
            long agentBudgetNanos = Math.max(1L, remainingBudgetNanos / (agentsCount - i));
            State rootCopy = rootState.deepCopy();
            // Determinize/shuffle the tile deck to simulate a random determinization for this sub-agent
            rootCopy.getTileDeck().determinize(rand);
            
            // Execute the MCTS search for this sub-agent
            DeterminizedActionNode<?> root = agents[i].searchForEnsemble(
                    rootCopy,
                    agentBudgetNanos,
                    TimeUnit.NANOSECONDS,
                    System.nanoTime()
            );
            int localIterations = root.getVisits();

            totalIterations += localIterations;

            // Merge the resulting action statistics into the ensemble data collections
            mergeActionStats(actionScoreSums, actionWeights, collectActionValueSums(root), collectActionVisits(root));
        }
        

        AgentHelper.logSearchSummary(logger, playerId, startTimeNanos, totalIterations);

        int val = selectBestAction(rootState, actionScoreSums, actionWeights);
        return new CarcassonneAction(val);
    }

    /// mergeActionStats
    ///
    /// Merges the search statistics of a single search tree into the global ensemble stats
    /// based on the selected AggregationMode.
    ///
    /// @param actionScoreSums map storing the accumulated scores of actions
    /// @param actionWeights map storing the accumulated weights/visits of actions
    /// @param actionValueSums action value sums from the current search tree
    /// @param actionVisits action visits from the current search tree
    private void mergeActionStats(
            Map<Integer, Float> actionScoreSums,
            Map<Integer, Integer> actionWeights,
            Map<Integer, Float> actionValueSums,
            Map<Integer, Integer> actionVisits
    ) {
        for (Map.Entry<Integer, Float> entry : actionValueSums.entrySet()) {
            int action = entry.getKey();
            int visits = actionVisits.getOrDefault(action, 0);
            if (visits <= 0) {
                continue;
            }

            // NORMALIZED_PER_TREE aggregation: average the scores within each tree first
            // and increment the tree count weight by 1.
            if (aggregationMode == AggregationMode.NORMALIZED_PER_TREE) {
                actionScoreSums.merge(action, entry.getValue() / visits, Float::sum);
                actionWeights.merge(action, 1, Integer::sum);
            } else {
                // POOLED_BY_VISITS aggregation: pool raw value sums and visits together
                actionScoreSums.merge(action, entry.getValue(), Float::sum);
                actionWeights.merge(action, visits, Integer::sum);
            }
        }
    }

    /// selectBestAction
    ///
    /// Selects the best action from the legal action set based on the aggregated ensemble statistics.
    /// Uses softmax sampling with a low temperature to select the action.
    ///
    /// @param rootState the root game state
    /// @param actionScoreSums map of accumulated action scores
    /// @param actionWeights map of accumulated action weights/visits
    /// @return the selected action representation
    private int selectBestAction(State rootState, Map<Integer, Float> actionScoreSums, Map<Integer, Integer> actionWeights) {
        ActionSet actions = (ActionSet) rootState.calculatePossibleActionsUnique();
        if (actions.isEmpty()) {
            throw new IllegalStateException("Ensemble MCTS failed to find any legal action");
        }

        // Count how many of the legal actions actually received search visits across the ensemble
        int count = 0;
        for (int i = 0; i < actions.size(); i++) {
            int action = actions.get(i);
            if (actionWeights.getOrDefault(action, 0) > 0) {
                count++;
            }
        }

        // Fallback: if no actions were visited (e.g. timeout), pick the first legal action
        if (count == 0) {
            return actions.get(0);
        }

        float[] scores = new float[count];
        int[] actionIds = new int[count];
        int index = 0;

        // Calculate average values and populate arrays for temperature/softmax sampling
        for (int i = 0; i < actions.size(); i++) {
            int action = actions.get(i);
            int weight = actionWeights.getOrDefault(action, 0);
            if (weight > 0) {
                float avgValue = actionScoreSums.get(action) / (float) weight;
                scores[index] = avgValue;
                actionIds[index] = action;
                index++;
            }
        }

        // Normalize scores to form a probability distribution
        AgentUtil.normalizeInPlace(scores);
        // Softmax temperature sampling to pick the final action representation
        int samp = AgentUtil.sampleWithTemperature(scores, 0.018, rand);
        return actionIds[samp];
    }

    /// collectActionValueSums
    ///
    /// Collects and returns the sum of child node values for each action from the given root node.
    ///
    /// @param root the root node
    /// @return a map of action values
    private Map<Integer, Float> collectActionValueSums(DeterminizedActionNode<?> root) {
        Map<Integer, Float> actionValueSums = new HashMap<>();
        DeterminizedActionNode<?>[] children = root.getChildren();
        if (children == null) {
            return actionValueSums;
        }

        for (DeterminizedActionNode<?> child : children) {
            if (child != null && child.getVisits() > 0) {
                actionValueSums.put(child.getAction(), child.getValue());
            }
        }
        return actionValueSums;
    }

    /// collectActionVisits
    ///
    /// Collects and returns the visit counts of child nodes for each action from the given root node.
    ///
    /// @param root the root node
    /// @return a map of action visit counts
    private Map<Integer, Integer> collectActionVisits(DeterminizedActionNode<?> root) {
        Map<Integer, Integer> actionVisits = new HashMap<>();
        DeterminizedActionNode<?>[] children = root.getChildren();
        if (children == null) {
            return actionVisits;
        }

        for (DeterminizedActionNode<?> child : children) {
            if (child != null && child.getVisits() > 0) {
                actionVisits.put(child.getAction(), child.getVisits());
            }
        }
        return actionVisits;
    }
}
