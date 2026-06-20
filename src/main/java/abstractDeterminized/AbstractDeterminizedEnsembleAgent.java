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


    @FunctionalInterface
    public interface DeterminizedAgentSupplier {
        AbstractDeterminizedAgent<?> create(Logger logger, AbstractAgentConfiguration config, Random rand);
    }

    protected enum AggregationMode {
        POOLED_BY_VISITS,
        NORMALIZED_PER_TREE
    }

    protected static AggregationMode pooledByVisitsAggregationMode() {
        return AggregationMode.POOLED_BY_VISITS;
    }

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


    protected AbstractDeterminizedEnsembleAgent(
            Logger logger,
            AbstractAgentConfiguration config,
            int agentsCount,
            DeterminizedAgentSupplier agentSupplier,
            Random rand
    ) {
        this(logger, config, agentsCount, AggregationMode.POOLED_BY_VISITS, rand, agentSupplier);
    }

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

    @Override
    public void setUp(int numberOfPlayers, int playerNumber) {
        playerId = playerNumber;
        AgentHelper.logAgentConfiguration(logger, playerId, config);
    }

    @Override
    public void tearDown() {
    }

    @Override
    public CarcassonneAction computeNextAction(CarcassonneGame game, long computationTime, TimeUnit timeUnit) {
        long startTimeNanos = System.nanoTime();
        State rootState = game.getBoard();
        long totalBudgetNanos = Math.max(1L, timeUnit.toNanos(computationTime));

        int totalIterations = 0;
        int totalVisitedNodes = 0;
        double visitSum = 0.0;
        double childSum = 0.0;
        int depthSum = 0;
        int widthSum = 0;
        int maxDepth = 0;
        int maxWidth = 0;
        int searchesRun = 0;
        Map<Integer, Float> actionScoreSums = new HashMap<>();
        Map<Integer, Integer> actionWeights = new HashMap<>();

        for (int i = 0; i < agentsCount; i++) {
            long elapsedNanos = System.nanoTime() - startTimeNanos;
            long remainingBudgetNanos = totalBudgetNanos - elapsedNanos;
            if (remainingBudgetNanos <= 0) {
                break;
            }

            long agentBudgetNanos = Math.max(1L, remainingBudgetNanos / (agentsCount - i));
            State rootCopy = rootState.deepCopy();
            rootCopy.getTileDeck().determinize(rand);
            DeterminizedActionNode<?> root = agents[i].searchForEnsemble(
                    rootCopy,
                    agentBudgetNanos,
                    TimeUnit.NANOSECONDS,
                    System.nanoTime()
            );
            int localIterations = root.getVisits();
            int visitedNodes = countVisitedNodes(root);
            int depth = computeDepth(root);
            int width = computeMaxWidth(root);

            searchesRun++;
            totalIterations += localIterations;
            totalVisitedNodes += visitedNodes;
            visitSum += localIterations;
            childSum += countExpandedChildren(root);
            depthSum += depth;
            widthSum += width;
            maxDepth = Math.max(maxDepth, depth);
            maxWidth = Math.max(maxWidth, width);

            mergeActionStats(actionScoreSums, actionWeights, collectActionValueSums(root), collectActionVisits(root));
        }
        

        AgentHelper.logSearchSummary(logger, playerId, startTimeNanos, totalIterations);
        logEnsembleIterations(totalIterations, searchesRun, totalVisitedNodes, visitSum, childSum, depthSum, widthSum, maxDepth, maxWidth);

        int val = selectBestAction(rootState, actionScoreSums, actionWeights);
        return AgentUtil.decodeAction(val);
    }

    private void logEnsembleIterations(
            int totalIterations,
            int searchesRun,
            int totalVisitedNodes,
            double visitSum,
            double childSum,
            int depthSum,
            int widthSum,
            int maxDepth,
            int maxWidth
    ) {
        if (logger != null && logger.isDebug()) {
            logger.debf_(
                    "MCTS Ensemble Player %d searches: %d iterations: %d avgIterations: %.2f maxDepth: %d avgDepth: %.2f maxWidth: %d avgWidth: %.2f visitedNodes: %d avgVisits: %.2f avgChildren: %.2f",
                    playerId,
                    searchesRun,
                    totalIterations,
                    searchesRun == 0 ? 0.0 : totalIterations / (double) searchesRun,
                    maxDepth,
                    searchesRun == 0 ? 0.0 : depthSum / (double) searchesRun,
                    maxWidth,
                    searchesRun == 0 ? 0.0 : widthSum / (double) searchesRun,
                    totalVisitedNodes,
                    totalVisitedNodes == 0 ? 0.0 : visitSum / totalVisitedNodes,
                    totalVisitedNodes == 0 ? 0.0 : childSum / totalVisitedNodes
            );
        }
    }

    private int countVisitedNodes(DeterminizedActionNode<?> root) {
        if (root == null || root.getVisits() <= 0) {
            return 0;
        }

        int count = 1;
        DeterminizedActionNode<?>[] children = root.getChildren();
        if (children == null) {
            return count;
        }

        for (DeterminizedActionNode<?> child : children) {
            count += countVisitedNodes(child);
        }
        return count;
    }

    private int countExpandedChildren(DeterminizedActionNode<?> root) {
        if (root == null || root.getVisits() <= 0) {
            return 0;
        }

        int expandedChildren = 0;
        DeterminizedActionNode<?>[] children = root.getChildren();
        if (children == null) {
            return 0;
        }

        for (DeterminizedActionNode<?> child : children) {
            if (child != null && child.getVisits() > 0) {
                expandedChildren++;
            }
        }
        return expandedChildren;
    }

    private int computeDepth(DeterminizedActionNode<?> root) {
        if (root == null || root.getVisits() <= 0) {
            return 0;
        }

        int maxChildDepth = 0;
        DeterminizedActionNode<?>[] children = root.getChildren();
        if (children != null) {
            for (DeterminizedActionNode<?> child : children) {
                maxChildDepth = Math.max(maxChildDepth, computeDepth(child));
            }
        }
        return 1 + maxChildDepth;
    }

    private int computeMaxWidth(DeterminizedActionNode<?> root) {
        return computeMaxWidth(root, 0);
    }

    private int computeMaxWidth(DeterminizedActionNode<?> node, int depth) {
        if (node == null || node.getVisits() <= 0) {
            return 0;
        }

        int nodesAtDepth = depth == 0 ? 1 : 0;
        int maxWidth = nodesAtDepth;
        DeterminizedActionNode<?>[] children = node.getChildren();
        if (children != null) {
            for (DeterminizedActionNode<?> child : children) {
                maxWidth = Math.max(maxWidth, computeMaxWidth(child, depth + 1));
            }
        }
        return maxWidth;
    }

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

            if (aggregationMode == AggregationMode.NORMALIZED_PER_TREE) {
                actionScoreSums.merge(action, entry.getValue() / visits, Float::sum);
                actionWeights.merge(action, 1, Integer::sum);
            } else {
                actionScoreSums.merge(action, entry.getValue(), Float::sum);
                actionWeights.merge(action, visits, Integer::sum);
            }
        }
    }

    private int selectBestAction(State rootState, Map<Integer, Float> actionScoreSums, Map<Integer, Integer> actionWeights) {
        ActionSet actions = (ActionSet) rootState.calculatePossibleActionsUnique();
        if (actions.isEmpty()) {
            throw new IllegalStateException("Ensemble MCTS failed to find any legal action");
        }

        int count = 0;
        for (int i = 0; i < actions.size(); i++) {
            int action = actions.get(i);
            if (actionWeights.getOrDefault(action, 0) > 0) {
                count++;
            }
        }

        if (count == 0) {
            return actions.get(0);
        }

        float[] scores = new float[count];
        int[] actionIds = new int[count];
        int index = 0;

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

        AgentUtil.normalizeInPlace(scores);
        int samp = AgentUtil.sampleWithTemperature(scores, 0.018, rand);
        return actionIds[samp];
    }

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
