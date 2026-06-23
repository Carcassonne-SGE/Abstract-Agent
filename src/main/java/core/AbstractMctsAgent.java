package core;

import at.ac.tuwien.ifs.sge.agent.GameAgent;
import at.ac.tuwien.ifs.sge.engine.Logger;
import model.state.State;
import sge.CarcassonneAction;
import sge.CarcassonneGame;

import java.util.Random;
import java.util.concurrent.TimeUnit;

/// AbstractMctsAgent
///
/// base agent class specialized for the Carcassonne Environment
public abstract class AbstractMctsAgent<N extends AbstractNode<?,N>>
        implements GameAgent<CarcassonneGame, CarcassonneAction> {


    public final Random rand;
    protected final AbstractAgentConfiguration config;
    protected final Logger logger;
    protected int playerId;

    /// AbstractMctsAgent
    ///
    /// Constructor for the abstract MCTS agent. Initializes the logger, configuration,
    /// and random number generator.
    ///
    /// @param logger the logger instance used for reporting
    /// @param config the configuration settings for the agent
    /// @param rand the random number generator, or null to initialize a new one
    protected AbstractMctsAgent(Logger logger, AbstractAgentConfiguration config, Random rand) {
        this.logger = logger;
        this.config = config;
        this.rand = rand == null ? new Random():rand;
    }

    /// computeNextAction
    ///
    /// Computes and returns the next action for the agent to take in the current game state
    /// within the given computation budget.
    ///
    /// @param game the Carcassonne game instance
    /// @param computationTime the search budget duration
    /// @param timeUnit the unit of computationTime
    /// @return the selected CarcassonneAction
    @Override
    public abstract CarcassonneAction computeNextAction(CarcassonneGame game, long computationTime, TimeUnit timeUnit);

    /// setUp
    ///
    /// Sets up the agent before the game starts. Configures the player ID and
    /// logs the agent configuration.
    ///
    /// @param numberOfPlayers the total number of players in the game
    /// @param playerNumber the player number/ID assigned to this agent
    @Override
    public void setUp(int numberOfPlayers, int playerNumber) {
        playerId = playerNumber;
        AgentHelper.logAgentConfiguration(logger, playerId, config);
    }


    /// iterations
    ///
    /// Helper function to perform a standard MCTS iteration with the provided root node
    protected final int iterations(N root, long computationTime, TimeUnit timeUnit, long startTimeNanos) {
        long deadline = AgentHelper.computeSearchDeadlineNanos(startTimeNanos, computationTime, timeUnit);
        int iterations = 0;

        while (System.nanoTime() <= deadline) {
            N selected = root.select();
            var nodeState =  selected.recreateState();
            N simulationNode = selected.expand(nodeState);
            float value = simulationNode.simulate(nodeState);
            simulationNode.backpropagate(value);
            iterations++;
        }

        return iterations;
    }



    /// utility
    /// the Utility of a state from the perspective of a player
    /// @param  state the state for which the utility is calculated
    public float utility(State state) {
        return (float) state.getCollaborativeUtility();
    }

    /// ucbTransform
    ///
    /// Transforms the raw average simulation value for use in UCB calculations.
    /// By default, returns the value unchanged.
    ///
    /// @param value the raw average simulation value
    /// @return the transformed value
    public float ucbTransform(double value) {
        return (float) value;
    }
}
