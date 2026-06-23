package core;

import at.ac.tuwien.ifs.sge.engine.Logger;
import model.bits.CarcassonneActionLayoutBit;

import java.util.concurrent.TimeUnit;

/// AgentHelper
///
/// Shared helper methods for agent logging and search budget management.
public final class AgentHelper {
    private AgentHelper() {
    }

    /// computeSearchDeadlineNanos
    ///
    /// Calculates the search deadline with a safety margin to prevent timeouts.
    ///
    /// @param startTimeNanos search start time in nanoseconds
    /// @param computationTime search budget duration
    /// @param timeUnit unit of computationTime
    public static long computeSearchDeadlineNanos(long startTimeNanos, long computationTime, TimeUnit timeUnit) {
        long requestedBudgetNanos = timeUnit.toNanos(computationTime);
        long marginNanos = Math.min(requestedBudgetNanos / 10, 100_000_000L); // max 100ms margin to prevent timeouts
        return startTimeNanos + Math.max(0, requestedBudgetNanos - marginNanos);
    }

    /// logAgentConfiguration
    ///
    /// Logs the configuration settings of the agent if a valid logger is provided.
    ///
    /// @param logger the logger instance used for reporting, or null
    /// @param playerId the zero-based ID of the player
    /// @param config the configuration settings of the agent
    public static void logAgentConfiguration(Logger logger, int playerId, AbstractAgentConfiguration config) {
        if (logger != null) {
            logger._infof("Using player %d config: %s", playerId, config);
        }
    }

    /// logSearchSummary
    ///
    /// Logs a summary of the MCTS search, including the elapsed time in seconds
    /// and the number of completed iterations.
    ///
    /// @param logger the logger instance used for reporting, or null
    /// @param playerId the zero-based ID of the player
    /// @param startTimeNanos the start time of the search in nanoseconds
    /// @param iterations the number of search iterations completed
    public static void logSearchSummary(Logger logger, int playerId, long startTimeNanos, int iterations) {
        if (logger != null) {
            double elapsedSeconds = (System.nanoTime() - startTimeNanos) / 1_000_000_000.0;
            logger._infof(
                    "MCTS Player %d searched for %.3f seconds and completed %d iterations",
                    playerId,
                    elapsedSeconds,
                    iterations);
        }
    }

    /// logIterations
    ///
    /// Logs the number of MCTS iterations at debug level if logging is enabled.
    ///
    /// @param logger the logger instance used for reporting, or null
    /// @param playerId the zero-based ID of the player
    /// @param iterations the number of search iterations completed
    public static void logIterations(Logger logger, int playerId, int iterations) {
        if (logger != null && logger.isDebug()) {
            logger.debf_(
                    "MCTS Player %d iterations:",
                    playerId,
                    iterations);
        }
    }
}
