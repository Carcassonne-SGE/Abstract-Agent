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

    public static long computeSearchDeadlineNanos(long startTimeNanos, long computationTime, TimeUnit timeUnit) {
        long requestedBudgetNanos = timeUnit.toNanos(computationTime);
        long marginNanos = Math.min(requestedBudgetNanos / 10, TimeUnit.MILLISECONDS.toNanos(10));
        return startTimeNanos + Math.max(0, requestedBudgetNanos - marginNanos);
    }

    public static void logAgentConfiguration(Logger logger, int playerId, AbstractAgentConfiguration config) {
        if (logger != null) {
            logger._infof("Using player %d config: %s", playerId, config);
        }
    }

    public static void logSearchSummary(Logger logger, int playerId, long startTimeNanos, int iterations) {
        if (logger != null) {
            double elapsedSeconds = (System.nanoTime() - startTimeNanos) / 1_000_000_000.0;
            logger._infof(
                    "MCTS Player %d searched for %.3f seconds and completed %d iterations",
                    playerId,
                    elapsedSeconds,
                    iterations
            );
        }
    }

    public static void logIterations(Logger logger, int playerId, int iterations) {
        if (logger != null && logger.isDebug()) {
            logger.debf_(
                    "MCTS Player %d iterations:",
                    playerId,
                    iterations
            );
        }
    }

    private static String formatAction(int action) {
        return String.format(
                "enc=%d x=%d y=%d rot=%d area=%d tile=%d isAction=%s",
                action,
                CarcassonneActionLayoutBit.getX(action),
                CarcassonneActionLayoutBit.getY(action),
                CarcassonneActionLayoutBit.getRotation(action),
                CarcassonneActionLayoutBit.getAreaId(action),
                CarcassonneActionLayoutBit.getTileId(action),
                CarcassonneActionLayoutBit.getIsAction(action)
        );
    }
}
