package core;

import model.bits.CarcassonneActionLayoutBit;
import sge.CarcassonneAction;

import java.util.Random;

public final class AgentUtil {
    private static final float NORMALIZATION_EPSILON = 0.0000000000001f;

    private AgentUtil() {
    }

    public static void normalizeInPlace(float[] values) {
        if (values.length == 0) {
            return;
        }

        float min = Float.POSITIVE_INFINITY;
        float max = Float.NEGATIVE_INFINITY;
        for (float value : values) {
            min = Math.min(min, value);
            max = Math.max(max, value);
        }

        float range = max - min + NORMALIZATION_EPSILON;
        for (int i = 0; i < values.length; i++) {
            values[i] = (values[i] - min) / range;
        }
    }

    public static int sampleWithTemperature(float[] probabilities, double temperature, Random rand) {
        int len = probabilities.length;
        double[] expValues = new double[len];
        double sum = 0.0;

        for (int i = 0; i < len; i++) {
            double t = Math.max(temperature, 0.0001);
            expValues[i] = Math.exp(probabilities[i] / t);
            sum += expValues[i];
        }

        double[] targetPdf = new double[len];
        for (int i = 0; i < len; i++) {
            targetPdf[i] = expValues[i] / sum;
        }

        double r = rand.nextDouble();
        double cumulativeSum = 0.0;
        for (int i = 0; i < len; i++) {
            cumulativeSum += targetPdf[i];
            if (r <= cumulativeSum) {
                return i;
            }
        }

        return len - 1;
    }

    public static CarcassonneAction decodeAction(int action) {
        return new CarcassonneAction(
                CarcassonneActionLayoutBit.getIsAction(action),
                CarcassonneActionLayoutBit.getX(action),
                CarcassonneActionLayoutBit.getY(action),
                CarcassonneActionLayoutBit.getRotation(action),
                CarcassonneActionLayoutBit.getAreaId(action),
                CarcassonneActionLayoutBit.getTileId(action)
        );
    }
}
