package core;

import model.bits.CarcassonneActionLayoutBit;
import sge.CarcassonneAction;

import java.util.Random;

/// Agent Util 
/// 
/// This class provides static helper function that can be used by the actual 
/// implementations. This class provides functionality for selecting the actual action
/// 
//  Allows to Interprets values as probabilities and samples from it. Has shown to
//  work better than greedy selection(at least in my experiments )
public final class AgentUtil {
    /// Small constant to prevent division by zero 
    private static final float NORMALIZATION_EPSILON = 0.0000000000001f;

    private AgentUtil() {
    }

    /// normalizeInPlace
    /// 
    /// takes an array of values and performs min, max normalization so that
    /// all values in [0,1]
    public static void normalizeInPlace(float[] values) {
        if (values.length == 0) {
            // check that parameter is valid
            return;
        }

        float min = Float.POSITIVE_INFINITY;
        float max = Float.NEGATIVE_INFINITY;
        for (float value : values) {
            min = Math.min(min, value);
            max = Math.max(max, value);
        }

        // prevent division by zero use epsilon
        float range = max - min + NORMALIZATION_EPSILON;
        for (int i = 0; i < values.length; i++) {
            values[i] = (values[i] - min) / range;
        }
    }

    /// sampleWithTemperature
    /// 
    /// Takes a array and samples an index from it using softmax and a temperature
    /// 
    /// @param logits raw values that will be interpreted as logits and via softmax
    /// used as probabilities.
    /// @param temperature the temperature for the softmax function. The higher the more uniform
    /// @param rand Random object for sample
    public static int sampleWithTemperature(float[] logits, double temperature, Random rand) {
        int len = logits.length;
        double[] expValues = new double[len];
        double sum = 0.0;

        // build pdf
        for (int i = 0; i < len; i++) {
            double t = Math.max(temperature, 0.00001);
            expValues[i] = Math.exp(logits[i] / t);
            sum += expValues[i];
        }

        double[] targetPdf = new double[len];
        for (int i = 0; i < len; i++) {
            targetPdf[i] = expValues[i] / sum;
        }

        // sample from pdf
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
}
