package io.bewrrrie.spark.mllib.data;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.regression.LabeledPoint;

import java.util.Objects;

/**
 * Data extraction utility class.
 */
public class DataExtractor {

    /**
     * Extracts data from text file.
     * This method provides all strings converted to integers.
     * It is supposed that:
     *  1) every string consists of only hexadecimal integer
     *     (it will be converted to decimal while reading);
     *  2) every line that has string feature contains letter 'x' will not appear in result RDD.
     *
     * @param context - Java Spark environment context object.
     * @param path - path to text file with data.
     * @return Spark JavaRDD that contains LabelPoint created with information from specified file.
     */
    public static JavaRDD<LabeledPoint> getHexadecimalData(JavaSparkContext context, String path) {
        return context.textFile(path).map((Function<String, LabeledPoint>) line -> {
            if (line.startsWith("x")) {
                return null;
            }

            String[] words = line.split(",");

            double[] values = new double[words.length - 1];
            for (int i = 0; i < words.length - 1; i++) {
                values[i] = words[i].length() == 0 ? Double.NaN : (
                    words[i].contains(".") ?
                        Double.parseDouble(words[i]) : Long.parseLong(words[i], 16)
                );
            }

            return new LabeledPoint(
                Double.parseDouble(words[words.length - 1]),
                Vectors.dense(values)
            );

        }).filter((Function<LabeledPoint, Boolean>) Objects::nonNull);
    }

    public static JavaRDD<LabeledPoint> getHexadecimalBinaryData(JavaSparkContext context, double[] positiveClassLabels, String path) {
        return context.textFile(path).map((Function<String, LabeledPoint>) line -> {
            if (line.startsWith("x")) {
                return null;
            }

            String[] words = line.split(",");

            double[] values = new double[words.length - 1];
            for (int i = 0; i < words.length - 1; i++) {
                values[i] = words[i].length() == 0 ? Double.NaN : (
                    words[i].contains(".") ?
                        Double.parseDouble(words[i]) : Long.parseLong(words[i], 16)
                );
            }

            for (double d : positiveClassLabels) {
                if (d == Double.parseDouble(words[words.length - 1])) {
                    return new LabeledPoint(1, Vectors.dense(values));
                }
            }
            return new LabeledPoint(0, Vectors.dense(values));

        }).filter((Function<LabeledPoint, Boolean>) Objects::nonNull);
    }
}
