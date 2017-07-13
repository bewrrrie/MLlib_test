package io.bewrrrie.spark.mllib.predictors.classification.random_forest;

import java.nio.file.Path;

import org.apache.spark.SparkContext;
import org.apache.spark.mllib.tree.model.RandomForestModel;
import org.apache.spark.mllib.linalg.Vectors;

/**
 * Predictor class.
 * It is able to load model and predict class according to given features array.
 */
public class MLRandomForestPredictor {

    private RandomForestModel model;

    public static MLRandomForestPredictor load(final SparkContext sc, final Path file) {
        return new MLRandomForestPredictor(sc, file);
    }

    private MLRandomForestPredictor(final SparkContext sc, final Path file) {
        model = RandomForestModel.load(sc, file.toString());
    }

    public double predict(final double[] x) {
        return model.predict(Vectors.dense(x));
    }
}
