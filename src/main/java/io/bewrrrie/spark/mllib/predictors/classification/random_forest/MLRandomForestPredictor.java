package io.bewrrrie.spark.mllib.predictors.classification.random_forest;

import java.nio.file.Path;

import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.tree.model.RandomForestModel;
import org.apache.spark.mllib.linalg.Vectors;

/**
 * Predictor class.
 * It is able to load model
 * and predict class according to given features array.
 */
public class MLRandomForestPredictor {
    private RandomForestModel model;
    private JavaSparkContext jsc;

    public static MLRandomForestPredictor load(final JavaSparkContext jsc, final Path file) {
        return new MLRandomForestPredictor(jsc, file);
    }

    private MLRandomForestPredictor(JavaSparkContext jsc, Path file) {
        this.jsc = jsc;
        model = RandomForestModel.load(jsc.sc(), file.toString());
    }

    public double value(final double[] x) {
        return model.predict(Vectors.dense(x));
    }

    /*
    public static void main(String[] args) {
        final SparkConf conf = new SparkConf();
        conf.setAppName("Classification/RandomForest");
        conf.setMaster("local");
        final JavaSparkContext jsc = new JavaSparkContext(conf);

        final MLRandomForestPredictor predictor = load(jsc, Paths.get("src/main/resources/models/myRandomForestClassificationModel"));

        final double[][] x = new double[1000][62];
        for (int i = 0; i < x.length; i++) {
            for (int j = 0; j < x[i].length; j++) {
                x[i][j] = Math.random();
            }
        }

        double s = 0;
        for (int i = 0; i < 1000000; i++) {
            s += predictor.value(x[i % x.length]);
        }

        final long t  = System.nanoTime();
        final int n = 1000000;
        for (int i = 0; i < n; i++) {
            s += predictor.value(x[i % x.length]);
        }
        System.out.println("qps = " + n / (1e-9 * (System.nanoTime() - t)));
        System.out.println(s);
    }
    */
}
