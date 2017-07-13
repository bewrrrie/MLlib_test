package io.bewrrrie.spark.mllib.predictors.classification.gradient_boosting;

import org.apache.spark.SparkConf;
import org.apache.spark.SparkContext;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.tree.model.GradientBoostedTreesModel;

import java.nio.file.Path;
import java.nio.file.Paths;

/**
 * Predictor class.
 * It is able to load model and predict class according to given features array.
 */
public class MLGradientBoostingPredictor {

    private GradientBoostedTreesModel model;

    public static MLGradientBoostingPredictor load(final SparkContext sc, final Path file) {
        return new MLGradientBoostingPredictor(sc, file);
    }

    private MLGradientBoostingPredictor(final SparkContext sc, final Path file) {
        model = GradientBoostedTreesModel.load(sc, file.toString());
    }

    public double predict(final double[] x) {
        return model.predict(Vectors.dense(x));
    }


    public static void main(String[] args) {
        final SparkConf conf = new SparkConf();
        conf.setAppName("Classification/GradientBoostedTrees");
        conf.setMaster("local");
        final SparkContext sc = new  SparkContext(conf);

        final MLGradientBoostingPredictor predictor = load(sc, Paths.get("src/main/resources/models/gradientBoostingClassifier"));


        // todo Градиентный бустинг из решающих деревьев.
        // todo Посчитать qps: число запросов в секунду для большого леса.
        // todo Изучить вопрос о потокобезопасности.


        final double[][] x = new double[1000][62];
        for (int i = 0; i < x.length; i++) {
            for (int j = 0; j < x[i].length; j++) {
                x[i][j] = Math.random();
            }
        }

        double s = 0;
        for (int i = 0; i < 1000000; i++) {
            s += predictor.predict(x[i % x.length]);
        }

        final long t  = System.nanoTime();
        final int n = 1000000;
        for (int i = 0; i < n; i++) {
            s += predictor.predict(x[i % x.length]);
        }
        System.out.println( "qps = " + (n / (1e-9 * (System.nanoTime() - t))) );
        System.out.println(s);
        // ~246689.8139468778
    }
}
