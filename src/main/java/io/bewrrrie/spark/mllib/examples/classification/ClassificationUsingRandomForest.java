package io.bewrrrie.spark.mllib.examples.classification;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaSparkContext;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.PairFunction;
import org.apache.spark.api.java.function.VoidFunction;
import org.apache.spark.api.java.function.VoidFunction2;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.tree.RandomForest;
import org.apache.spark.mllib.tree.model.RandomForestModel;
import scala.Tuple2;

import java.util.HashMap;
import java.util.Objects;


/**
 * First classification example.
 */
public class ClassificationUsingRandomForest {

    private static final String APP_NAME = "Classification/LogisticRegression";
    private static final String MASTER_URL = "local";

    private static final String TRAINING_DATA_PATH = "src/main/resources/data/train.csv";
    private static final String TEST_DATA_PATH = "src/main/resources/data/test.csv";

    private static final int NUMBER_OF_CLASSES = 7;
    private static final int NUMBER_OF_TREES = 3;
    private static final int MAX_DEPTH = 5;
    private static final int MAX_BINS = 32;
    private static final int SEED = 5;


    public static void main(String[] args) {
        final SparkConf conf = new SparkConf();
        conf.setAppName(APP_NAME);
        conf.setMaster(MASTER_URL);
        final JavaSparkContext jsc = new JavaSparkContext(conf);

        // Load csv.
        JavaRDD<LabeledPoint> data = getData(jsc, TRAINING_DATA_PATH);

        //Split data.
        JavaRDD<LabeledPoint>[] splits = data.randomSplit(new double[]{0.7, 0.3});
        JavaRDD<LabeledPoint> trainingData = splits[0];
        JavaRDD<LabeledPoint> testData = splits[1];

        // Train a RandomForest model.
        // Empty categoricalFeaturesInfo indicates all features are continuous.
        HashMap<Integer, Integer> categoricalFeaturesInfo = new HashMap<>();
        String featureSubsetStrategy = "auto";
        String impurity = "gini";

        final RandomForestModel model = RandomForest.trainClassifier(
            trainingData,
            NUMBER_OF_CLASSES,
            categoricalFeaturesInfo,
            NUMBER_OF_TREES,
            featureSubsetStrategy,
            impurity,
            MAX_DEPTH,
            MAX_BINS,
            SEED
        );


        // Evaluate model on test instances and compute test error
        JavaRDD<Tuple2<Double, Double>> predictionAndLabel = testData.map(
            (Function<LabeledPoint, Tuple2<Double, Double>>) p -> new Tuple2<>(
                model.predict(p.features()),
                p.label()
            )
        );

        Double testError = 1.0 * predictionAndLabel.filter(
            (Function<Tuple2<Double, Double>, Boolean>) pl -> !pl._1().equals(pl._2())
        ).count() / testData.count();
        System.out.println("Test Error: " + testError);
        System.out.println("Learned classification forest model:\n" + model.toDebugString());

        // Save model
        //model.save(jsc.sc(), "src/main/resources/models/myRandomForestClassificationModel");
    }


    private static JavaRDD<LabeledPoint> getData(JavaSparkContext context, String path) {
        return context.textFile(path).map((Function<String, LabeledPoint>) line -> {
            if (!line.contains("x")) {
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
            }

            return null;
        }).filter((Function<LabeledPoint, Boolean>) Objects::nonNull);
    }
}
