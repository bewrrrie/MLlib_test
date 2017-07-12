package io.bewrrrie.spark.mllib.predictors.classification.random_forest;

import io.bewrrrie.spark.mllib.data.DataExtractor;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.tree.RandomForest;
import org.apache.spark.mllib.tree.model.RandomForestModel;
import scala.Tuple2;

import java.util.HashMap;


/**
 * Fit and save random forest classification model.
 */
public class ClassificationUsingRandomForest {

    private static final String APP_NAME = "Classification/RandomForest";
    private static final String MASTER_URL = "local";
    private static final String DATA_PATH = "src/main/resources/data/train.csv";

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
        JavaRDD<LabeledPoint> data = DataExtractor.getHexadecimalData(jsc, DATA_PATH);

        //Split data.
        JavaRDD<LabeledPoint>[] splits = data.randomSplit(new double[]{0.7, 0.3});
        JavaRDD<LabeledPoint> trainingData = splits[0];
        JavaRDD<LabeledPoint> testData = splits[1];

        // Train a RandomForest model.
        // Empty categoricalFeaturesInfo indicates all features are continuous.
        HashMap<Integer, Integer> categoricalFeaturesInfo = new HashMap<>();

        final RandomForestModel model = RandomForest.trainClassifier(
            trainingData,
            NUMBER_OF_CLASSES,
            categoricalFeaturesInfo,
            NUMBER_OF_TREES,
            "auto",
            "gini",
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
        model.save(jsc.sc(), "src/main/resources/models/myRandomForestClassificationModel");
    }
}
