package io.bewrrrie.spark.mllib.predictors.classification.gradient_boosting;

import java.util.HashMap;
import java.util.Map;

import io.bewrrrie.spark.mllib.data.DataExtractor;
import scala.Tuple2;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.tree.GradientBoostedTrees;
import org.apache.spark.mllib.tree.configuration.BoostingStrategy;
import org.apache.spark.mllib.tree.model.GradientBoostedTreesModel;
import org.apache.spark.mllib.util.MLUtils;

/**
 * Fit decision trees composition using gradient boosting
 * and save it to file.
 */
public class ClassificationUsingDecisionTreesBoosting {

    private static final String APP_NAME = "Classification/GradientBoostedTrees";
    private static final String MASTER_URL = "local";
    private static final String TRAINING_DATA_PATH = "src/main/resources/data/train.csv";
    private static final String TEST_DATA_PATH = "src/main/resources/data/test.csv";

    private static final String MODEL_STORAGE_PATH = "src/main/resources/models/myGradientBoostingClassificationModel";

    private static final int ITERATIONS = 1000;
    private static final int CLASSES = 2;
    private static final int MAX_DEPTH = 30;

    private static final double[] POSITIVE_CLASSES = new double[] {0, 1, 2};


    // todo Градиентный бустинг из решающих деревьев.
    // todo Посчитать qps: число запросов в секунду для большого леса.
    // todo Изучить вопрос о потокобезопасности.

    public static void main(String[] args) {
        final SparkConf sc = new SparkConf();
        sc.setAppName(APP_NAME);
        sc.setMaster(MASTER_URL);
        final JavaSparkContext jsc = new JavaSparkContext(sc);

        // Load and parse the data file.
        JavaRDD<LabeledPoint> trainingData = DataExtractor.getHexadecimalBinaryData(
            jsc,
            POSITIVE_CLASSES,
            TRAINING_DATA_PATH
        );
        // Split the data into training and test sets (30% held out for testing).
        JavaRDD<LabeledPoint> testData = DataExtractor.getHexadecimalBinaryData(
            jsc,
            POSITIVE_CLASSES,
            TEST_DATA_PATH
        );

        // Train a GradientBoostedTrees model.
        // The defaultParams for Classification use LogLoss by default.
        BoostingStrategy boostingStrategy = BoostingStrategy.defaultParams("Classification");
        boostingStrategy.setNumIterations(ITERATIONS);
        boostingStrategy.getTreeStrategy().setNumClasses(CLASSES);
        boostingStrategy.getTreeStrategy().setMaxDepth(MAX_DEPTH);

        // Empty categoricalFeaturesInfo indicates all features are continuous.
        Map<Integer, Integer> categoricalFeaturesInfo = new HashMap<>();
        boostingStrategy.treeStrategy().setCategoricalFeaturesInfo(categoricalFeaturesInfo);

        GradientBoostedTreesModel model = GradientBoostedTrees.train(trainingData, boostingStrategy);

        // Evaluate model on test instances and compute test error
        JavaPairRDD<Double, Double> predictionAndLabel = testData.mapToPair(
            p -> new Tuple2<>(model.predict(p.features()), p.label())
        );

        double testErr = predictionAndLabel.filter(
            pl -> !pl._1().equals(pl._2())
        ).count() / (double) testData.count();

        System.out.println("Test Error: " + testErr);
        System.out.println("Learned classification GBT model:\n" + model.toDebugString());

        // Save model
        model.save(jsc.sc(), MODEL_STORAGE_PATH);
    }
}
