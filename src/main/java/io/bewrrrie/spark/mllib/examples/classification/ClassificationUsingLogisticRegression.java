package io.bewrrrie.spark.mllib.examples.classification;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.VoidFunction;
import org.apache.spark.mllib.classification.LogisticRegressionModel;
import org.apache.spark.mllib.classification.LogisticRegressionWithLBFGS;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.regression.LabeledPoint;

import java.util.Objects;

/**
 * First classification example.
 */
public class ClassificationUsingLogisticRegression {

    private static final String APP_NAME = "Classification/LogisticRegression";
    private static final String MASTER_URL = "local";

    private static final int NUMBER_OF_CLASSES = 7;
    private static final String TRAINING_DATA_PATH = "src/main/resources/data/train.csv";
    private static final String TEST_DATA_PATH = "src/main/resources/data/test.csv";

    
    public static void main(String[] args) {
        final SparkConf conf = new SparkConf();
        conf.setAppName(APP_NAME);
        conf.setMaster(MASTER_URL);
        final JavaSparkContext context = new JavaSparkContext(conf);

        //Put data to operative memory, supposed that there is no missing data in data set.
        //All missing data become 0.0 (zero).
        final JavaRDD<LabeledPoint> data = getData(context, TRAINING_DATA_PATH);

        final LogisticRegressionModel model = new LogisticRegressionWithLBFGS()
            .setNumClasses(NUMBER_OF_CLASSES)
            .run(data.rdd());

        data.foreach((VoidFunction<LabeledPoint>) s -> System.out.println(s));
    }


    private static JavaRDD<LabeledPoint> getData(JavaSparkContext context, String path) {
        return context.textFile(path).map((Function<String, LabeledPoint>) line -> {
            if (!line.contains("x")) {
                String[] words = line.split(",");

                double[] values = new double[words.length - 1];
                for (int i = 0; i < words.length - 1; i++) {
                    values[i] = words[i].length() == 0 ? 0 : (
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
