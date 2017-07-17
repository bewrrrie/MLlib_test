package io.bewrrrie.spark.mllib.predictors.classification.gradient_boosting;

import org.apache.spark.SparkConf;
import org.apache.spark.SparkContext;
import org.junit.Test;

import java.nio.file.Paths;

import static org.junit.Assert.*;

/**
 * Gradient boosting predictor test.
 */
public class MLGradientBoostingPredictorTest {

    @Test
    public void testMultithreadedPrediction() throws Exception {
        // Creating Spark environment.
        final SparkConf conf = new SparkConf()
            .setAppName("test/MultithreadedGradientBoostingPrediction")
            .setMaster("local");
        final SparkContext sc = new SparkContext(conf);

        // Loading model.
        final MLGradientBoostingPredictor model = MLGradientBoostingPredictor.load(
            sc,
            Paths.get("src/main/resources/models/gradientBoostingClassifier")
        );

        // Creating a lot of threads.
        final Thread[] threads = new Thread[10000];

        for (int i = 0; i < threads.length; i++) {
            // Generating random features vector.
            double[] arr = new double[62];
            for (int j = 0; j < 62; j++) {
                arr[j] = Math.random();
            }

            // Needed for using current index inside anonymous class.
            final int k = i;

            // Creating thread object.
            threads[k] = new Thread(new Runnable() {
                private double[] features = arr;

                @Override
                public void run() {
                    try {
                        // Wait for 1 - 250 milliseconds.
                        Thread.sleep((long) (Math.random() * 249 + 1));

                        // Call predict(..) method.
                        Double prediction = model.predict(features);
                        System.out.println(k + "th thread prediction: " + prediction);
                    } catch(InterruptedException e) {
                        // Fail test when caught exception.
                        fail();
                    }
                }
            });
        }

        //Starting all threads!
        for (Thread t : threads) {
            t.start();
        }
        Thread.sleep(2000);


        //Asserting if there wasn't any exception.
        assertTrue(true);
    }
}