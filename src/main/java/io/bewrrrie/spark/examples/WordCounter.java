package io.bewrrrie.spark.examples;

import java.util.Arrays;
import java.util.Map;
import org.apache.spark.SparkContext;
import org.apache.spark.api.java.function.FlatMapFunction;
import org.apache.spark.api.java.function.Function2;
import org.apache.spark.api.java.function.PairFunction;
import scala.Tuple2;

/**
 * Counts word in text file.
 */
public final class WordCounter {
	public static Map<String, Integer> countWord(final SparkContext sc, final String filePath) {
		return sc.textFile(filePath, 1).toJavaRDD()
            .flatMap((FlatMapFunction<String, String>) s -> Arrays.asList(s.split(" ")).iterator())
            .mapToPair((PairFunction<String, String, Integer>) s -> new Tuple2<>(s, 1))
            .reduceByKey((Function2<Integer, Integer, Integer>) (x, y) -> x + y)
            .collectAsMap();
	}
}
