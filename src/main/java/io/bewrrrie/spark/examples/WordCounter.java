package io.bewrrrie.spark.examples;

import java.util.Arrays;
import java.util.Iterator;
import java.util.Map;
import org.apache.spark.SparkContext;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.function.FlatMapFunction;
import org.apache.spark.api.java.function.Function2;
import org.apache.spark.api.java.function.PairFunction;
import scala.Tuple2;

/**
 * Counts word in text file.
 */
public class WordCounter {
	public static Map<String, Integer> countWord(final SparkContext sc, final String filePath) {
		final JavaRDD<String> lines = sc.textFile(filePath, 1).toJavaRDD();

		final JavaRDD<String> words = lines.flatMap(new FlatMapFunction<String, String>() {
			@Override
			public Iterator<String> call(final String s) throws Exception {
				return Arrays.asList(s.split(" ")).iterator();
			}
		});
		final JavaPairRDD<String, Integer> ones = words.mapToPair(new PairFunction<String, String, Integer>() {
			@Override
			public Tuple2<String, Integer> call(final String s) throws Exception {
				return new Tuple2<>(s, 1);
			}
		});
		final JavaPairRDD<String, Integer> counts = ones.reduceByKey(new Function2<Integer, Integer, Integer>() {
            @Override
            public Integer call(final Integer x, final Integer y) throws Exception {
                return x + y;
            }
        });

		return counts.collectAsMap();
	}
}
