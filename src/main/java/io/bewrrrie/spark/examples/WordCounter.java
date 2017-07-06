package io.bewrrrie.spark.examples;

import java.util.HashMap;
import java.util.Map;
import org.apache.spark.SparkContext;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.function.VoidFunction;

/**
 * Counts word in text file.
 */
public class WordCounter {
	public static Map<String, Integer> countWord(final SparkContext sc, final String filePath) {
		final JavaRDD<String> rdd = sc.textFile(filePath, 1).toJavaRDD();

		rdd.foreach(new VoidFunction<String>() {
			@Override
			public void call(String s) throws Exception {
				System.out.println(s);
			}
		});

		/*have not done yet!*/
		return new HashMap<>();
	}
}
