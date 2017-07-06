package io.bewrrrie.spark.examples;

import org.apache.spark.SparkConf;
import org.apache.spark.SparkContext;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.File;
import java.nio.file.Files;
import java.util.Map;

import static org.junit.Assert.*;

/**
 * Word counter example test class.
 */
public class WordCounterTest {

	@Rule
	public TemporaryFolder folder = new TemporaryFolder();

	@Test
	public void countWord() throws Exception {
		final SparkConf conf = new SparkConf();
		conf.setAppName("WordCounter");
		conf.setMaster("local");

		final SparkContext context = new SparkContext(conf);

		final File temp = folder.newFile();
		Files.write(temp.toPath(), "some sort of some text\n".getBytes());

		final Map<String, Integer> result = WordCounter.countWord(context, temp.getAbsolutePath());
		assertEquals(result.get("some"), new Integer(2));
	}
}