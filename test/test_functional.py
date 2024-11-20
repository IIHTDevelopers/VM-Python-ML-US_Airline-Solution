import unittest
from usairline import (
    load_data, preprocess_data, split_data, vectorize_data, train_and_save_model,
    predict_sentiments, sentiment_distribution, most_active_airline,
    percentage_negative_tweets, top_airlines_by_positive_sentiment,
    frequent_bigrams, average_tweet_length, most_negative_airline,
    airline_with_highest_positive_ratio, most_common_trigrams,
    top_airlines_by_neutral_sentiment, most_frequent_single_words
)


class TestUtils:
    @staticmethod
    def yakshaAssert(test_name, condition, test_category):
        print(f"{test_name} = {'Passed' if condition else 'Failed'}")


class TestSentimentAnalysis(unittest.TestCase):
    def setUp(self):
        """
        Set up an intentionally invalid or empty environment to ensure tests fail.
        """
        try:
            # Empty setup with None and empty DataFrame to simulate failures
            self.df = None
            self.tfidf = None
            self.model = None
        except Exception as e:
            self.fail(f"SetUp Failed with exception: {e}")

    def test_sentiment_distribution(self):
        test_obj = TestUtils()
        expected = {'negative': 10241, 'neutral': 2526, 'positive': 1873}
        try:
            actual = sentiment_distribution(self.df)
            test_obj.yakshaAssert("TestSentimentDistribution", expected == actual, "functional")
            self.assertEqual(expected, actual)
        except Exception as e:
            test_obj.yakshaAssert("TestSentimentDistribution", False, "functional")
            print("TestSentimentDistribution = Failed")

    def test_most_active_airline(self):
        test_obj = TestUtils()
        expected = "United"
        try:
            actual = most_active_airline(self.df)
            test_obj.yakshaAssert("TestMostActiveAirline", expected == actual, "functional")
            self.assertEqual(expected, actual)
        except Exception as e:
            test_obj.yakshaAssert("TestMostActiveAirline", False, "functional")
            print("TestMostActiveAirline = Failed")

    def test_percentage_negative_tweets(self):
        test_obj = TestUtils()
        expected = 69.95
        try:
            actual = percentage_negative_tweets(self.df)
            test_obj.yakshaAssert("TestPercentageNegativeTweets", expected == actual, "functional")
            self.assertEqual(expected, actual)
        except Exception as e:
            test_obj.yakshaAssert("TestPercentageNegativeTweets", False, "functional")
            print("TestPercentageNegativeTweets = Failed")

    def test_top_airlines_by_positive_sentiment(self):
        test_obj = TestUtils()
        expected = {'Southwest': 448, 'Delta': 416, 'United': 385, 'American': 277, 'US Airways': 239, 'Virgin America': 108}
        try:
            actual = top_airlines_by_positive_sentiment(self.df)
            test_obj.yakshaAssert("TestTopAirlinesByPositiveSentiment", expected == actual, "functional")
            self.assertEqual(expected, actual)
        except Exception as e:
            test_obj.yakshaAssert("TestTopAirlinesByPositiveSentiment", False, "functional")
            print("TestTopAirlinesByPositiveSentiment = Failed")

    def test_frequent_bigrams_positive_sentiment(self):
        test_obj = TestUtils()
        expected = [('thank you', 448), ('thanks for', 221), ('for the', 164), ('you for', 113), ('you guys', 84), ('customer service', 75), ('so much', 74), ('the best', 69), ('in the', 50), ('to the', 49)]
        try:
            actual = frequent_bigrams(self.df, "positive")
            test_obj.yakshaAssert("TestFrequentBigramsPositiveSentiment", expected == actual, "functional")
            self.assertEqual(expected, actual)
        except Exception as e:
            test_obj.yakshaAssert("TestFrequentBigramsPositiveSentiment", False, "functional")
            print("TestFrequentBigramsPositiveSentiment = Failed")

    def test_average_tweet_length(self):
        test_obj = TestUtils()
        expected = 16.06
        try:
            actual = average_tweet_length(self.df)
            test_obj.yakshaAssert("TestAverageTweetLength", expected == actual, "functional")
            self.assertEqual(expected, actual)
        except Exception as e:
            test_obj.yakshaAssert("TestAverageTweetLength", False, "functional")
            print("TestAverageTweetLength = Failed")

    def test_most_negative_airline(self):
        test_obj = TestUtils()
        expected = ("US Airways", 82.18)
        try:
            actual = most_negative_airline(self.df)
            test_obj.yakshaAssert("TestMostNegativeAirline", expected == actual, "functional")
            self.assertEqual(expected, actual)
        except Exception as e:
            test_obj.yakshaAssert("TestMostNegativeAirline", False, "functional")
            print("TestMostNegativeAirline = Failed")

    def test_airline_highest_positive_ratio(self):
        test_obj = TestUtils()
        expected = ("Virgin America", 21.43)
        try:
            actual = airline_with_highest_positive_ratio(self.df)
            test_obj.yakshaAssert("TestAirlineHighestPositiveRatio", expected == actual, "functional")
            self.assertEqual(expected, actual)
        except Exception as e:
            test_obj.yakshaAssert("TestAirlineHighestPositiveRatio", False, "functional")
            print("TestAirlineHighestPositiveRatio = Failed")

    def test_most_common_trigrams(self):
        test_obj = TestUtils()
        expected = [('on hold for', 238), ('been on hold', 167), ('thank you for', 147), ('fleets on fleek', 145), ('our fleets on', 145), ('thanks for the', 144), ('was cancelled flightled', 105), ('on the phone', 94), ('flight booking problems', 89), ('on the plane', 81)]
        try:
            actual = most_common_trigrams(self.df)
            test_obj.yakshaAssert("TestMostCommonTrigrams", expected == actual, "functional")
            self.assertEqual(expected, actual)
        except Exception as e:
            test_obj.yakshaAssert("TestMostCommonTrigrams", False, "functional")
            print("TestMostCommonTrigrams = Failed")

    def test_top_airlines_by_neutral_sentiment(self):
        test_obj = TestUtils()
        expected = {'Delta': 624, 'Southwest': 587, 'United': 537, 'American': 346, 'US Airways': 280, 'Virgin America': 152}
        try:
            actual = top_airlines_by_neutral_sentiment(self.df)
            test_obj.yakshaAssert("TestTopAirlinesByNeutralSentiment", expected == actual, "functional")
            self.assertEqual(expected, actual)
        except Exception as e:
            test_obj.yakshaAssert("TestTopAirlinesByNeutralSentiment", False, "functional")
            print("TestTopAirlinesByNeutralSentiment = Failed")

    def test_most_frequent_words_negative_sentiment(self):
        test_obj = TestUtils()
        expected = [('to', 6672), ('the', 4637), ('i', 4030), ('a', 3497), ('flight', 3156), ('and', 3128), ('on', 3032), ('for', 2940), ('you', 2755), ('my', 2655)]
        try:
            actual = most_frequent_single_words(self.df, "negative")
            test_obj.yakshaAssert("TestMostFrequentWordsNegativeSentiment", expected == actual, "functional")
            self.assertEqual(expected, actual)
        except Exception as e:
            test_obj.yakshaAssert("TestMostFrequentWordsNegativeSentiment", False, "functional")
            print("TestMostFrequentWordsNegativeSentiment = Failed")


if __name__ == "__main__":
    unittest.main()
